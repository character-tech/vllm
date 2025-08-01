# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import math
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import Optional, Union, cast

import jinja2
import torch
from fastapi import Request
from typing_extensions import assert_never

from vllm.beam.beam import BeamScorer
from vllm.beam.filtering import _CHUNK_SIZE, BeamValidator
from vllm.beam.metrics import report_metrics
from vllm.beam.penalty import MEOW_CLASSI_IDX, PenaltyComputer
from vllm.beam.tracing import trace_streaming_completion, trace_async_method
from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.logger import RequestLogger
# yapf conflicts with isort for this block
# yapf: disable
from vllm.entrypoints.openai.protocol import (CompletionLogProbs,
                                              CompletionRequest,
                                              CompletionResponse,
                                              CompletionResponseChoice,
                                              CompletionResponseStreamChoice,
                                              CompletionStreamResponse,
                                              ErrorResponse,
                                              PromptTokenUsageInfo,
                                              RequestResponseMetadata,
                                              UsageInfo)
from vllm.entrypoints.openai.serving_engine import (
    EmbedsPrompt as ServingEngineEmbedsPrompt)
from vllm.entrypoints.openai.serving_engine import (OpenAIServing,
                                                    TextTokensPrompt,
                                                    clamp_prompt_logprobs,
                                                    is_text_tokens_prompt)
# yapf: enable
from vllm.entrypoints.openai.serving_models import OpenAIServingModels
from vllm.entrypoints.utils import get_max_tokens
from vllm.inputs.data import (EmbedsPrompt, TokensPrompt, is_embeds_prompt,
                              is_tokens_prompt)
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.sequence import Logprob
from vllm.tracing import init_tracer
from vllm.transformers_utils.tokenizer import AnyTokenizer
from vllm.utils import merge_async_iterators, random_uuid

logger = init_logger(__name__)


class OpenAIServingCompletion(OpenAIServing):

    def __init__(
        self,
        engine_client: EngineClient,
        model_config: ModelConfig,
        models: OpenAIServingModels,
        *,
        request_logger: Optional[RequestLogger],
        return_tokens_as_token_ids: bool = False,
        enable_force_include_usage: bool = False,
        enable_prompt_tokens_details: bool = False,
    ):
        super().__init__(engine_client=engine_client,
                         model_config=model_config,
                         models=models,
                         request_logger=request_logger,
                         return_tokens_as_token_ids=return_tokens_as_token_ids,
                         enable_force_include_usage=enable_force_include_usage)
        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.default_sampling_params = (
            self.model_config.get_diff_sampling_param())
        if self.default_sampling_params:
            source = self.model_config.generation_config
            source = "model" if source == "auto" else source
            logger.info("Using default completion sampling params from %s: %s",
                        source, self.default_sampling_params)

        if self.model_config.has_additional_heads:
            self.beam_scorer = BeamScorer(classi_idx=MEOW_CLASSI_IDX)
            self.beam_validator = BeamValidator(classi_idx=MEOW_CLASSI_IDX, classifier_names=MEOW_CLASSI_IDX.keys())

    @trace_streaming_completion()
    async def create_completion_with_chunkwise_beam(
        self,
        request: CompletionRequest,
        raw_request: Optional[Request] = None,
) -> Union[AsyncGenerator[str, None], CompletionResponse, ErrorResponse]:
        """
    Chunkwise beam search hack
    """
        # set request.arrival_time to current time for all chunks
        request.arrival_time = time.time()
        @trace_async_method(span_name='_process_prefix')
        async def _process_prefix(request: CompletionRequest):
            og_max_tokens = request.max_tokens
            og_n = request.n
            request.max_tokens = 0
            request.n = 1
            request.echo = True
            request.stream = False
            res = await self.create_completion(
            request,
            raw_request=raw_request,
        )
            request.max_tokens = og_max_tokens
            request.n = og_n
            request.echo = False
            request.stream = True
            return res

        if not self.model_config.has_additional_heads:
            return self.create_error_response(
                "Chunkwise beam search is not supported for this model")

        res = await _process_prefix(request)
        if isinstance(res, ErrorResponse):
            return res

        candidate_id = f"cmpl-{random_uuid()}"
        input_str_len = len(res.choices[0].text)

        async def _should_stop(final):
            return final.finish_reason == "stop" or final.is_filtered
        
        max_chunks = math.ceil(request.max_tokens / _CHUNK_SIZE)
        async def _chunk_generator():
            num_chunks = 0
            should_stop = False
            output = None

            # TODO(@tanuj): calc created tokens
            while num_chunks < max_chunks and not should_stop:
                num_chunks += 1
                beams = await self.beam_validator.get_n_valid_beams(create_completion=self.create_completion, request=request, raw_request=raw_request, chunk_num=num_chunks)
                beams.id = candidate_id
                if isinstance(beams, ErrorResponse):
                    yield f"data: {beams.model_dump_json()}\n\n"
                    break
            
                final = await self.beam_scorer.pick_best_beam(beams.choices)
                request.prompt = final.text
                should_stop = await _should_stop(final)
                final.text = final.text[input_str_len:]
                output = final.text
                beams.choices = [final]
                if self.request_logger:
                    logger.info(f"yielding chunk {num_chunks} text: {final.text}")
                yield f"data: {beams.model_dump_json()}\n\n"
            
                if should_stop:
                    break
        
            yield "data: [DONE]\n\n"

            report_metrics(request, output, beams)
    
        return _chunk_generator()

    async def create_completion(
        self,
        request: CompletionRequest,
        raw_request: Optional[Request] = None,
    ) -> Union[AsyncGenerator[str, None], CompletionResponse, ErrorResponse]:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/completions/create
        for the API specification. This API mimics the OpenAI Completion API.

        NOTE: Currently we do not support the following feature:
            - suffix (the language models we currently support do not support
            suffix)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        # Return error for unsupported features.
        if request.suffix is not None:
            return self.create_error_response(
                "suffix is not currently supported")

        if request.echo and request.prompt_embeds is not None:
            return self.create_error_response(
                "Echo is unsupported with prompt embeds.")

        request_id = f"cmpl-{self._base_request_id(raw_request)}"
        created_time = int(time.time())

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            request_prompts, engine_prompts = await self._preprocess_completion(
                request,
                tokenizer,
                request.prompt,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
                add_special_tokens=request.add_special_tokens,
            )
        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except TypeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except RuntimeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except jinja2.TemplateError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                sampling_params: Union[SamplingParams, BeamSearchParams]
                # Mypy does not infer that engine_prompt will have only one of
                # "prompt_token_ids" or "prompt_embeds" defined, and both of
                # these as Union[object, the expected type], where it infers
                # object if engine_prompt is a subclass of one of the
                # typeddicts that defines both keys. Worse, because of
                # https://github.com/python/mypy/issues/8586, mypy does not
                # infer the type of engine_prompt correctly because of the
                # enumerate. So we need an unnecessary cast here.
                engine_prompt = cast(Union[EmbedsPrompt, TokensPrompt],
                                     engine_prompt)
                if is_embeds_prompt(engine_prompt):
                    input_length = len(engine_prompt["prompt_embeds"])
                elif is_tokens_prompt(engine_prompt):
                    input_length = len(engine_prompt["prompt_token_ids"])
                else:
                    assert_never(engine_prompt)

                if self.default_sampling_params is None:
                    self.default_sampling_params = {}

                max_tokens = get_max_tokens(
                    max_model_len=self.max_model_len,
                    request=request,
                    input_length=input_length,
                    default_sampling_params=self.default_sampling_params)

                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        max_tokens, self.default_sampling_params)
                else:
                    sampling_params = request.to_sampling_params(
                        max_tokens, self.model_config.logits_processor_pattern,
                        self.default_sampling_params)

                request_id_item = f"{request_id}-{i}"
                streaming_params = request.to_streaming_params()

                self._log_inputs(request_id_item,
                                 request_prompts[i],
                                 params=sampling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                # Mypy inconsistently requires this second cast in different
                # environments. It shouldn't be necessary (redundant from above)
                # but pre-commit in CI fails without it.
                engine_prompt = cast(Union[EmbedsPrompt, TokensPrompt],
                                     engine_prompt)
                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.engine_client.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                        lora_request=lora_request,
                    )
                else:
                    generator = self.engine_client.generate(
                        engine_prompt,
                        sampling_params,
                        streaming_params,
                        request_id_item,
                        lora_request=lora_request,
                        prompt_adapter_request=prompt_adapter_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                        arrival_time=request.arrival_time,
                    )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator = merge_async_iterators(*generators)

        model_name = self._get_model_name(request.model, lora_request)
        num_prompts = len(engine_prompts)

        # Similar to the OpenAI API, when n != best_of, we do not stream the
        # results. Noting that best_of is only supported in V0. In addition,
        # we do not stream the results when use beam search.
        stream = (request.stream
                  and (request.best_of is None or request.n == request.best_of)
                  and not request.use_beam_search)

        # Streaming response
        if stream:
            return self.completion_stream_generator(
                request,
                request_prompts,
                result_generator,
                request_id,
                created_time,
                model_name,
                num_prompts=num_prompts,
                tokenizer=tokenizer,
                request_metadata=request_metadata,
                enable_force_include_usage=self.enable_force_include_usage)

        # Non-streaming response
        final_res_batch: list[Optional[RequestOutput]] = [None] * num_prompts
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            for i, final_res in enumerate(final_res_batch):
                assert final_res is not None

                # The output should contain the input text
                # We did not pass it into vLLM engine to avoid being redundant
                # with the inputs token IDs
                if final_res.prompt is None:
                    request_prompt = request_prompts[i]
                    if is_text_tokens_prompt(request_prompt):
                        final_res.prompt = request_prompt["prompt"]
                    else:
                        final_res.prompt = None

            final_res_batch_checked = cast(list[RequestOutput],
                                           final_res_batch)

            response = self.request_output_to_completion_response(
                final_res_batch_checked,
                request,
                request_id,
                created_time,
                model_name,
                tokenizer,
                request_metadata,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        if request.stream:
            response_json = response.model_dump_json()

            async def fake_stream_generator() -> AsyncGenerator[str, None]:
                yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

            return fake_stream_generator()

        return response

    async def completion_stream_generator(
        self,
        request: CompletionRequest,
        request_prompts: list[Union[TextTokensPrompt,
                                    ServingEngineEmbedsPrompt]],
        result_generator: AsyncIterator[tuple[int, RequestOutput]],
        request_id: str,
        created_time: int,
        model_name: str,
        num_prompts: int,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
        enable_force_include_usage: bool,
    ) -> AsyncGenerator[str, None]:
        num_choices = 1 if request.n is None else request.n
        previous_text_lens = [0] * num_choices * num_prompts
        previous_num_tokens = [0] * num_choices * num_prompts
        has_echoed = [False] * num_choices * num_prompts
        num_prompt_tokens = [0] * num_prompts
        num_cached_tokens = [0] * num_prompts
        accumulated_text = [""] * num_choices * num_prompts
        accumulated_tokens = [[] * num_choices * num_prompts]
        accumulated_logprobs = [[] * num_choices * num_prompts]

        stream_options = request.stream_options
        if stream_options:
            include_usage = stream_options.include_usage or \
                            enable_force_include_usage
            include_continuous_usage = include_usage and \
                                       stream_options.continuous_usage_stats
        else:
            include_usage, include_continuous_usage = False, False

        chunk = None
        try:
            async for prompt_idx, res in result_generator:
                prompt_token_ids = res.prompt_token_ids
                prompt_logprobs = res.prompt_logprobs
                cached_tokens = res.num_cached_tokens

                if res.prompt is not None:
                    prompt_text = res.prompt
                else:
                    request_prompt = request_prompts[prompt_idx]
                    if is_text_tokens_prompt(request_prompt):
                        prompt_text = request_prompt["prompt"]
                    else:
                        prompt_text = None

                # Prompt details are excluded from later streamed outputs
                if prompt_token_ids is not None:
                    num_prompt_tokens[prompt_idx] = len(prompt_token_ids)

                if cached_tokens is not None:
                    num_cached_tokens[prompt_idx] = cached_tokens

                delta_token_ids: GenericSequence[int]
                out_logprobs: Optional[GenericSequence[Optional[dict[
                    int, Logprob]]]]

                for output in res.outputs:
                    i = output.index + prompt_idx * num_choices

                    assert request.max_tokens is not None
                    if request.echo and not has_echoed[i]:
                        assert prompt_token_ids is not None
                        assert prompt_text is not None
                        if request.max_tokens == 0:
                            # only return the prompt
                            delta_text = prompt_text
                            delta_token_ids = prompt_token_ids
                            out_logprobs = prompt_logprobs
                        else:
                            # echo the prompt and first token
                            delta_text = prompt_text + output.text
                            delta_token_ids = [
                                *prompt_token_ids, *output.token_ids
                            ]
                            out_logprobs = [
                                *(prompt_logprobs or []),
                                *(output.logprobs or []),
                            ]
                        has_echoed[i] = True
                    elif request.accumulate_on_model_server:
                        i = output.index + prompt_idx * num_choices
                        # return the accumulated response
                        accumulated_text[i] += output.text
                        accumulated_tokens[i].extend(output.token_ids)
                        accumulated_logprobs[i].extend(output.logprobs or [])

                        delta_text = accumulated_text[i]
                        delta_token_ids = accumulated_tokens[i]
                        out_logprobs = accumulated_logprobs[i]
                    else:
                        # return just the delta
                        delta_text = output.text
                        delta_token_ids = output.token_ids
                        out_logprobs = output.logprobs

                        if not delta_text and not delta_token_ids \
                            and not previous_num_tokens[i]:
                            # Chunked prefill case, don't return empty chunks
                            continue

                    if request.logprobs is not None:
                        assert out_logprobs is not None, (
                            "Did not output logprobs")
                        logprobs = self._create_completion_logprobs(
                            token_ids=delta_token_ids,
                            top_logprobs=out_logprobs,
                            num_output_top_logprobs=request.logprobs,
                            tokenizer=tokenizer,
                            initial_text_offset=previous_text_lens[i],
                            return_as_token_id=request.
                            return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    previous_text_lens[i] += len(output.text)
                    previous_num_tokens[i] += len(output.token_ids)
                    finish_reason = output.finish_reason
                    stop_reason = output.stop_reason

                    chunk = CompletionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model_name,
                        choices=[
                            CompletionResponseStreamChoice(
                                index=i,
                                text=delta_text,
                                logprobs=logprobs,
                                finish_reason=finish_reason,
                                stop_reason=stop_reason,
                            )
                        ])
                    if include_continuous_usage:
                        prompt_tokens = num_prompt_tokens[prompt_idx]
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=prompt_tokens + completion_tokens,
                        )

                    response_json = chunk.model_dump_json(exclude_unset=False)
                    yield f"data: {response_json}\n\n"

            total_prompt_tokens = sum(num_prompt_tokens)
            total_completion_tokens = sum(previous_num_tokens)
            total_cached_tokens = sum(num_cached_tokens)
            final_usage_info = UsageInfo(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens)
            if self.enable_prompt_tokens_details and total_cached_tokens:
                final_usage_info.prompt_tokens_details = PromptTokenUsageInfo(
                    cached_tokens=total_cached_tokens
                )

            if include_usage:
                final_usage_chunk = CompletionStreamResponse(
                    id=request_id,
                    created=created_time,
                    model=model_name,
                    choices=[],
                    usage=final_usage_info,
                )

                # if accumulate, send the usage info attached to last chunk instead
                if request.accumulate_on_model_server and chunk is not None:
                    chunk.usage = final_usage_info
                    final_usage_chunk = chunk

                final_usage_data = (final_usage_chunk.model_dump_json(
                    exclude_unset=False, exclude_none=True))
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            request_metadata.final_usage_info = final_usage_info

        except Exception as e:
            # TODO: Use a vllm-specific Validation Error
            data = self.create_streaming_error_response(str(e))
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    def request_output_to_completion_response(
        self,
        final_res_batch: list[RequestOutput],
        request: CompletionRequest,
        request_id: str,
        created_time: int,
        model_name: str,
        tokenizer: AnyTokenizer,
        request_metadata: RequestResponseMetadata,
    ) -> CompletionResponse:
        choices: list[CompletionResponseChoice] = []
        num_prompt_tokens = 0
        num_generated_tokens = 0

        for final_res in final_res_batch:
            prompt_token_ids = final_res.prompt_token_ids
            assert prompt_token_ids is not None
            prompt_logprobs = clamp_prompt_logprobs(final_res.prompt_logprobs)
            prompt_text = final_res.prompt

            token_ids: GenericSequence[int]
            out_logprobs: Optional[GenericSequence[Optional[dict[int,
                                                                 Logprob]]]]

            for output in final_res.outputs:
                assert request.max_tokens is not None
                if request.echo:
                    assert prompt_text is not None
                    if request.max_tokens == 0:
                        token_ids = prompt_token_ids
                        out_logprobs = prompt_logprobs
                        output_text = prompt_text
                    else:
                        token_ids = [*prompt_token_ids, *output.token_ids]

                        if request.logprobs is None:
                            out_logprobs = None
                        else:
                            assert prompt_logprobs is not None
                            assert output.logprobs is not None
                            out_logprobs = [
                                *prompt_logprobs,
                                *output.logprobs,
                            ]

                        output_text = prompt_text + output.text
                else:
                    token_ids = output.token_ids
                    out_logprobs = output.logprobs
                    output_text = output.text

                if request.logprobs is not None:
                    assert out_logprobs is not None, "Did not output logprobs"
                    logprobs = self._create_completion_logprobs(
                        token_ids=token_ids,
                        top_logprobs=out_logprobs,
                        tokenizer=tokenizer,
                        num_output_top_logprobs=request.logprobs,
                        return_as_token_id=request.return_tokens_as_token_ids,
                    )
                else:
                    logprobs = None


                choice_data = CompletionResponseChoice(
                    index=len(choices),
                    text=output_text,
                    logprobs=logprobs,
                    finish_reason=output.finish_reason,
                    stop_reason=output.stop_reason,
                    prompt_logprobs=final_res.prompt_logprobs,
                    additional_heads=output.additional_heads,
                )
                choices.append(choice_data)

                num_generated_tokens += len(output.token_ids)

            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        if self.enable_prompt_tokens_details and final_res_batch[0].num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res_batch[0].num_cached_tokens)

        request_metadata.final_usage_info = usage

        return CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            kv_transfer_params=final_res_batch[0].kv_transfer_params)

    def _create_completion_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[Optional[dict[int, Logprob]]],
        num_output_top_logprobs: int,
        tokenizer: AnyTokenizer,
        initial_text_offset: int = 0,
        return_as_token_id: Optional[bool] = None,
    ) -> CompletionLogProbs:
        """Create logprobs for OpenAI Completion API."""
        out_text_offset: list[int] = []
        out_token_logprobs: list[Optional[float]] = []
        out_tokens: list[str] = []
        out_top_logprobs: list[Optional[dict[str, float]]] = []

        last_token_len = 0

        should_return_as_token_id = return_as_token_id if \
            return_as_token_id is not None else self.return_tokens_as_token_ids
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None:
                token = tokenizer.decode(token_id)
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"

                out_tokens.append(token)
                out_token_logprobs.append(None)
                out_top_logprobs.append(None)
            else:
                step_token = step_top_logprobs[token_id]

                token = self._get_decoded_token(
                    step_token,
                    token_id,
                    tokenizer,
                    return_as_token_id=should_return_as_token_id,
                )
                token_logprob = max(step_token.logprob, -9999.0)

                out_tokens.append(token)
                out_token_logprobs.append(token_logprob)

                # makes sure to add the top num_output_top_logprobs + 1
                # logprobs, as defined in the openai API
                # (cf. https://github.com/openai/openai-openapi/blob/
                # 893ba52242dbd5387a97b96444ee1c742cfce9bd/openapi.yaml#L7153)
                out_top_logprobs.append({
                    # Convert float("-inf") to the
                    # JSON-serializable float that OpenAI uses
                    self._get_decoded_token(top_lp[1],
                                            top_lp[0],
                                            tokenizer,
                                            return_as_token_id=should_return_as_token_id):
                    max(top_lp[1].logprob, -9999.0)
                    for i, top_lp in enumerate(step_top_logprobs.items())
                    if num_output_top_logprobs >= i
                })

            if len(out_text_offset) == 0:
                out_text_offset.append(initial_text_offset)
            else:
                out_text_offset.append(out_text_offset[-1] + last_token_len)
            last_token_len = len(token)

        return CompletionLogProbs(
            text_offset=out_text_offset,
            token_logprobs=out_token_logprobs,
            tokens=out_tokens,
            top_logprobs=out_top_logprobs,
        )
