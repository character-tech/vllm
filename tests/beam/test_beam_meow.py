import pytest

from vllm.beam.beam import BeamScorer
from vllm.beam.penalty import MEOW_CLASSI_IDX
from vllm.entrypoints.openai.protocol import CompletionResponse, CompletionResponseChoice, EmbeddingResponse, UsageInfo

@pytest.fixture()
async def meow_random_beams():
    return (
        " Aizawa: You haven't given me your name, age, and quirk",

    )
@pytest.mark.asyncio
async def test_beam_scorer():
    responses = [CompletionResponse(
        choices=[CompletionResponseChoice(text="Hello", index=0, logprobs=None, finish_reason="length", additional_heads=[[10000, 0, 0]],),],
        model="test",
        usage=UsageInfo(),
    ),
        CompletionResponse(
            choices=[CompletionResponseChoice(text="Hello", index=0, logprobs=None, finish_reason="length",
                                              additional_heads=[[-100, 0, 0]], ), ],
            model="test",
            usage=UsageInfo(),
        )
    ]

    scorer = BeamScorer(MEOW_CLASSI_IDX)
    res = await scorer.pick_best_beam(responses)
    assert res == responses[1]
