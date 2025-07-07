import pytest

from vllm.beam.beam import BeamScorer
from vllm.entrypoints.openai.protocol import CompletionResponse, CompletionResponseChoice, EmbeddingResponse, UsageInfo

classi_idx = {
    "annotations_sexually_suggestive": 0,
    "annotations_racist": 1,
}

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

    scorer = BeamScorer(classi_idx)
    res = await scorer.pick_best_beam(responses)
    assert res == responses[1]
