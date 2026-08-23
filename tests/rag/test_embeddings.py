"""验证BGE-M3 Embedding适配器的批量、向量、指纹与deadline边界。"""

from __future__ import annotations

import asyncio
import math

import pytest

from aicare_agent_service.rag.embeddings import BgeM3EmbeddingProvider
from aicare_agent_service.rag.errors import RagError, RagErrorCode
from aicare_agent_service.rag.model_runtime import BgeModelRuntime
from tests.fakes.rag_models import FakeEmbeddingModel, FakeRerankerModel, non_finite_values


def _provider(
    *,
    dimensions: int = 1024,
    invalid_number: float | None = None,
    delay: float = 0,
    max_concurrency: int = 1,
) -> tuple[BgeM3EmbeddingProvider, BgeModelRuntime]:
    runtime = BgeModelRuntime(
        embedding_model=FakeEmbeddingModel(
            dimensions=dimensions,
            invalid_number=invalid_number,
            delay=delay,
        ),
        reranker_model=FakeRerankerModel(),
        max_concurrency=max_concurrency,
        deadline_seconds=1,
    )
    provider = BgeM3EmbeddingProvider(
        runtime=runtime,
        model_id="BAAI/bge-m3",
        revision="a" * 40,
        expected_revision="a" * 40,
        batch_size=2,
    )
    return provider, runtime


@pytest.mark.asyncio
@pytest.mark.parametrize("texts", [(), ("a", "b", "c")])
async def test_embedding_rejects_empty_or_oversized_batch(texts: tuple[str, ...]) -> None:
    provider, runtime = _provider()
    try:
        with pytest.raises(ValueError, match="RAG_EMBEDDING_BATCH_INVALID"):
            await provider.embed_documents(texts, deadline=asyncio.get_running_loop().time() + 1)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_embedding_normalizes_exact_1024_dimension_vectors() -> None:
    provider, runtime = _provider()
    try:
        vectors = await provider.embed_documents(
            ("Steam礼物", "CDK"), deadline=asyncio.get_running_loop().time() + 1
        )
    finally:
        await runtime.close()

    assert len(vectors) == 2
    assert all(len(vector) == 1024 for vector in vectors)
    assert all(
        math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0) for vector in vectors
    )
    assert len(provider.model_fingerprint) == 64


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_number", non_finite_values())
async def test_embedding_rejects_non_finite_model_output(invalid_number: float) -> None:
    provider, runtime = _provider(invalid_number=invalid_number)
    try:
        with pytest.raises(RagError) as exc_info:
            await provider.embed_query("query", deadline=asyncio.get_running_loop().time() + 1)
    finally:
        await runtime.close()

    assert exc_info.value.code == RagErrorCode.MODEL_UNAVAILABLE


@pytest.mark.asyncio
async def test_embedding_rejects_wrong_dimensions() -> None:
    provider, runtime = _provider(dimensions=1023)
    try:
        with pytest.raises(RagError) as exc_info:
            await provider.embed_query("query", deadline=asyncio.get_running_loop().time() + 1)
    finally:
        await runtime.close()

    assert exc_info.value.code == RagErrorCode.MODEL_UNAVAILABLE


def test_embedding_rejects_revision_mismatch_before_inference() -> None:
    runtime = BgeModelRuntime(
        embedding_model=FakeEmbeddingModel(),
        reranker_model=FakeRerankerModel(),
        max_concurrency=1,
        deadline_seconds=1,
    )

    with pytest.raises(ValueError, match="RAG_MODEL_REVISION_MISMATCH"):
        BgeM3EmbeddingProvider(
            runtime=runtime,
            model_id="BAAI/bge-m3",
            revision="a" * 40,
            expected_revision="b" * 40,
            batch_size=2,
        )


@pytest.mark.asyncio
async def test_embedding_deadline_includes_executor_queue_and_discards_late_result() -> None:
    provider, runtime = _provider(delay=0.15, max_concurrency=1)
    loop = asyncio.get_running_loop()
    first = asyncio.create_task(provider.embed_query("first", deadline=loop.time() + 1))
    await asyncio.sleep(0.01)
    try:
        with pytest.raises(RagError) as exc_info:
            await provider.embed_query("second", deadline=loop.time() + 0.02)
        assert exc_info.value.code == RagErrorCode.RETRIEVAL_TIMEOUT
        assert len(await first) == 1024
    finally:
        await runtime.close()
