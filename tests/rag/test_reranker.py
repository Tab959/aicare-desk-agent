"""验证BGE Reranker的候选预算、截断、稳定排序和身份保留。"""

from __future__ import annotations

import asyncio

import pytest

from aicare_agent_service.contracts import Citation
from aicare_agent_service.rag.contracts import KnowledgeMetadata, RetrievedChunk
from aicare_agent_service.rag.errors import RagError, RagErrorCode
from aicare_agent_service.rag.model_runtime import BgeModelRuntime
from aicare_agent_service.rag.reranker import BgeReranker
from tests.fakes.rag_models import FakeEmbeddingModel, FakeRerankerModel, non_finite_values


def _candidate(index: int, content: str, fused_score: float = 0.5) -> RetrievedChunk:
    metadata = KnowledgeMetadata(
        tenant_id="tenant-1",
        knowledge_base_id="kb-1",
        document_id=f"doc-{index}",
        version=1,
        language="zh-CN",
        category="POLICY",
    )
    return RetrievedChunk(
        chunk_id=f"chunk-{index}",
        metadata=metadata,
        content=content,
        citation=Citation(
            document_id=metadata.document_id,
            version=1,
            title_path=("政策",),
            source_uri=f"https://kb.example/doc-{index}",
        ),
        fused_score=fused_score,
    )


def _reranker(*, invalid_number: float | None = None) -> tuple[BgeReranker, BgeModelRuntime]:
    runtime = BgeModelRuntime(
        embedding_model=FakeEmbeddingModel(),
        reranker_model=FakeRerankerModel(invalid_number=invalid_number),
        max_concurrency=1,
        deadline_seconds=1,
    )
    return (
        BgeReranker(
            runtime=runtime,
            model_id="BAAI/bge-reranker-v2-m3",
            revision="b" * 40,
            expected_revision="b" * 40,
            batch_size=8,
            max_candidates=30,
            max_passage_tokens=768,
            result_limit=6,
        ),
        runtime,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 31])
async def test_reranker_rejects_empty_or_oversized_candidates(count: int) -> None:
    reranker, runtime = _reranker()
    candidates = tuple(_candidate(index, "内容") for index in range(count))
    try:
        with pytest.raises(ValueError, match="RAG_RERANK_CANDIDATES_INVALID"):
            await reranker.rerank(
                query="退款",
                candidates=candidates,
                deadline=asyncio.get_running_loop().time() + 1,
            )
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_reranker_returns_relevant_first_and_preserves_chunk_identity() -> None:
    reranker, runtime = _reranker()
    candidates = (
        _candidate(1, "今日天气晴朗"),
        _candidate(2, "Steam退款需要人工审核"),
        _candidate(3, "退款说明"),
    )
    try:
        ranked = await reranker.rerank(
            query="Steam退款",
            candidates=candidates,
            deadline=asyncio.get_running_loop().time() + 1,
        )
    finally:
        await runtime.close()

    assert ranked[0].chunk_id == "chunk-2"
    assert ranked[0].citation == candidates[1].citation
    assert ranked[0].rerank_score is not None
    assert len(ranked) == 3


@pytest.mark.asyncio
async def test_reranker_truncates_passage_and_stably_breaks_score_ties() -> None:
    reranker, runtime = _reranker()
    candidates = tuple(_candidate(index, "相同" + "长" * 1000) for index in range(1, 9))
    try:
        ranked = await reranker.rerank(
            query="相同",
            candidates=candidates,
            deadline=asyncio.get_running_loop().time() + 1,
        )
    finally:
        await runtime.close()

    assert [chunk.chunk_id for chunk in ranked] == [f"chunk-{index}" for index in range(1, 7)]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_number", non_finite_values())
async def test_reranker_rejects_non_finite_scores(invalid_number: float) -> None:
    reranker, runtime = _reranker(invalid_number=invalid_number)
    try:
        with pytest.raises(RagError) as exc_info:
            await reranker.rerank(
                query="退款",
                candidates=(_candidate(1, "退款"),),
                deadline=asyncio.get_running_loop().time() + 1,
            )
    finally:
        await runtime.close()

    assert exc_info.value.code == RagErrorCode.MODEL_UNAVAILABLE
