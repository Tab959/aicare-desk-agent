"""验证混合召回deadline、单路降级、双路阻断、精排和证据门禁。"""

from __future__ import annotations

import pytest

from aicare_agent_service.contracts import Citation
from aicare_agent_service.rag.contracts import (
    KnowledgeMetadata,
    RetrievalFilter,
    RetrievalQuery,
    RetrievedChunk,
)
from aicare_agent_service.rag.errors import RagError, RagErrorCode
from aicare_agent_service.rag.query_rewrite import QueryRewriteResult
from aicare_agent_service.rag.retriever import HybridKnowledgeRetriever


def _chunk(chunk_id: str, *, rerank_score: float | None = None) -> RetrievedChunk:
    """构造具备完整版本和来源的单租户候选。"""
    metadata = KnowledgeMetadata(
        tenant_id="tenant-a",
        knowledge_base_id="kb-help",
        document_id=f"doc-{chunk_id}",
        version=2,
        language="zh-CN",
        category="DELIVERY",
    )
    return RetrievedChunk(
        chunk_id=chunk_id,
        metadata=metadata,
        content=f"Steam礼物人工交付 {chunk_id}",
        citation=Citation(
            document_id=metadata.document_id,
            version=2,
            title_path=("交付",),
            source_uri=f"aicare://knowledge/{metadata.document_id}",
        ),
        fused_score=0.0,
        rerank_score=rerank_score,
    )


class FakeRewriter:
    """返回已知安全查询并记录收到的原始问题。"""

    def __init__(self) -> None:
        self.questions: list[str] = []

    async def rewrite(self, question: str, *, deadline: float) -> QueryRewriteResult:
        self.questions.append(question)
        return QueryRewriteResult(
            query=question.replace("密码=abc12345", "密码=[REDACTED_PASSWORD]"),
            redacted_original=question.replace("密码=abc12345", "密码=[REDACTED_PASSWORD]"),
            language_hint="zh-CN",
            intent_hint="POLICY",
            confidence=0.9,
            used_fallback=False,
        )


class FakeEmbeddings:
    """记录Embedding文本并返回合法固定向量。"""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed_query(self, text: str, *, deadline: float) -> tuple[float, ...]:
        self.queries.append(text)
        return tuple([1.0] + [0.0] * 1023)


class FakeStore:
    """可注入两路结果或瞬时错误的检索边界替身。"""

    def __init__(self, *, sparse: object, dense: object) -> None:
        self.sparse = sparse
        self.dense = dense
        self.sparse_calls = 0
        self.dense_calls = 0
        self.queries: list[RetrievalQuery] = []

    async def sparse_search(self, query: RetrievalQuery) -> tuple[RetrievedChunk, ...]:
        self.sparse_calls += 1
        self.queries.append(query)
        if isinstance(self.sparse, Exception):
            raise self.sparse
        return tuple(self.sparse)

    async def dense_search(
        self,
        query: RetrievalQuery,
        vector: tuple[float, ...],
        *,
        num_candidates: int,
    ) -> tuple[RetrievedChunk, ...]:
        self.dense_calls += 1
        self.queries.append(query)
        assert len(vector) == 1024
        assert num_candidates == 120
        if isinstance(self.dense, Exception):
            raise self.dense
        return tuple(self.dense)


class FakeReranker:
    """按指定相关Chunk置高分，模拟MRR提升。"""

    async def rerank(
        self, *, query: str, candidates: object, deadline: float
    ) -> tuple[RetrievedChunk, ...]:
        values = tuple(candidates)
        return tuple(
            chunk.model_copy(update={"rerank_score": 0.9 if chunk.chunk_id == "relevant" else 0.2})
            for chunk in sorted(values, key=lambda item: item.chunk_id != "relevant")
        )[:6]


def _query() -> RetrievalQuery:
    return RetrievalQuery(
        tenant_id="tenant-a",
        text="密码=abc12345，Steam礼物怎么交付？",
        filters=RetrievalFilter(knowledge_base_ids=("kb-help",), languages=("zh-CN",)),
        candidate_limit=40,
        result_limit=6,
    )


@pytest.mark.asyncio
async def test_hybrid_retrieval_redacts_embeds_filters_fuses_and_reranks() -> None:
    embeddings = FakeEmbeddings()
    store = FakeStore(
        sparse=(_chunk("noise"), _chunk("relevant")),
        dense=(_chunk("relevant"),),
    )
    retriever = HybridKnowledgeRetriever(
        rewriter=FakeRewriter(),
        embeddings=embeddings,
        index=store,
        reranker=FakeReranker(),
        deadline_seconds=2,
        evidence_score_threshold=0.5,
    )

    result = await retriever.retrieve(_query())

    assert result.sufficient_evidence is True
    assert result.chunks[0].chunk_id == "relevant"
    assert embeddings.queries == ["密码=[REDACTED_PASSWORD]，Steam礼物怎么交付？"]
    assert all(query.tenant_id == "tenant-a" for query in store.queries)
    assert all(query.filters.knowledge_base_ids == ("kb-help",) for query in store.queries)


@pytest.mark.asyncio
async def test_one_channel_can_degrade_but_both_channels_fail_closed() -> None:
    transient = RagError(RagErrorCode.INDEX_UNAVAILABLE)
    one_failed = FakeStore(sparse=transient, dense=(_chunk("relevant"),))
    retriever = HybridKnowledgeRetriever(
        rewriter=FakeRewriter(),
        embeddings=FakeEmbeddings(),
        index=one_failed,
        reranker=FakeReranker(),
        deadline_seconds=2,
        evidence_score_threshold=0.5,
    )

    result = await retriever.retrieve(_query())

    assert result.sufficient_evidence is True
    assert one_failed.sparse_calls == 2
    assert one_failed.dense_calls == 1

    both_failed = FakeStore(sparse=transient, dense=transient)
    blocked = HybridKnowledgeRetriever(
        rewriter=FakeRewriter(),
        embeddings=FakeEmbeddings(),
        index=both_failed,
        reranker=FakeReranker(),
        deadline_seconds=2,
    )
    with pytest.raises(RagError) as captured:
        await blocked.retrieve(_query())
    assert captured.value.code is RagErrorCode.INDEX_UNAVAILABLE
    assert both_failed.sparse_calls == 2
    assert both_failed.dense_calls == 2


@pytest.mark.asyncio
async def test_low_reranker_score_is_deterministically_insufficient() -> None:
    class LowReranker(FakeReranker):
        async def rerank(
            self, *, query: str, candidates: object, deadline: float
        ) -> tuple[RetrievedChunk, ...]:
            return tuple(chunk.model_copy(update={"rerank_score": 0.1}) for chunk in candidates)

    retriever = HybridKnowledgeRetriever(
        rewriter=FakeRewriter(),
        embeddings=FakeEmbeddings(),
        index=FakeStore(sparse=(_chunk("weak"),), dense=()),
        reranker=LowReranker(),
        deadline_seconds=2,
        evidence_score_threshold=0.5,
    )

    result = await retriever.retrieve(_query())

    assert result.sufficient_evidence is False
    assert result.chunks[0].rerank_score == 0.1
