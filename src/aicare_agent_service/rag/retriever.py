"""编排脱敏改写、BM25/Dense并发召回、RRF、BGE精排与确定性证据门禁。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Protocol, TypeVar, cast

from aicare_agent_service.rag.contracts import RetrievalQuery, RetrievalResult, RetrievedChunk
from aicare_agent_service.rag.errors import RagError, RagErrorCode
from aicare_agent_service.rag.fusion import reciprocal_rank_fusion
from aicare_agent_service.rag.query_rewrite import QueryRewriter

T = TypeVar("T")


class HybridIndex(Protocol):
    """混合检索器依赖的两路受控ES查询接口。"""

    async def sparse_search(self, query: RetrievalQuery) -> Sequence[RetrievedChunk]: ...

    async def dense_search(
        self,
        query: RetrievalQuery,
        vector: Sequence[float],
        *,
        num_candidates: int,
    ) -> Sequence[RetrievedChunk]: ...


class Embeddings(Protocol):
    """查询向量化所需的最小生产协议。"""

    async def embed_query(self, text: str, *, deadline: float) -> Sequence[float]: ...


class Reranker(Protocol):
    """RRF后有限候选精排所需的最小协议。"""

    async def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RetrievedChunk],
        deadline: float,
    ) -> Sequence[RetrievedChunk]: ...


class HybridKnowledgeRetriever:
    """在单一绝对deadline内完成生产混合召回和证据判断。"""

    def __init__(
        self,
        *,
        rewriter: QueryRewriter | Any,
        embeddings: Embeddings,
        index: HybridIndex,
        reranker: Reranker,
        deadline_seconds: float,
        sparse_limit: int = 40,
        dense_limit: int = 40,
        num_candidates: int = 120,
        fusion_limit: int = 30,
        evidence_score_threshold: float = 0.45,
    ) -> None:
        """绑定生产适配器与固定召回、融合、精排和证据预算。"""
        # 1、数量与deadline必须符合已冻结的Task 8预算。
        if (
            deadline_seconds <= 0
            or not 1 <= sparse_limit <= 100
            or not 1 <= dense_limit <= 100
            or num_candidates < dense_limit
            or not 1 <= fusion_limit <= 30
            or not 0 <= evidence_score_threshold <= 1
        ):
            raise ValueError("RAG_RETRIEVER_CONFIGURATION_INVALID")
        # 2、依赖和预算在构造后不再动态扩大。
        self._rewriter = rewriter
        self._embeddings = embeddings
        self._index = index
        self._reranker = reranker
        self._deadline_seconds = deadline_seconds
        self._sparse_limit = sparse_limit
        self._dense_limit = dense_limit
        self._num_candidates = num_candidates
        self._fusion_limit = fusion_limit
        self._evidence_threshold = evidence_score_threshold

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """并发执行两路召回，允许单路降级但双路失败时阻断。"""
        # 1、整条检索链共用一个绝对deadline，包括改写、Embedding、重试和精排。
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        deadline = loop.time() + self._deadline_seconds
        try:
            async with asyncio.timeout_at(deadline):
                rewrite = await self._rewriter.rewrite(query.text, deadline=deadline)
                sparse_query = query.model_copy(
                    update={"text": rewrite.query, "candidate_limit": self._sparse_limit}
                )
                dense_query = query.model_copy(
                    update={"text": rewrite.query, "candidate_limit": self._dense_limit}
                )

                async def sparse_operation() -> Sequence[RetrievedChunk]:
                    return await self._index.sparse_search(sparse_query)

                async def dense_operation() -> Sequence[RetrievedChunk]:
                    vector = await self._embeddings.embed_query(rewrite.query, deadline=deadline)
                    return await self._index.dense_search(
                        dense_query,
                        vector,
                        num_candidates=self._num_candidates,
                    )

                # 2、两路并发且各自最多重试一次瞬时索引错误，重试仍受同一deadline约束。
                sparse_result, dense_result = await asyncio.gather(
                    self._retry_transient(sparse_operation, deadline=deadline),
                    self._retry_transient(dense_operation, deadline=deadline),
                    return_exceptions=True,
                )
                sparse_failed = isinstance(sparse_result, BaseException)
                dense_failed = isinstance(dense_result, BaseException)
                if sparse_failed and dense_failed:
                    raise RagError(RagErrorCode.INDEX_UNAVAILABLE)
                sparse_chunks = (
                    () if sparse_failed else tuple(cast(Sequence[RetrievedChunk], sparse_result))
                )
                dense_chunks = (
                    () if dense_failed else tuple(cast(Sequence[RetrievedChunk], dense_result))
                )

                # 3、RRF只使用通道排名，融合后最多30条进入BGE Reranker。
                fused = reciprocal_rank_fusion(
                    sparse=sparse_chunks,
                    dense=dense_chunks,
                    limit=self._fusion_limit,
                )
                if not fused:
                    return self._result(query.tenant_id, (), False, started)
                reranked = tuple(
                    await self._reranker.rerank(
                        query=rewrite.query,
                        candidates=tuple(item.chunk for item in fused),
                        deadline=deadline,
                    )
                )[: query.result_limit]
                # 4、证据充分性由数量、分数、版本和来源字段确定，不读取模型自述。
                sufficient = self._has_sufficient_evidence(reranked)
                return self._result(query.tenant_id, reranked, sufficient, started)
        except TimeoutError as exc:
            raise RagError(RagErrorCode.RETRIEVAL_TIMEOUT) from exc

    async def _retry_transient(
        self,
        operation: Callable[[], Awaitable[T]],
        *,
        deadline: float,
    ) -> T:
        """只对瞬时索引不可用执行一次且不越过绝对deadline的重试。"""
        # 1、首次调用成功直接返回，其他稳定错误不重试。
        try:
            return await operation()
        except RagError as exc:
            if exc.code is not RagErrorCode.INDEX_UNAVAILABLE:
                raise
        # 2、仅在仍有deadline预算时执行第二次，绝不形成循环重试。
        if asyncio.get_running_loop().time() >= deadline:
            raise RagError(RagErrorCode.RETRIEVAL_TIMEOUT)
        return await operation()

    def _has_sufficient_evidence(self, chunks: Sequence[RetrievedChunk]) -> bool:
        """以确定性字段检查精排结果是否足以进入回答生成。"""
        # 1、至少一个候选且首条精排分数达到配置阈值。
        if not chunks or chunks[0].rerank_score is None:
            return False
        if chunks[0].rerank_score < self._evidence_threshold:
            return False
        # 2、所有保留证据必须具备正版本、非空文档身份、标题路径和安全来源。
        return all(
            chunk.metadata.version >= 1
            and bool(chunk.metadata.document_id)
            and bool(chunk.citation.title_path)
            and bool(chunk.citation.source_uri)
            for chunk in chunks
        )

    @staticmethod
    def _result(
        tenant_id: str,
        chunks: Sequence[RetrievedChunk],
        sufficient: bool,
        started: float,
    ) -> RetrievalResult:
        """构造不包含向量、ES响应或原始通道分数的安全检索结果。"""
        # 1、只记录毫秒耗时并由RetrievalResult再次执行单租户校验。
        elapsed_ms = min(int((time.perf_counter() - started) * 1000), 300_000)
        return RetrievalResult(
            tenant_id=tenant_id,
            chunks=tuple(chunks),
            sufficient_evidence=sufficient,
            elapsed_ms=elapsed_ms,
        )
