"""定义知识文档、Chunk、索引、检索和回答阶段的严格 Pydantic 契约。"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from aicare_agent_service.contracts import Citation

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class RagContractModel(BaseModel):
    """RAG 内部模型基类：未知字段拒绝、实例冻结并隐藏输入错误值。"""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )


class KnowledgeMetadata(RagContractModel):
    """所有知识对象共享的固定身份和业务过滤字段。"""

    tenant_id: NonEmptyText
    knowledge_base_id: NonEmptyText
    document_id: NonEmptyText
    version: PositiveInt
    language: NonEmptyText
    category: NonEmptyText
    game_id: NonEmptyText | None = None
    purchase_method: NonEmptyText | None = None
    issue_type: NonEmptyText | None = None


class RawKnowledgeDocument(RagContractModel):
    """从受信 Java 内部接口获得、尚未解析的有限原始文档。"""

    metadata: KnowledgeMetadata
    file_name: NonEmptyText
    media_type: NonEmptyText
    source_uri: AnyUrl
    content: Annotated[bytes, Field(min_length=1, max_length=20 * 1024 * 1024)]


class ParsedSection(RagContractModel):
    """解析器保留结构后产生的单个正文区段。"""

    metadata: KnowledgeMetadata
    title_path: Annotated[tuple[NonEmptyText, ...], Field(max_length=16)] = ()
    ordinal: PositiveInt
    text: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000_000)
    ]


class KnowledgeChunk(RagContractModel):
    """可索引知识片段；Embedding仅在索引边界存在，不进入回答契约。"""

    metadata: KnowledgeMetadata
    chunk_id: NonEmptyText
    title_path: Annotated[tuple[NonEmptyText, ...], Field(max_length=16)] = ()
    ordinal: PositiveInt
    content: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000)
    ]
    token_count: Annotated[int, Field(strict=True, ge=1, le=640)]
    content_checksum: Sha256
    embedding: tuple[FiniteFloat, ...] | None = None

    @field_validator("embedding")
    @classmethod
    def require_embedding_dimensions(
        cls, embedding: tuple[float, ...] | None
    ) -> tuple[float, ...] | None:
        """只允许缺省或精确 1024 维的有限向量。"""
        # 1、解析和Chunk阶段可暂时没有向量。
        if embedding is None:
            return None
        # 2、BGE-M3维度契约固定为1024。
        if len(embedding) != 1024:
            raise ValueError("Embedding必须为1024维")
        # 3、二次检查有限值，避免未来字段约束调整后放入NaN或Infinity。
        if not all(math.isfinite(value) for value in embedding):
            raise ValueError("Embedding只允许有限数值")
        return embedding


class IndexedDocument(RagContractModel):
    """一次原子索引替换所需的文档身份、模型指纹和全部Chunk。"""

    metadata: KnowledgeMetadata
    embedding_fingerprint: Sha256
    chunks: Annotated[tuple[KnowledgeChunk, ...], Field(min_length=1, max_length=5000)]

    @model_validator(mode="after")
    def require_consistent_embedded_chunks(self) -> IndexedDocument:
        """保证所有Chunk属于同一文档且已完成Embedding。"""
        # 1、每个Chunk必须携带与文档完全相同的固定元数据。
        if any(chunk.metadata != self.metadata for chunk in self.chunks):
            raise ValueError("索引Chunk身份与文档不一致")
        # 2、索引边界不接受尚未向量化的Chunk。
        if any(chunk.embedding is None for chunk in self.chunks):
            raise ValueError("索引Chunk缺少Embedding")
        return self


class IndexStatus(StrEnum):
    """文档索引操作的有限终态。"""

    INDEXED = "INDEXED"
    DELETED = "DELETED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class IndexResult(RagContractModel):
    """索引或删除操作的结构化结果，不携带ES原始响应。"""

    tenant_id: NonEmptyText
    document_id: NonEmptyText
    version: PositiveInt
    status: IndexStatus
    indexed_chunks: Annotated[int, Field(strict=True, ge=0, le=5000)] = 0
    error_code: NonEmptyText | None = None


class RetrievalFilter(RagContractModel):
    """模型可建议的有限业务过滤器；刻意不包含tenant_id。"""

    knowledge_base_ids: Annotated[tuple[NonEmptyText, ...], Field(max_length=16)] = ()
    languages: Annotated[tuple[NonEmptyText, ...], Field(max_length=16)] = ()
    categories: Annotated[tuple[NonEmptyText, ...], Field(max_length=32)] = ()
    game_ids: Annotated[tuple[NonEmptyText, ...], Field(max_length=32)] = ()
    purchase_methods: Annotated[tuple[NonEmptyText, ...], Field(max_length=16)] = ()
    issue_types: Annotated[tuple[NonEmptyText, ...], Field(max_length=32)] = ()


class RetrievalQuery(RagContractModel):
    """由代码注入租户身份并附带有限过滤器的检索请求。"""

    tenant_id: NonEmptyText
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    filters: RetrievalFilter = Field(default_factory=RetrievalFilter)
    candidate_limit: Annotated[int, Field(strict=True, ge=1, le=100)] = 30
    result_limit: Annotated[int, Field(strict=True, ge=1, le=30)] = 6

    @model_validator(mode="after")
    def require_result_within_candidates(self) -> RetrievalQuery:
        """最终结果数不能超过候选召回数。"""
        # 1、拒绝无法由候选集满足的结果预算。
        if self.result_limit > self.candidate_limit:
            raise ValueError("结果数量不能超过候选数量")
        # 2、返回已验证实例。
        return self


class RetrievedChunk(RagContractModel):
    """可供生成阶段使用的有限Chunk，不携带Embedding或ES原始分数。"""

    chunk_id: NonEmptyText
    metadata: KnowledgeMetadata
    content: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=20_000)
    ]
    citation: Citation
    fused_score: FiniteFloat
    rerank_score: FiniteFloat | None = None

    @model_validator(mode="after")
    def require_matching_citation(self) -> RetrievedChunk:
        """保证引用身份与Chunk固定元数据一致。"""
        # 1、文档ID和版本任一不一致都会造成错误归因。
        if (
            self.citation.document_id != self.metadata.document_id
            or self.citation.version != self.metadata.version
        ):
            raise ValueError("引用身份与检索Chunk不一致")
        # 2、返回已验证实例。
        return self


class RetrievalResult(RagContractModel):
    """一次租户内检索的安全候选集合和证据状态。"""

    tenant_id: NonEmptyText
    chunks: Annotated[tuple[RetrievedChunk, ...], Field(max_length=30)] = ()
    sufficient_evidence: bool
    elapsed_ms: Annotated[int, Field(strict=True, ge=0, le=300_000)]

    @model_validator(mode="after")
    def require_single_tenant(self) -> RetrievalResult:
        """阻止任何跨租户Chunk混入一个检索结果。"""
        # 1、逐项核对运行时绑定的租户身份。
        if any(chunk.metadata.tenant_id != self.tenant_id for chunk in self.chunks):
            raise ValueError("检索结果包含跨租户Chunk")
        # 2、宣称证据充足时至少要有一个Chunk。
        if self.sufficient_evidence and not self.chunks:
            raise ValueError("证据充足的结果必须包含Chunk")
        return self


class RagAnswer(RagContractModel):
    """RAG子图可交给专业Agent或根图的最小回答与引用结果。"""

    content: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12_000)
    ]
    sufficient_evidence: bool
    citations: Annotated[tuple[Citation, ...], Field(max_length=6)] = ()

    @model_validator(mode="after")
    def require_citations_for_supported_answer(self) -> RagAnswer:
        """有依据回答必须至少带一条现有Citation。"""
        # 1、禁止无引用却宣称证据充分。
        if self.sufficient_evidence and not self.citations:
            raise ValueError("证据充分的RAG回答必须包含引用")
        # 2、返回已验证实例。
        return self
