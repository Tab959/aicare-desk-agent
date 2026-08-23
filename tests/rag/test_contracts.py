"""验证 RAG 契约的严格字段、边界、租户隔离与安全序列化。"""

import math

import pytest
from pydantic import ValidationError

from aicare_agent_service.rag.contracts import (
    KnowledgeChunk,
    KnowledgeMetadata,
    RagAnswer,
    RawKnowledgeDocument,
    RetrievalFilter,
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
)
from aicare_agent_service.rag.errors import RagError, RagErrorCode


def _metadata() -> KnowledgeMetadata:
    """返回固定字段的最小知识元数据。"""
    return KnowledgeMetadata(
        tenant_id="tenant-1",
        knowledge_base_id="kb-1",
        document_id="doc-1",
        version=1,
        language="zh-CN",
        category="DELIVERY_POLICY",
    )


def _chunk(**overrides: object) -> KnowledgeChunk:
    """返回合法的1024维索引Chunk。"""
    values: dict[str, object] = {
        "metadata": _metadata(),
        "chunk_id": "chunk-1",
        "title_path": ("交付政策",),
        "ordinal": 1,
        "content": "CDK将在支付完成后自动交付。",
        "token_count": 12,
        "content_checksum": "a" * 64,
        "embedding": tuple(0.0 for _ in range(1024)),
    }
    values.update(overrides)
    return KnowledgeChunk(**values)


@pytest.mark.parametrize("field", ["tenant_id", "knowledge_base_id", "document_id"])
def test_metadata_rejects_blank_identifiers(field: str) -> None:
    values = _metadata().model_dump()
    values[field] = " "

    with pytest.raises(ValidationError):
        KnowledgeMetadata(**values)


def test_metadata_rejects_unknown_unbounded_fields() -> None:
    with pytest.raises(ValidationError) as exc_info:
        KnowledgeMetadata(**_metadata().model_dump(), arbitrary_metadata={"secret": "value"})

    assert "value" not in str(exc_info.value)


def test_metadata_rejects_non_positive_version() -> None:
    with pytest.raises(ValidationError):
        KnowledgeMetadata(**{**_metadata().model_dump(), "version": 0})


def test_raw_document_rejects_invalid_source_uri() -> None:
    with pytest.raises(ValidationError):
        RawKnowledgeDocument(
            metadata=_metadata(),
            file_name="policy.md",
            media_type="text/markdown",
            source_uri="not a uri",
            content=b"policy",
        )


@pytest.mark.parametrize("dimension", [0, 1023, 1025])
def test_chunk_rejects_wrong_embedding_dimension(dimension: int) -> None:
    with pytest.raises(ValidationError, match="1024"):
        _chunk(embedding=tuple(0.0 for _ in range(dimension)))


@pytest.mark.parametrize("invalid_number", [math.nan, math.inf, -math.inf])
def test_chunk_rejects_non_finite_embedding_values(invalid_number: float) -> None:
    embedding = [0.0 for _ in range(1024)]
    embedding[42] = invalid_number

    with pytest.raises(ValidationError):
        _chunk(embedding=tuple(embedding))


def test_retrieval_filter_has_no_tenant_field_for_model_to_override() -> None:
    with pytest.raises(ValidationError):
        RetrievalFilter(tenant_id="another-tenant")


def test_retrieval_query_binds_tenant_outside_business_filters() -> None:
    query = RetrievalQuery(
        tenant_id="tenant-1",
        text="Steam礼物多久交付？",
        filters=RetrievalFilter(categories=("DELIVERY_POLICY",)),
        candidate_limit=30,
        result_limit=6,
    )

    assert query.tenant_id == "tenant-1"
    assert "tenant_id" not in RetrievalFilter.model_fields


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("candidate_limit", 101), ("result_limit", 31), ("candidate_limit", 0)],
)
def test_retrieval_query_rejects_unsafe_candidate_limits(field_name: str, value: int) -> None:
    with pytest.raises(ValidationError):
        RetrievalQuery(tenant_id="tenant-1", text="query", **{field_name: value})


@pytest.mark.parametrize("invalid_score", [math.nan, math.inf, -math.inf])
def test_retrieved_chunk_rejects_non_finite_scores(invalid_score: float) -> None:
    from aicare_agent_service.contracts import Citation

    with pytest.raises(ValidationError):
        RetrievedChunk(
            chunk_id="chunk-1",
            metadata=_metadata(),
            content="安全片段",
            citation=Citation(
                document_id="doc-1",
                version=1,
                title_path=("标题",),
                source_uri="https://kb.example/doc-1",
            ),
            fused_score=invalid_score,
        )


def test_retrieval_result_rejects_cross_tenant_chunk() -> None:
    from aicare_agent_service.contracts import Citation

    foreign_metadata = _metadata().model_copy(update={"tenant_id": "tenant-2"})
    foreign_chunk = RetrievedChunk(
        chunk_id="chunk-foreign",
        metadata=foreign_metadata,
        content="其他租户片段",
        citation=Citation(
            document_id="doc-1",
            version=1,
            title_path=("标题",),
            source_uri="https://kb.example/doc-1",
        ),
        fused_score=0.8,
    )

    with pytest.raises(ValidationError, match="跨租户"):
        RetrievalResult(
            tenant_id="tenant-1",
            chunks=(foreign_chunk,),
            sufficient_evidence=True,
            elapsed_ms=10,
        )


def test_rag_answer_reuses_citation_and_never_serializes_embeddings() -> None:
    from aicare_agent_service.contracts import Citation

    answer = RagAnswer(
        content="根据交付政策，支付后会自动交付。",
        sufficient_evidence=True,
        citations=(
            Citation(
                document_id="doc-1",
                version=1,
                title_path=("交付政策",),
                source_uri="https://kb.example/doc-1",
            ),
        ),
    )

    rendered = answer.model_dump_json()
    assert "embedding" not in rendered.lower()
    assert "完整原文canary" not in rendered


def test_rag_error_exposes_only_stable_safe_payload() -> None:
    error = RagError(RagErrorCode.INDEX_UNAVAILABLE)

    assert error.to_safe_payload() == {
        "code": "RAG_INDEX_UNAVAILABLE",
        "retryable": True,
        "message": "知识检索服务暂时不可用，请稍后重试。",
    }
    assert "password" not in str(error).lower()
