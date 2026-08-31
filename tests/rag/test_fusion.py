"""验证Python RRF不比较异构原始分数并保持去重与稳定排序。"""

from __future__ import annotations

from aicare_agent_service.contracts import Citation
from aicare_agent_service.rag.contracts import KnowledgeMetadata, RetrievedChunk
from aicare_agent_service.rag.fusion import reciprocal_rank_fusion


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    """构造只在原始分数上不同的固定候选。"""
    metadata = KnowledgeMetadata(
        tenant_id="tenant-a",
        knowledge_base_id="kb-a",
        document_id=f"doc-{chunk_id}",
        version=1,
        language="zh-CN",
        category="FAQ",
    )
    return RetrievedChunk(
        chunk_id=chunk_id,
        metadata=metadata,
        content=f"content {chunk_id}",
        citation=Citation(
            document_id=metadata.document_id,
            version=1,
            title_path=("FAQ",),
            source_uri=f"aicare://knowledge/{metadata.document_id}",
        ),
        fused_score=score,
    )


def test_rrf_uses_rank_formula_deduplicates_and_retains_channel_ranks() -> None:
    sparse = (_chunk("a", 1000.0), _chunk("b", 0.1), _chunk("c", 0.01))
    dense = (_chunk("b", 0.99), _chunk("a", 0.98), _chunk("d", 0.97))

    fused = reciprocal_rank_fusion(sparse=sparse, dense=dense, rank_constant=60, limit=30)

    assert [item.chunk.chunk_id for item in fused] == ["a", "b", "c", "d"]
    assert fused[0].sparse_rank == 1 and fused[0].dense_rank == 2
    assert fused[1].sparse_rank == 2 and fused[1].dense_rank == 1
    assert fused[0].score == 1 / 61 + 1 / 62
    assert len({item.chunk.chunk_id for item in fused}) == len(fused)


def test_rrf_order_is_stable_when_scores_tie() -> None:
    first = reciprocal_rank_fusion(
        sparse=(_chunk("b", 5.0),),
        dense=(_chunk("a", -500.0),),
    )
    second = reciprocal_rank_fusion(
        sparse=(_chunk("b", -1.0),),
        dense=(_chunk("a", 999.0),),
    )

    assert [item.chunk.chunk_id for item in first] == ["a", "b"]
    assert [item.chunk.chunk_id for item in second] == ["a", "b"]
