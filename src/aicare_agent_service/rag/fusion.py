"""以确定性Python RRF融合BM25与Dense排名并保留两路rank。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from aicare_agent_service.rag.contracts import RetrievedChunk


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    """保存融合后的安全Chunk、RRF分数和两个低基数排名。"""

    chunk: RetrievedChunk
    score: float
    sparse_rank: int | None
    dense_rank: int | None


@dataclass(slots=True)
class _FusionRecord:
    """RRF计算期间保存可变累计分数和两个通道排名。"""

    chunk: RetrievedChunk
    score: float = 0.0
    sparse_rank: int | None = None
    dense_rank: int | None = None


def reciprocal_rank_fusion(
    *,
    sparse: Sequence[RetrievedChunk],
    dense: Sequence[RetrievedChunk],
    rank_constant: int = 60,
    limit: int = 30,
) -> tuple[FusedCandidate, ...]:
    """按1/(k+rank)融合并按Chunk ID去重，不比较原始ES分数。"""
    # 1、固定预算必须为正且输出不超过Reranker允许的30条。
    if rank_constant < 1 or not 1 <= limit <= 30:
        raise ValueError("RAG_RRF_LIMIT_INVALID")
    records: dict[str, _FusionRecord] = {}
    # 2、分别按从1开始的通道排名累加RRF分数，重复Chunk只保留一份正文。
    for channel, candidates in (("sparse", sparse), ("dense", dense)):
        seen_in_channel: set[str] = set()
        for rank, chunk in enumerate(candidates, start=1):
            if chunk.chunk_id in seen_in_channel:
                continue
            seen_in_channel.add(chunk.chunk_id)
            record = records.setdefault(chunk.chunk_id, _FusionRecord(chunk=chunk))
            record.score += 1.0 / (rank_constant + rank)
            if channel == "sparse":
                record.sparse_rank = rank
            else:
                record.dense_rank = rank
    # 3、分数相同按Chunk ID稳定决胜，原始BM25/cosine分数不参与排序。
    ordered = sorted(records.items(), key=lambda item: (-item[1].score, item[0]))
    result: list[FusedCandidate] = []
    for _, record in ordered[:limit]:
        score = record.score
        if not math.isfinite(score):
            raise ValueError("RAG_RRF_SCORE_INVALID")
        chunk = record.chunk
        result.append(
            FusedCandidate(
                chunk=chunk.model_copy(update={"fused_score": score}),
                score=score,
                sparse_rank=record.sparse_rank,
                dense_rank=record.dense_rank,
            )
        )
    return tuple(result)
