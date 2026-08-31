"""定义RAG检索与生成的离线评测数据契约、指标计算、门禁和Chunk网格选择。"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EvaluationCategory(StrEnum):
    """人工检索集要求覆盖的五类问题。"""

    EXACT_FACT = "exact_fact"
    SYNONYM = "synonym"
    MULTI_HOP = "multi_hop"
    BUSINESS_FILTER = "business_filter"
    NO_ANSWER_ADVERSARIAL = "no_answer_adversarial"


class RelevantChunk(BaseModel):
    """人工标注的相关文档版本、Chunk和分级相关度。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: NonEmptyText
    version: Annotated[int, Field(strict=True, ge=1)]
    chunk_id: NonEmptyText
    relevance: Annotated[int, Field(strict=True, ge=1, le=3)] = 3


class RetrievalEvaluationCase(BaseModel):
    """一条不依赖最终答案相似度的人工检索标注。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: NonEmptyText
    category: EvaluationCategory
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    tenant_id: NonEmptyText
    knowledge_base_ids: tuple[NonEmptyText, ...] = ()
    categories: tuple[NonEmptyText, ...] = ()
    relevant_chunks: tuple[RelevantChunk, ...] = ()

    @model_validator(mode="after")
    def require_expected_answer_shape(self) -> RetrievalEvaluationCase:
        """只有无答案/对抗类允许没有相关Chunk。"""
        # 1、无答案类必须明确没有相关证据。
        if self.category is EvaluationCategory.NO_ANSWER_ADVERSARIAL:
            if self.relevant_chunks:
                raise ValueError("无答案评测不得标注相关Chunk")
            return self
        # 2、其他四类至少标注一个精确文档版本和Chunk。
        if not self.relevant_chunks:
            raise ValueError("可回答评测必须标注相关Chunk")
        return self


class RankedChunk(BaseModel):
    """一次检索预测中的有序Chunk身份。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: NonEmptyText
    version: Annotated[int, Field(strict=True, ge=1)]
    chunk_id: NonEmptyText
    tenant_id: NonEmptyText


class RetrievalPrediction(BaseModel):
    """与人工用例绑定的有限有序检索结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: NonEmptyText
    ranked_chunks: Annotated[tuple[RankedChunk, ...], Field(max_length=30)] = ()


class RetrievalMetrics(BaseModel):
    """首轮检索、跨章节召回和租户隔离的聚合指标。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recall_at_5: Annotated[float, Field(ge=0, le=1)]
    mrr: Annotated[float, Field(ge=0, le=1)]
    ndcg_at_5: Annotated[float, Field(ge=0, le=1)]
    multihop_recall_at_8: Annotated[float, Field(ge=0, le=1)]
    cross_tenant_leaks: Annotated[int, Field(strict=True, ge=0)]
    evaluated_answerable_cases: Annotated[int, Field(strict=True, ge=1)]

    @property
    def passed(self) -> bool:
        """执行Task 8固定检索门禁。"""
        # 1、四项阈值和零跨租户泄漏必须同时成立。
        return (
            self.recall_at_5 >= 0.90
            and self.mrr >= 0.80
            and self.ndcg_at_5 >= 0.80
            and self.cross_tenant_leaks == 0
        )


class GenerationJudgment(BaseModel):
    """单条生成回答的人工或结构化评审结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: NonEmptyText
    faithful: bool
    relevant: bool
    citation_coverage: Annotated[float, Field(ge=0, le=1)]
    unsupported_claims: Annotated[int, Field(strict=True, ge=0)]


class GenerationMetrics(BaseModel):
    """生成质量聚合指标与固定门禁。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    faithfulness: Annotated[float, Field(ge=0, le=1)]
    answer_relevance: Annotated[float, Field(ge=0, le=1)]
    citation_coverage: Annotated[float, Field(ge=0, le=1)]
    unsupported_claim_rate: Annotated[float, Field(ge=0, le=1)]

    @property
    def passed(self) -> bool:
        """执行Task 8固定生成门禁。"""
        # 1、四项指标必须全部达到固定阈值。
        return (
            self.faithfulness >= 0.95
            and self.answer_relevance >= 0.90
            and self.citation_coverage == 1.0
            and self.unsupported_claim_rate == 0.0
        )


class ChunkGridScore(BaseModel):
    """一个Chunk目标/重叠组合的版本化评测结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_tokens: Annotated[int, Field(strict=True, ge=1)]
    overlap_percent: Annotated[int, Field(strict=True, ge=1, le=99)]
    recall_at_5: Annotated[float, Field(ge=0, le=1)]
    ndcg_at_5: Annotated[float, Field(ge=0, le=1)]
    report_version: NonEmptyText = "rag-chunk-grid-v1"


def load_evaluation_dataset(path: Path) -> tuple[RetrievalEvaluationCase, ...]:
    """从JSONL读取并验证无重复ID的人工检索集。"""
    # 1、逐行解析，空行跳过，结构错误立即阻断而不静默丢弃样本。
    cases = tuple(
        RetrievalEvaluationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    # 2、重复case_id会让预测错配，必须拒绝。
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("RAG_EVALUATION_CASE_ID_DUPLICATED")
    return cases


def evaluate_retrieval(
    cases: Sequence[RetrievalEvaluationCase],
    predictions: Sequence[RetrievalPrediction],
) -> RetrievalMetrics:
    """按人工Chunk标注分别计算Recall@5、MRR、NDCG@5和多跳Recall@8。"""
    # 1、每个用例必须恰好有一条预测，缺失或重复均阻断评测。
    by_case = {prediction.case_id: prediction for prediction in predictions}
    if len(by_case) != len(predictions) or set(by_case) != {case.case_id for case in cases}:
        raise ValueError("RAG_EVALUATION_PREDICTION_MISMATCH")
    answerable = [case for case in cases if case.relevant_chunks]
    if not answerable:
        raise ValueError("RAG_EVALUATION_NO_ANSWERABLE_CASE")

    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    multihop_recalls: list[float] = []
    leaks = 0
    # 2、租户泄漏扫描覆盖包括无答案在内的全部预测。
    for case in cases:
        leaks += sum(
            1 for chunk in by_case[case.case_id].ranked_chunks if chunk.tenant_id != case.tenant_id
        )
    # 3、可回答样本按精确document/version/chunk三元组匹配，不能只匹配文档标题。
    for case in answerable:
        relevant = {_key(item) for item in case.relevant_chunks}
        ranking = tuple(_ranked_key(item) for item in by_case[case.case_id].ranked_chunks)
        recalls.append(len(relevant.intersection(ranking[:5])) / len(relevant))
        first_rank = next((index for index, key in enumerate(ranking, 1) if key in relevant), None)
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        relevance = {_key(item): item.relevance for item in case.relevant_chunks}
        dcg = sum(
            (2 ** relevance.get(key, 0) - 1) / math.log2(index + 1)
            for index, key in enumerate(ranking[:5], 1)
        )
        ideal = sorted(relevance.values(), reverse=True)[:5]
        idcg = sum((2**grade - 1) / math.log2(index + 1) for index, grade in enumerate(ideal, 1))
        ndcgs.append(0.0 if idcg == 0 else dcg / idcg)
        if case.category is EvaluationCategory.MULTI_HOP:
            multihop_recalls.append(len(relevant.intersection(ranking[:8])) / len(relevant))
    # 4、各指标独立平均，避免用最终答案质量掩盖检索问题。
    return RetrievalMetrics(
        recall_at_5=sum(recalls) / len(recalls),
        mrr=sum(reciprocal_ranks) / len(reciprocal_ranks),
        ndcg_at_5=sum(ndcgs) / len(ndcgs),
        multihop_recall_at_8=(
            sum(multihop_recalls) / len(multihop_recalls) if multihop_recalls else 0.0
        ),
        cross_tenant_leaks=leaks,
        evaluated_answerable_cases=len(answerable),
    )


def evaluate_generation(judgments: Sequence[GenerationJudgment]) -> GenerationMetrics:
    """聚合忠实度、相关性、引用覆盖和不支持声明率。"""
    # 1、空评审集不能产生看似通过的零除默认值。
    if not judgments:
        raise ValueError("RAG_GENERATION_EVALUATION_EMPTY")
    count = len(judgments)
    # 2、不支持声明率按“包含任一不支持声明的回答比例”计算。
    return GenerationMetrics(
        faithfulness=sum(item.faithful for item in judgments) / count,
        answer_relevance=sum(item.relevant for item in judgments) / count,
        citation_coverage=sum(item.citation_coverage for item in judgments) / count,
        unsupported_claim_rate=sum(item.unsupported_claims > 0 for item in judgments) / count,
    )


def chunking_grid() -> tuple[tuple[int, int], ...]:
    """返回Task 8冻结的十二种目标token与重叠百分比组合。"""
    # 1、笛卡尔积顺序固定，报告可跨运行稳定比较。
    return tuple((target, overlap) for target in (256, 384, 512, 640) for overlap in (10, 15, 20))


def select_chunking_configuration(scores: Iterable[ChunkGridScore]) -> ChunkGridScore:
    """按Recall@5优先、NDCG@5次优并以更小Chunk作稳定平局决胜。"""
    # 1、冻结输入并拒绝缺少十二种网格结果的报告。
    values = tuple(scores)
    expected = set(chunking_grid())
    actual = {(item.target_tokens, item.overlap_percent) for item in values}
    if actual != expected or len(values) != len(expected):
        raise ValueError("RAG_CHUNK_GRID_INCOMPLETE")
    # 2、主次指标相同则优先更小target和更小overlap以降低延迟与重复索引。
    return max(
        values,
        key=lambda item: (
            item.recall_at_5,
            item.ndcg_at_5,
            -item.target_tokens,
            -item.overlap_percent,
        ),
    )


def write_grid_report(path: Path, scores: Sequence[ChunkGridScore]) -> ChunkGridScore:
    """把完整网格与选中配置写入版本化JSON报告。"""
    # 1、先执行完整性与选择规则，避免写入半份报告。
    selected = select_chunking_configuration(scores)
    payload = {
        "reportVersion": selected.report_version,
        "selectionRule": "recall_at_5 desc, ndcg_at_5 desc, target_tokens asc, overlap asc",
        "selected": selected.model_dump(mode="json"),
        "scores": [item.model_dump(mode="json") for item in scores],
    }
    # 2、评测命令只写调用方明确给出的报告路径，不修改运行配置。
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return selected


def _key(item: RelevantChunk) -> tuple[str, int, str]:
    """把人工标注转换为精确匹配键。"""
    # 1、版本是键的一部分，旧版本不能算命中。
    return item.document_id, item.version, item.chunk_id


def _ranked_key(item: RankedChunk) -> tuple[str, int, str]:
    """把预测Chunk转换为精确匹配键。"""
    # 1、保持与人工标注相同字段顺序。
    return item.document_id, item.version, item.chunk_id
