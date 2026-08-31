"""验证RAG人工数据集构成、分层指标与固定上线门禁。"""

from collections import Counter
from pathlib import Path

from aicare_agent_service.rag.evaluation import (
    EvaluationCategory,
    GenerationJudgment,
    RankedChunk,
    RetrievalPrediction,
    evaluate_generation,
    evaluate_retrieval,
    load_evaluation_dataset,
)

DATASET = Path(__file__).parents[1] / "fixtures" / "rag" / "evaluation_dataset.jsonl"


def _perfect_predictions():
    """按人工相关顺序构造无跨租户泄漏的基准预测。"""
    # 1、每条预测只使用人工标注的精确文档版本和Chunk。
    return tuple(
        RetrievalPrediction(
            case_id=case.case_id,
            ranked_chunks=tuple(
                RankedChunk(
                    document_id=item.document_id,
                    version=item.version,
                    chunk_id=item.chunk_id,
                    tenant_id=case.tenant_id,
                )
                for item in case.relevant_chunks
            ),
        )
        for case in load_evaluation_dataset(DATASET)
    )


def test_dataset_has_manually_reviewed_category_minimums_and_exact_chunk_labels() -> None:
    """固定数据集必须满足Task 8的40条分类配额。"""
    cases = load_evaluation_dataset(DATASET)
    counts = Counter(case.category for case in cases)

    assert len(cases) >= 40
    assert counts[EvaluationCategory.EXACT_FACT] >= 15
    assert counts[EvaluationCategory.SYNONYM] >= 10
    assert counts[EvaluationCategory.MULTI_HOP] >= 5
    assert counts[EvaluationCategory.BUSINESS_FILTER] >= 5
    assert counts[EvaluationCategory.NO_ANSWER_ADVERSARIAL] >= 5
    assert all(
        item.document_id and item.version >= 1 and item.chunk_id
        for case in cases
        for item in case.relevant_chunks
    )


def test_retrieval_metrics_are_separate_and_enforce_zero_tenant_leak() -> None:
    """完美预测通过门禁，任一跨租户结果独立阻断。"""
    cases = load_evaluation_dataset(DATASET)
    predictions = list(_perfect_predictions())

    passed = evaluate_retrieval(cases, predictions)
    assert passed.recall_at_5 == 1.0
    assert passed.mrr == 1.0
    assert passed.ndcg_at_5 == 1.0
    assert passed.multihop_recall_at_8 == 1.0
    assert passed.passed is True

    first = predictions[0]
    leaked = first.model_copy(
        update={
            "ranked_chunks": first.ranked_chunks
            + (
                RankedChunk(
                    document_id="foreign-doc",
                    version=1,
                    chunk_id="foreign-chunk",
                    tenant_id="tenant-foreign",
                ),
            )
        }
    )
    predictions[0] = leaked
    failed = evaluate_retrieval(cases, predictions)
    assert failed.cross_tenant_leaks == 1
    assert failed.passed is False


def test_generation_metrics_use_fixed_independent_gates() -> None:
    """生成质量不能用一个总相似度替代四项门禁。"""
    judgments = tuple(
        GenerationJudgment(
            case_id=f"case-{index}",
            faithful=True,
            relevant=True,
            citation_coverage=1.0,
            unsupported_claims=0,
        )
        for index in range(20)
    )

    assert evaluate_generation(judgments).passed is True
    failed = judgments[:-1] + (
        judgments[-1].model_copy(update={"citation_coverage": 0.5, "unsupported_claims": 1}),
    )
    metrics = evaluate_generation(failed)
    assert metrics.citation_coverage < 1.0
    assert metrics.unsupported_claim_rate > 0
    assert metrics.passed is False
