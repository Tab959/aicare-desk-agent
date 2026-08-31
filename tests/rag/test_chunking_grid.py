"""验证十二组Chunk网格、确定性选择规则和报告版本。"""

import json
from pathlib import Path

from aicare_agent_service.rag.evaluation import (
    ChunkGridScore,
    chunking_grid,
    select_chunking_configuration,
    write_grid_report,
)


def test_chunk_grid_has_twelve_required_combinations_and_recall_first_selection(tmp_path) -> None:
    """选择器必须优先Recall，再比较NDCG，不能按主观答案观感选择。"""
    assert chunking_grid() == tuple(
        (target, overlap) for target in (256, 384, 512, 640) for overlap in (10, 15, 20)
    )
    scores = [
        ChunkGridScore(
            target_tokens=target,
            overlap_percent=overlap,
            recall_at_5=0.91,
            ndcg_at_5=0.82,
        )
        for target, overlap in chunking_grid()
    ]
    scores[7] = scores[7].model_copy(update={"recall_at_5": 0.94, "ndcg_at_5": 0.81})
    scores[8] = scores[8].model_copy(update={"recall_at_5": 0.94, "ndcg_at_5": 0.89})

    selected = select_chunking_configuration(scores)
    report = tmp_path / "rag-chunking-grid-v1.json"
    written = write_grid_report(report, scores)

    assert (selected.target_tokens, selected.overlap_percent) == (512, 20)
    assert written == selected
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["reportVersion"] == "rag-chunk-grid-v1"
    assert len(payload["scores"]) == 12


def test_tracked_grid_report_covers_the_production_bounded_grid() -> None:
    """版本化报告必须覆盖全部十二组，且任何目标都不能突破640 token。"""
    # 1、读取随代码提交的真实评测报告，防止只测试临时示例报告。
    report_path = Path(__file__).parents[2] / "reports" / "rag-chunking-grid-v1.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    scores = tuple(ChunkGridScore.model_validate(item) for item in payload["scores"])

    # 2、选择器同时验证十二组完整性；硬上限和网格值必须与生产约束一致。
    selected = select_chunking_configuration(scores)
    assert {(item.target_tokens, item.overlap_percent) for item in scores} == set(chunking_grid())
    assert max(item.target_tokens for item in scores) == 640
    assert payload["productionHardLimitTokens"] == 640
    assert payload["selectedGridConfiguration"] == {
        "target_tokens": selected.target_tokens,
        "overlap_percent": selected.overlap_percent,
    }
