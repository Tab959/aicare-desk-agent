"""把锁定BGE Cross-Encoder适配为有界、可追溯且稳定排序的重排器。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

from aicare_agent_service.rag.contracts import RetrievedChunk
from aicare_agent_service.rag.embeddings import model_fingerprint
from aicare_agent_service.rag.errors import RagError, RagErrorCode

if TYPE_CHECKING:
    from aicare_agent_service.rag.model_runtime import BgeModelRuntime


class BgeReranker:
    """只对RRF后的有限候选执行BGE Reranker并保留原Chunk身份。"""

    def __init__(
        self,
        *,
        runtime: BgeModelRuntime,
        model_id: str,
        revision: str,
        expected_revision: str,
        batch_size: int,
        max_candidates: int = 30,
        max_passage_tokens: int = 768,
        result_limit: int = 6,
    ) -> None:
        """绑定共享运行时、模型身份和重排预算。"""
        # 1、模型revision和所有数量预算在首次推理前固定。
        if revision != expected_revision:
            raise ValueError("RAG_MODEL_REVISION_MISMATCH")
        if not 1 <= batch_size <= max_candidates or not 1 <= result_limit <= max_candidates:
            raise ValueError("RAG_RERANK_LIMIT_INVALID")
        if max_candidates > 30 or max_passage_tokens < 1 or max_passage_tokens > 768:
            raise ValueError("RAG_RERANK_LIMIT_INVALID")
        # 2、保存只读配置；fingerprint便于评测和索引运行元数据核对。
        self._runtime = runtime
        self._batch_size = batch_size
        self._max_candidates = max_candidates
        self._max_passage_tokens = max_passage_tokens
        self._result_limit = result_limit
        self.model_fingerprint = model_fingerprint(model_id, revision, "cross-encoder-score")

    def _truncate_passage(self, content: str) -> str:
        """用Reranker自身tokenizer把passage限制在固定token预算内。"""
        # 1、只截断送给模型的副本，RetrievedChunk正文和Citation保持不变。
        tokenizer = self._runtime.reranker_model.tokenizer
        encoded = tokenizer(
            content,
            add_special_tokens=False,
            truncation=True,
            max_length=self._max_passage_tokens,
        )
        token_ids = encoded.get("input_ids")
        if not isinstance(token_ids, list):
            raise TypeError("RAG_RERANK_TOKENIZER_INVALID")
        # 2、decode关闭特殊token和清理，防止无意改变模型输入边界。
        return str(
            tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

    async def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RetrievedChunk],
        deadline: float,
    ) -> tuple[RetrievedChunk, ...]:
        """在绝对deadline内分批打分并返回稳定Top-K。"""
        # 1、只接受非空、有限RRF候选集和非空查询。
        values = tuple(candidates)
        if not query.strip() or not 1 <= len(values) <= self._max_candidates:
            raise ValueError("RAG_RERANK_CANDIDATES_INVALID")
        # 2、构造截断pair并分批执行同步Cross-Encoder。
        try:
            pairs = [[query, self._truncate_passage(chunk.content)] for chunk in values]
            scores: list[float] = []
            for start in range(0, len(pairs), self._batch_size):
                batch = pairs[start : start + self._batch_size]
                # 2、当前批次通过partial按值绑定，工作线程不会读取下一轮循环变量。
                compute_batch_scores = partial(
                    self._runtime.reranker_model.compute_score,
                    batch,
                    normalize=True,
                )
                raw_scores: Any = await self._runtime.run(
                    compute_batch_scores,
                    deadline=deadline,
                )
                if hasattr(raw_scores, "tolist"):
                    raw_scores = raw_scores.tolist()
                if isinstance(raw_scores, (float, int)):
                    raw_scores = [raw_scores]
                if not isinstance(raw_scores, (list, tuple)) or len(raw_scores) != len(batch):
                    raise ValueError("RAG_RERANK_OUTPUT_INVALID")
                scores.extend(float(score) for score in raw_scores)
        except TimeoutError as exc:
            raise RagError(RagErrorCode.RETRIEVAL_TIMEOUT) from exc
        except RagError:
            raise
        except Exception as exc:
            raise RagError(RagErrorCode.MODEL_UNAVAILABLE) from exc
        # 3、NaN/Infinity一律阻断，不能污染排序和后续图状态。
        if len(scores) != len(values) or not all(math.isfinite(score) for score in scores):
            raise RagError(RagErrorCode.MODEL_UNAVAILABLE)
        # 4、相同分数按原RRF顺序稳定决胜，并仅更新rerank_score字段。
        ranked = sorted(enumerate(scores), key=lambda item: (-item[1], item[0]))
        return tuple(
            values[index].model_copy(update={"rerank_score": score})
            for index, score in ranked[: self._result_limit]
        )
