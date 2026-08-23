"""把锁定的BGE-M3模型适配为有界、归一化且带指纹的Embedding提供者。"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from aicare_agent_service.rag.errors import RagError, RagErrorCode

if TYPE_CHECKING:
    from aicare_agent_service.rag.model_runtime import BgeModelRuntime

_REVISION = re.compile(r"^[a-f0-9]{40}$")
_DIMENSIONS = 1024


def model_fingerprint(model_id: str, revision: str, contract: str) -> str:
    """由固定模型ID和精确Git revision生成索引兼容指纹。"""
    # 1、只有完整SHA revision可以参与生产指纹，拒绝分支名和短哈希。
    if not model_id.strip() or not contract.strip() or not _REVISION.fullmatch(revision):
        raise ValueError("RAG_MODEL_REVISION_INVALID")
    # 2、加入输出契约，避免同revision的Embedding和重排结果误用同一指纹。
    payload = f"{model_id}@{revision}:{contract}".encode()
    return hashlib.sha256(payload).hexdigest()


class BgeM3EmbeddingProvider:
    """通过共享有界运行时执行BGE-M3文档和查询向量化。"""

    dimensions = _DIMENSIONS

    def __init__(
        self,
        *,
        runtime: BgeModelRuntime,
        model_id: str,
        revision: str,
        expected_revision: str,
        batch_size: int,
    ) -> None:
        """绑定已加载模型、锁定revision、批量上限和模型指纹。"""
        # 1、运行配置revision必须与模型锁期望完全相同。
        if revision != expected_revision:
            raise ValueError("RAG_MODEL_REVISION_MISMATCH")
        if batch_size < 1:
            raise ValueError("RAG_EMBEDDING_BATCH_INVALID")
        # 2、保存共享运行时和不可变模型身份。
        self._runtime = runtime
        self._batch_size = batch_size
        self.model_fingerprint = model_fingerprint(model_id, revision, "dense:1024")

    async def embed_documents(
        self,
        texts: Sequence[str],
        *,
        deadline: float,
    ) -> tuple[tuple[float, ...], ...]:
        """在绝对deadline内批量生成并校验文档向量。"""
        # 1、空批次、超量批次和空正文都在进入执行器前拒绝。
        values = tuple(texts)
        if not 1 <= len(values) <= self._batch_size or any(not text.strip() for text in values):
            raise ValueError("RAG_EMBEDDING_BATCH_INVALID")
        # 2、同步模型推理只能通过共享有界运行时执行。
        try:
            result = await self._runtime.run(
                lambda: self._runtime.embedding_model.encode(
                    list(values),
                    batch_size=min(self._batch_size, len(values)),
                    max_length=640,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                ),
                deadline=deadline,
            )
        except TimeoutError as exc:
            raise RagError(RagErrorCode.RETRIEVAL_TIMEOUT) from exc
        except Exception as exc:
            raise RagError(RagErrorCode.MODEL_UNAVAILABLE) from exc
        # 3、仅接受dense_vecs并转换为普通有限float、精确1024维、L2归一化向量。
        try:
            raw_vectors: Any = result["dense_vecs"]
            if hasattr(raw_vectors, "tolist"):
                raw_vectors = raw_vectors.tolist()
            if len(raw_vectors) != len(values):
                raise ValueError
            normalized = tuple(self._normalize_vector(vector) for vector in raw_vectors)
        except Exception as exc:
            raise RagError(RagErrorCode.MODEL_UNAVAILABLE) from exc
        return normalized

    async def embed_query(self, text: str, *, deadline: float) -> tuple[float, ...]:
        """使用与文档完全相同的模型身份和归一化规则生成查询向量。"""
        # 1、复用批量实现，确保索引和查询不存在预处理漂移。
        vectors = await self.embed_documents((text,), deadline=deadline)
        # 2、单查询始终返回第一条精确向量。
        return vectors[0]

    @staticmethod
    def _normalize_vector(vector: Sequence[float]) -> tuple[float, ...]:
        """验证维度和有限值后执行L2归一化。"""
        # 1、第三方输出必须精确匹配BGE-M3的1024维契约。
        values = tuple(float(value) for value in vector)
        if len(values) != _DIMENSIONS or not all(math.isfinite(value) for value in values):
            raise ValueError("RAG_EMBEDDING_INVALID")
        # 2、零向量不能形成有效余弦相似度，必须失败而不是写入ES。
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("RAG_EMBEDDING_INVALID")
        # 3、返回普通Python float元组，隔离NumPy/PyTorch对象生命周期。
        return tuple(value / norm for value in values)
