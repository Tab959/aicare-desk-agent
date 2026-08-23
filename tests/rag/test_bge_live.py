"""使用锁定本地BGE模型验证真实向量稳定性、相关性、重排与资源基线。"""

from __future__ import annotations

import asyncio
import math
import os
import time

import psutil
import pytest

from aicare_agent_service.config import Settings
from aicare_agent_service.contracts import Citation
from aicare_agent_service.rag.chunking import chunk_sections
from aicare_agent_service.rag.contracts import KnowledgeMetadata, ParsedSection, RetrievedChunk
from aicare_agent_service.rag.model_runtime import create_rag_resources


def _candidate(index: int, content: str) -> RetrievedChunk:
    metadata = KnowledgeMetadata(
        tenant_id="tenant-live",
        knowledge_base_id="kb-live",
        document_id=f"doc-{index}",
        version=1,
        language="zh-CN",
        category="DELIVERY_POLICY",
    )
    return RetrievedChunk(
        chunk_id=f"chunk-{index}",
        metadata=metadata,
        content=content,
        citation=Citation(
            document_id=metadata.document_id,
            version=1,
            title_path=("交付政策",),
            source_uri=f"https://kb.example/doc-{index}",
        ),
        fused_score=1.0 - index * 0.1,
    )


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(first * second for first, second in zip(left, right, strict=True))


@pytest.mark.bge_live
@pytest.mark.asyncio
async def test_real_bge_models_are_stable_relevant_and_improve_first_result() -> None:
    if os.getenv("AICARE_RUN_BGE_LIVE", "").lower() != "true":
        pytest.skip("set AICARE_RUN_BGE_LIVE=true to run locked local BGE models")

    process = psutil.Process()
    started = time.perf_counter()
    settings = Settings()
    resources = await create_rag_resources(settings)
    try:
        deadline = asyncio.get_running_loop().time() + 90
        query = await resources.embeddings.embed_query("Steam礼物通常如何交付？", deadline=deadline)
        repeated = await resources.embeddings.embed_query(
            "Steam礼物通常如何交付？", deadline=deadline
        )
        relevant = await resources.embeddings.embed_query(
            "Steam礼物由人工客服确认后发送给用户。", deadline=deadline
        )
        distractor = await resources.embeddings.embed_query(
            "今天天气晴朗，适合外出散步。", deadline=deadline
        )
        assert settings.rag_chunk_hmac_key is not None
        chunks = chunk_sections(
            document_title="Steam礼物交付帮助",
            sections=(
                ParsedSection(
                    metadata=_candidate(9, "占位").metadata,
                    title_path=("交付政策",),
                    ordinal=1,
                    text="Steam礼物由人工客服核验地区后发送。" * 200,
                ),
            ),
            tokenizer=resources.models.embedding_model.tokenizer,
            hmac_key=settings.rag_chunk_hmac_key.get_secret_value().encode(),
        )
        candidates = (
            _candidate(1, "今天天气晴朗，适合外出散步。"),
            _candidate(2, "Steam礼物由人工客服确认地区与库存后发送给用户。"),
        )
        ranked = await resources.reranker.rerank(
            query="Steam礼物如何交付？",
            candidates=candidates,
            deadline=deadline,
        )

        assert math.isclose(_dot(query, repeated), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        assert _dot(query, relevant) > _dot(query, distractor)
        assert len(chunks) > 1
        assert all(chunk.token_count <= 640 for chunk in chunks)
        assert candidates[0].chunk_id == "chunk-1"
        assert ranked[0].chunk_id == "chunk-2"
    finally:
        await resources.models.close()
        await resources.elasticsearch.close()

    memory = process.memory_info()
    peak_bytes = int(getattr(memory, "peak_wset", memory.rss))
    elapsed_seconds = time.perf_counter() - started
    print(
        f"BGE_LIVE elapsed_seconds={elapsed_seconds:.2f} peak_rss_mib={peak_bytes / 1024 / 1024:.1f}"
    )
