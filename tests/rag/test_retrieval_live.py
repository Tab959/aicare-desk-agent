"""真实执行DeepSeek改写、BGE向量/精排、ES Hybrid Search与LangSmith回查。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import langsmith as ls
import pytest
from elasticsearch import AsyncElasticsearch
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client

from aicare_agent_service.config import Settings
from aicare_agent_service.models.deepseek import DeepSeekModelProvider
from aicare_agent_service.rag.contracts import (
    IndexedDocument,
    KnowledgeChunk,
    KnowledgeMetadata,
    RetrievalFilter,
    RetrievalQuery,
)
from aicare_agent_service.rag.elasticsearch_store import ElasticsearchKnowledgeIndex
from aicare_agent_service.rag.embeddings import model_fingerprint
from aicare_agent_service.rag.index_manager import ElasticsearchIndexManager
from aicare_agent_service.rag.model_runtime import create_rag_resources
from aicare_agent_service.rag.query_rewrite import QueryRewriter
from aicare_agent_service.rag.retriever import HybridKnowledgeRetriever

LIVE_FLAG = "AICARE_RUN_RAG_RETRIEVAL_LIVE"
LIVE_RUN_NAME = "rag.query.rewrite"
PASSWORD_CANARY = "task8f-live-password-canary-5921"


pytestmark = pytest.mark.skipif(
    os.getenv(LIVE_FLAG) != "1",
    reason="真实RAG检索测试需要显式开启DeepSeek、LangSmith、BGE和ES",
)


def _admin_client(settings: Settings) -> AsyncElasticsearch:
    """用独立管理账号创建只供测试索引初始化和清理的TLS客户端。"""
    # 1、管理凭据、节点和CA缺一不可，不允许回退在线账号。
    assert settings.elasticsearch_admin_username
    assert settings.elasticsearch_admin_password
    assert settings.elasticsearch_ca_cert_path
    # 2、返回不自动重试的显式管理连接。
    return AsyncElasticsearch(
        hosts=[str(node) for node in settings.elasticsearch_node_urls],
        basic_auth=(
            settings.elasticsearch_admin_username,
            settings.elasticsearch_admin_password.get_secret_value(),
        ),
        ca_certs=str(settings.elasticsearch_ca_cert_path),
        verify_certs=True,
        request_timeout=10,
        retry_on_timeout=False,
        max_retries=0,
    )


def _chunk(
    *,
    tenant_id: str,
    document_id: str,
    content: str,
    embedding: tuple[float, ...],
    fingerprint: str,
) -> IndexedDocument:
    """把固定测试正文和真实BGE向量封装成单Chunk索引文档。"""
    # 1、所有文档属于同一测试知识库，检索时由代码强制过滤。
    metadata = KnowledgeMetadata(
        tenant_id=tenant_id,
        knowledge_base_id="kb-task8f-live",
        document_id=document_id,
        version=1,
        language="zh-CN",
        category="DELIVERY_POLICY",
    )
    checksum = hashlib.sha256(content.encode()).hexdigest()
    # 2、Chunk ID和checksum均确定性生成，向量使用真实锁定BGE-M3输出。
    chunk = KnowledgeChunk(
        metadata=metadata,
        chunk_id=hashlib.sha256(f"{tenant_id}:{document_id}:{checksum}".encode()).hexdigest(),
        title_path=("交付政策", document_id),
        ordinal=1,
        content=content,
        token_count=32,
        content_checksum=checksum,
        embedding=embedding,
    )
    return IndexedDocument(
        metadata=metadata,
        embedding_fingerprint=fingerprint,
        chunks=(chunk,),
    )


@pytest.mark.bge_live
@pytest.mark.elasticsearch_integration
@pytest.mark.asyncio
async def test_live_hybrid_retrieval_is_relevant_isolated_and_traced() -> None:
    """验证Task 8F真实全链路，并确认LangSmith只收到脱敏查询。"""
    # 1、预检全部生产配置，显式创建隔离租户索引后再加载重量级模型。
    settings = Settings()
    assert settings.bge_embedding_revision and settings.rag_chunk_hmac_key
    assert settings.langsmith_api_key
    fingerprint = model_fingerprint("BAAI/bge-m3", settings.bge_embedding_revision, "dense:1024")
    tenant_id = f"task8f-live-{uuid.uuid4().hex[:10]}"
    admin = _admin_client(settings)
    manager = ElasticsearchIndexManager(
        client=admin,
        index_prefix=settings.elasticsearch_index_prefix,
        tenant_hmac_key=settings.rag_chunk_hmac_key.get_secret_value().encode(),
        embedding_fingerprint=fingerprint,
    )
    names = None
    resources = None
    try:
        await manager.install_template()
        names = await manager.initialize_tenant(tenant_id)
        resources = await create_rag_resources(settings)
        store = ElasticsearchKnowledgeIndex(
            client=resources.elasticsearch,
            index_prefix=settings.elasticsearch_index_prefix,
            tenant_hmac_key=settings.rag_chunk_hmac_key.get_secret_value().encode(),
            embedding_fingerprint=fingerprint,
        )

        # 2、真实BGE批量向量化相关、近邻业务和干扰正文，再逐文档写入真实ES。
        contents = (
            "Steam礼物由人工客服核对收货地区后，发送到用户自己的Steam账户。",
            "CDK商品支付完成后自动发放激活码，用户需要在Steam客户端中兑换。",
            "平台首页会展示游戏折扣和二十四小时销量排行榜。",
        )
        deadline = asyncio.get_running_loop().time() + 120
        vectors = await resources.embeddings.embed_documents(contents, deadline=deadline)
        document_ids = ("steam-gift-delivery", "cdk-delivery", "sales-ranking")
        for document_id, content, vector in zip(document_ids, contents, vectors, strict=True):
            result = await store.replace_document(
                _chunk(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    content=content,
                    embedding=vector,
                    fingerprint=fingerprint,
                )
            )
            assert result.status.value == "INDEXED"

        # 3、真实DeepSeek改写只接收脱敏文本，检索同时执行BM25、HNSW、RRF和BGE精排。
        retriever = HybridKnowledgeRetriever(
            rewriter=QueryRewriter(model_provider=DeepSeekModelProvider(settings)),
            embeddings=resources.embeddings,
            index=store,
            reranker=resources.reranker,
            deadline_seconds=120,
            evidence_score_threshold=settings.rag_evidence_score_threshold,
        )
        query = RetrievalQuery(
            tenant_id=tenant_id,
            text=f"密码={PASSWORD_CANARY}，别人把游戏送到我的Steam账号时是怎么交付的？",
            filters=RetrievalFilter(
                knowledge_base_ids=("kb-task8f-live",),
                languages=("zh-CN",),
                categories=("DELIVERY_POLICY",),
            ),
            candidate_limit=40,
            result_limit=3,
        )
        langsmith_client = Client(api_key=settings.langsmith_api_key.get_secret_value())
        projects = list(langsmith_client.list_projects(name=settings.langsmith_project, limit=1))
        assert len(projects) == 1
        started_at = datetime.now(UTC) - timedelta(seconds=2)
        with ls.tracing_context(
            client=langsmith_client,
            project_name=settings.langsmith_project,
            enabled=True,
        ):
            retrieval = await retriever.retrieve(query)
        wait_for_all_tracers()

        # 4、相关Steam礼物文档必须排首位，所有证据保持当前租户和发布版本。
        assert retrieval.sufficient_evidence is True
        assert retrieval.chunks[0].metadata.document_id == "steam-gift-delivery"
        assert all(chunk.metadata.tenant_id == tenant_id for chunk in retrieval.chunks)
        assert all(chunk.metadata.version == 1 for chunk in retrieval.chunks)

        # 5、回查真实LangSmith根Trace，确认Prompt版本和敏感canary均符合门禁。
        runs = await _wait_for_trace(
            langsmith_client,
            str(projects[0].id),
            started_at,
        )
        trace = next(run for run in runs if run.name == LIVE_RUN_NAME)
        payload = json.dumps(
            {
                "inputs": trace.inputs,
                "outputs": trace.outputs,
                "error": trace.error,
                "metadata": trace.metadata,
            },
            ensure_ascii=False,
            default=str,
        )
        assert PASSWORD_CANARY not in payload
        assert "[REDACTED_PASSWORD]" in payload
        assert (trace.metadata or {}).get("prompt_version") == "rag-query-rewrite-v1"
        for secret in (
            settings.deepseek_api_key,
            settings.langsmith_api_key,
            settings.elasticsearch_password,
        ):
            if secret is not None:
                assert secret.get_secret_value() not in payload
    finally:
        # 6、按模型→ES→隔离索引管理连接的顺序关闭并只删除本次物理索引。
        if resources is not None:
            await resources.models.close()
            await resources.elasticsearch.close()
        if names is not None:
            await admin.indices.delete(index=names.physical_index, ignore_unavailable=True)
        await admin.close()


async def _wait_for_trace(
    client: Client,
    project_id: str,
    started_at: datetime,
) -> list[Any]:
    """有限轮询LangSmith，等待查询改写根Trace完成索引。"""
    # 1、最多等待30秒，避免外部追踪延迟让测试无限挂起。
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        runs = [
            run
            async for run in client.runs.query(
                project_ids=[project_id],
                is_root=True,
                min_start_time=started_at,
                page_size=10,
                selects=["NAME", "INPUTS", "OUTPUTS", "ERROR", "METADATA", "TAGS"],
            )
        ]
        # 2、只要本次固定run name出现就返回，未出现时短暂等待再查。
        if LIVE_RUN_NAME in {run.name for run in runs}:
            return runs
        await asyncio.sleep(1)
    raise AssertionError("LangSmith未在30秒内返回Task 8F查询改写Trace")
