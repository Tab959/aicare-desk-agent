"""真实执行40条检索评测、Knowledge RAG生成审核、readiness与CPU延迟记录。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import statistics
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import langsmith as ls
import pytest
from elasticsearch import AsyncElasticsearch
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client

from aicare_agent_service.config import Settings
from aicare_agent_service.graph.context import AgentRuntimeContext
from aicare_agent_service.graph.state import AgentIdentity
from aicare_agent_service.models.deepseek import DeepSeekModelProvider
from aicare_agent_service.rag.contracts import (
    IndexedDocument,
    KnowledgeChunk,
    KnowledgeMetadata,
    RetrievalFilter,
    RetrievalQuery,
)
from aicare_agent_service.rag.embeddings import model_fingerprint
from aicare_agent_service.rag.evaluation import (
    GenerationJudgment,
    RankedChunk,
    RetrievalPrediction,
    evaluate_generation,
    evaluate_retrieval,
    load_evaluation_dataset,
)
from aicare_agent_service.rag.index_manager import ElasticsearchIndexManager
from aicare_agent_service.rag.model_runtime import create_rag_resources
from aicare_agent_service.subgraphs.knowledge_rag import KnowledgeRagSubgraph

LIVE_FLAG = "AICARE_RUN_RAG_EVALUATION_LIVE"
DATASET = Path(__file__).parents[1] / "fixtures" / "rag" / "evaluation_dataset.jsonl"

pytestmark = pytest.mark.skipif(
    os.getenv(LIVE_FLAG) != "1",
    reason="真实RAG评测需要显式开启DeepSeek、LangSmith、BGE和ES",
)

_CORPUS = {
    "doc-steam-gift": (
        1,
        "kb-delivery",
        "STEAM_GIFT",
        (
            "Steam礼物由人工客服核对收货地区后，发送到用户自己的Steam账户。",
            "Steam礼物受Steam地区限制，收礼账户地区必须与礼物适用地区一致。",
        ),
    ),
    "doc-cdk": (1, "kb-delivery", "CDK", ("CDK支付完成后自动发放激活码，需要在Steam客户端兑换。",)),
    "doc-offline-account": (
        1,
        "kb-account",
        "OFFLINE_ACCOUNT",
        ("离线账号首次登录前应关闭Steam云同步，不允许修改邮箱或密码。",),
    ),
    "doc-download": (
        1,
        "kb-delivery",
        "DOWNLOAD",
        ("下载链接失效时先检查有效期和网络；下载资源仅允许一个设备同时访问。",),
    ),
    "doc-refund": (
        2,
        "kb-after-sales",
        "REFUND",
        ("退款申请应在购买后七天内提交；已经揭示或兑换的CDK不支持无理由退款。",),
    ),
    "doc-wallet": (
        1,
        "kb-trade",
        "WALLET",
        ("余额扣款明细可在余额流水中查看；平台当前未开放余额充值。",),
    ),
    "doc-flash-sale": (
        1,
        "kb-trade",
        "FLASH_SALE",
        ("秒杀订单超时后不能继续支付，系统释放该订单预占的秒杀库存。",),
    ),
    "doc-finished-account": (
        1,
        "kb-account",
        "FINISHED_ACCOUNT",
        ("成品账号交付后建议先修改邮箱和密码，并启用双重验证。",),
    ),
    "doc-after-sales": (
        1,
        "kb-after-sales",
        "TROUBLESHOOTING",
        ("下载问题无法自行恢复时，应提交下载失败售后问题并附带安全错误码。",),
    ),
}


def _admin_client(settings: Settings) -> AsyncElasticsearch:
    """创建隔离索引管理连接。"""
    # 1、管理账号只用于测试前后初始化与精确清理。
    assert settings.elasticsearch_admin_username and settings.elasticsearch_admin_password
    assert settings.elasticsearch_ca_cert_path
    return AsyncElasticsearch(
        hosts=[str(node) for node in settings.elasticsearch_node_urls],
        basic_auth=(
            settings.elasticsearch_admin_username,
            settings.elasticsearch_admin_password.get_secret_value(),
        ),
        ca_certs=str(settings.elasticsearch_ca_cert_path),
        verify_certs=True,
        request_timeout=10,
        max_retries=0,
    )


def _documents(
    tenant_id: str, vectors: list[tuple[float, ...]], fingerprint: str
) -> tuple[IndexedDocument, ...]:
    """把固定语料与真实BGE向量封装成版本化索引文档。"""
    # 1、按语料稳定顺序消费向量，每个Chunk ID与人工数据集精确一致。
    cursor = 0
    documents: list[IndexedDocument] = []
    for document_id, (version, knowledge_base_id, category, contents) in _CORPUS.items():
        metadata = KnowledgeMetadata(
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            version=version,
            language="zh-CN",
            category=category,
        )
        chunks: list[KnowledgeChunk] = []
        for ordinal, content in enumerate(contents, 1):
            checksum = hashlib.sha256(content.encode()).hexdigest()
            chunks.append(
                KnowledgeChunk(
                    metadata=metadata,
                    chunk_id=f"{document_id}-c{ordinal}",
                    title_path=("评测知识", category),
                    ordinal=ordinal,
                    content=content,
                    token_count=40,
                    content_checksum=checksum,
                    embedding=vectors[cursor],
                )
            )
            cursor += 1
        documents.append(
            IndexedDocument(
                metadata=metadata,
                embedding_fingerprint=fingerprint,
                chunks=tuple(chunks),
            )
        )
    return tuple(documents)


@pytest.mark.bge_live
@pytest.mark.elasticsearch_integration
@pytest.mark.asyncio
async def test_live_rag_evaluation_gates_full_chain_readiness_and_latency() -> None:
    """40条真实检索和一条真实生成链必须通过固定质量与安全门禁。"""
    # 1、初始化随机隔离租户和真实生产资源。
    settings = Settings()
    assert settings.bge_embedding_revision and settings.rag_chunk_hmac_key
    assert settings.langsmith_api_key
    fingerprint = model_fingerprint("BAAI/bge-m3", settings.bge_embedding_revision, "dense:1024")
    tenant_id = f"task8h-live-{uuid.uuid4().hex[:10]}"
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

        # 2、一次批量真实Embedding后逐文档原子写入ES。
        contents = [text for _, _, _, values in _CORPUS.values() for text in values]
        embed_started = time.perf_counter()
        vectors: list[tuple[float, ...]] = []
        for offset in range(0, len(contents), settings.rag_embedding_batch_size):
            vectors.extend(
                await resources.embeddings.embed_documents(
                    contents[offset : offset + settings.rag_embedding_batch_size],
                    deadline=asyncio.get_running_loop().time() + 120,
                )
            )
        embedding_ms = (time.perf_counter() - embed_started) * 1000
        for document in _documents(tenant_id, vectors, fingerprint):
            await resources.index.replace_document(document)

        # 3、用40条人工用例真实执行改写、BM25/Dense、RRF和BGE精排。
        cases = tuple(
            case.model_copy(update={"tenant_id": tenant_id})
            for case in load_evaluation_dataset(DATASET)
        )
        predictions: list[RetrievalPrediction] = []
        retrieval_latencies: list[float] = []
        first_chunks = ()
        for case in cases:
            result = await resources.retriever.retrieve(
                RetrievalQuery(
                    tenant_id=tenant_id,
                    text=case.query,
                    filters=RetrievalFilter(
                        knowledge_base_ids=case.knowledge_base_ids,
                        categories=case.categories,
                        languages=("zh-CN",),
                    ),
                    candidate_limit=30,
                    result_limit=8 if case.category.value == "multi_hop" else 6,
                )
            )
            retrieval_latencies.append(float(result.elapsed_ms))
            first_chunks = first_chunks or result.chunks
            predictions.append(
                RetrievalPrediction(
                    case_id=case.case_id,
                    ranked_chunks=tuple(
                        RankedChunk(
                            document_id=chunk.metadata.document_id,
                            version=chunk.metadata.version,
                            chunk_id=chunk.chunk_id,
                            tenant_id=chunk.metadata.tenant_id,
                        )
                        for chunk in result.chunks
                    ),
                )
            )
        retrieval_metrics = evaluate_retrieval(cases, predictions)
        assert retrieval_metrics.passed is True
        assert retrieval_metrics.multihop_recall_at_8 >= 0.90

        # 4、真实Knowledge子图完成DeepSeek回答与忠实度审核，并由LangSmith追踪。
        identity = AgentIdentity(
            tenant_id=tenant_id,
            customer_id="customer-task8h",
            conversation_id="conversation-task8h",
            run_id="run-task8h",
            trigger_message_id="message-task8h",
            trigger_sequence=1,
        )
        context = AgentRuntimeContext(
            expected_identity=identity,
            java_client=SimpleNamespace(),
            model_provider=DeepSeekModelProvider(settings),
            request_deadline=datetime.now(UTC) + timedelta(seconds=180),
            knowledge_retriever=resources.retriever,
        )
        langsmith_client = Client(api_key=settings.langsmith_api_key.get_secret_value())
        e2e_started = time.perf_counter()
        with ls.tracing_context(
            client=langsmith_client,
            project_name=settings.langsmith_project,
            enabled=True,
        ):
            answer = await KnowledgeRagSubgraph().ainvoke(
                query="Steam礼物如何交付？",
                context=context,
            )
        wait_for_all_tracers()
        e2e_ms = (time.perf_counter() - e2e_started) * 1000
        generation_metrics = evaluate_generation(
            (
                GenerationJudgment(
                    case_id="generation-steam-gift",
                    faithful=answer.sufficient_evidence,
                    relevant="Steam" in answer.answer.content and "[K" in answer.answer.content,
                    citation_coverage=1.0 if answer.citations else 0.0,
                    unsupported_claims=0 if answer.sufficient_evidence else 1,
                ),
            )
        )
        assert generation_metrics.passed is True

        # 5、对固定候选重复精排，记录CPU环境耗时分位数但不宣称容量上限。
        rerank_latencies: list[float] = []
        for _ in range(5):
            started = time.perf_counter()
            await resources.reranker.rerank(
                query="Steam礼物如何交付？",
                candidates=first_chunks,
                deadline=asyncio.get_running_loop().time() + 30,
            )
            rerank_latencies.append((time.perf_counter() - started) * 1000)
        readiness = await resources.readiness.check()
        assert readiness.ready is True

        # 6、输出可复制进版本化报告的真实指标，不包含身份、正文、向量或连接信息。
        report = {
            "reportVersion": "rag-evaluation-v1",
            "datasetCases": len(cases),
            "retrieval": retrieval_metrics.model_dump(mode="json"),
            "generation": generation_metrics.model_dump(mode="json"),
            "cpuLatencyMs": {
                "embeddingBatch": round(embedding_ms, 2),
                "retrievalP50": round(_percentile(retrieval_latencies, 0.50), 2),
                "retrievalP95": round(_percentile(retrieval_latencies, 0.95), 2),
                "rerankP50": round(_percentile(rerank_latencies, 0.50), 2),
                "rerankP95": round(_percentile(rerank_latencies, 0.95), 2),
                "endToEndP50": round(e2e_ms, 2),
                "endToEndP95": round(e2e_ms, 2),
            },
            "capacityClaim": False,
        }
        print("TASK8H_REPORT=" + json.dumps(report, ensure_ascii=False, sort_keys=True))
    finally:
        # 7、关闭资源并只删除本次随机租户的物理索引。
        if resources is not None:
            await resources.models.close()
            await resources.elasticsearch.close()
        if names is not None:
            await admin.indices.delete(index=names.physical_index, ignore_unavailable=True)
        await admin.close()


def _percentile(values: list[float], quantile: float) -> float:
    """用包含端点的线性分位数记录小样本延迟。"""
    # 1、statistics.quantiles需要至少两个值，单值直接返回本身。
    if len(values) == 1:
        return values[0]
    # 2、把0.50/0.95映射到百分位切点并返回对应值。
    cuts = statistics.quantiles(values, n=100, method="inclusive")
    return cuts[max(0, min(98, round(quantile * 100) - 1))]
