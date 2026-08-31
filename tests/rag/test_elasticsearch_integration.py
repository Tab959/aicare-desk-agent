"""在真实TLS Elasticsearch上验证索引初始化、版本、别名与租户隔离。"""

from __future__ import annotations

import os
import ssl
import uuid
from pathlib import Path

import pytest
from elastic_transport import TlsError
from elasticsearch import AsyncElasticsearch, AuthenticationException, BadRequestError

from aicare_agent_service.config import Settings
from aicare_agent_service.rag.contracts import (
    IndexedDocument,
    IndexStatus,
    KnowledgeChunk,
    KnowledgeMetadata,
    RetrievalFilter,
    RetrievalQuery,
)
from aicare_agent_service.rag.elasticsearch_store import ElasticsearchKnowledgeIndex
from aicare_agent_service.rag.embeddings import model_fingerprint
from aicare_agent_service.rag.index_manager import ElasticsearchIndexManager


def _client(settings: Settings, *, admin: bool) -> AsyncElasticsearch:
    """使用在线或管理账号创建证书校验开启的真实异步客户端。"""
    # 1、测试不接受缺失凭据，也不回退匿名或明文连接。
    username = settings.elasticsearch_admin_username if admin else settings.elasticsearch_username
    password = settings.elasticsearch_admin_password if admin else settings.elasticsearch_password
    assert username and password and settings.elasticsearch_ca_cert_path
    # 2、连接参数与生产客户端保持一致。
    return AsyncElasticsearch(
        hosts=[str(node) for node in settings.elasticsearch_node_urls],
        basic_auth=(username, password.get_secret_value()),
        ca_certs=str(settings.elasticsearch_ca_cert_path),
        verify_certs=True,
        request_timeout=10,
        retry_on_timeout=False,
        max_retries=0,
    )


def _document(tenant_id: str, *, version: int, fingerprint: str) -> IndexedDocument:
    """构造包含精确术语和合法1024维向量的隔离测试文档。"""
    # 1、每租户使用相同业务ID，证明隔离依赖租户索引而非碰巧不同ID。
    metadata = KnowledgeMetadata(
        tenant_id=tenant_id,
        knowledge_base_id="kb-integration",
        document_id="shared-document",
        version=version,
        language="zh-CN",
        category="DELIVERY",
    )
    # 2、确定性Chunk和向量便于重复replace验证。
    chunk = KnowledgeChunk(
        metadata=metadata,
        chunk_id=f"{tenant_id[-8:]}-v{version}-chunk",
        title_path=("集成测试",),
        ordinal=1,
        content=f"租户隔离术语 {tenant_id[-8:]} Steam礼物人工交付",
        token_count=16,
        content_checksum=(f"{version:x}" * 64)[:64],
        embedding=tuple([1.0] + [0.0] * 1023),
    )
    return IndexedDocument(
        metadata=metadata,
        embedding_fingerprint=fingerprint,
        chunks=(chunk,),
    )


@pytest.mark.elasticsearch_integration
@pytest.mark.asyncio
async def test_real_es_document_lifecycle_alias_switch_and_tenant_isolation(
    tmp_path: Path,
) -> None:
    if os.getenv("AICARE_RUN_ELASTICSEARCH_INTEGRATION", "").lower() != "true":
        pytest.skip("set AICARE_RUN_ELASTICSEARCH_INTEGRATION=true")
    settings = Settings()
    assert settings.bge_embedding_revision and settings.rag_chunk_hmac_key
    fingerprint = model_fingerprint("BAAI/bge-m3", settings.bge_embedding_revision, "dense:1024")
    suffix = uuid.uuid4().hex[:10]
    tenants = (f"integration-a-{suffix}", f"integration-b-{suffix}")
    admin = _client(settings, admin=True)
    online = _client(settings, admin=False)
    manager = ElasticsearchIndexManager(
        client=admin,
        index_prefix=settings.elasticsearch_index_prefix,
        tenant_hmac_key=settings.rag_chunk_hmac_key.get_secret_value().encode(),
        embedding_fingerprint=fingerprint,
    )
    created: list[str] = []
    try:
        # 1、显式安装模板并初始化两个租户，线上账号只做文档读写。
        await manager.install_template()
        for tenant in tenants:
            names = await manager.initialize_tenant(tenant)
            created.append(names.physical_index)
        stores = [
            ElasticsearchKnowledgeIndex(
                client=online,
                index_prefix=settings.elasticsearch_index_prefix,
                tenant_hmac_key=settings.rag_chunk_hmac_key.get_secret_value().encode(),
                embedding_fingerprint=fingerprint,
            )
            for _ in tenants
        ]
        first = await stores[0].replace_document(
            _document(tenants[0], version=1, fingerprint=fingerprint)
        )
        duplicate = await stores[0].replace_document(
            _document(tenants[0], version=1, fingerprint=fingerprint)
        )
        upgraded = await stores[0].replace_document(
            _document(tenants[0], version=2, fingerprint=fingerprint)
        )
        old = await stores[0].replace_document(
            _document(tenants[0], version=1, fingerprint=fingerprint)
        )
        second = await stores[1].replace_document(
            _document(tenants[1], version=1, fingerprint=fingerprint)
        )
        assert [first.status, duplicate.status, upgraded.status, old.status, second.status] == [
            IndexStatus.INDEXED,
            IndexStatus.SKIPPED,
            IndexStatus.INDEXED,
            IndexStatus.SKIPPED,
            IndexStatus.INDEXED,
        ]

        # 2、严格Mapping拒绝任意字段和错误向量维度。
        first_names = manager.names(tenants[0])
        with pytest.raises(BadRequestError):
            await online.index(
                index=first_names.write_alias,
                id="invalid-dynamic",
                routing=first_names.routing,
                require_alias=True,
                document={"unexpected": "blocked"},
            )
        with pytest.raises(BadRequestError):
            await online.index(
                index=first_names.write_alias,
                id="invalid-vector",
                routing=first_names.routing,
                require_alias=True,
                document={"embedding": [1.0, 2.0]},
            )

        # 3、每租户read alias只能看到自己的namespace；原始tenant ID未写入source。
        for tenant in tenants:
            names = manager.names(tenant)
            result = await online.search(
                index=names.read_alias,
                routing=names.routing,
                query={"term": {"doc_kind": "chunk"}},
                source_excludes=("embedding",),
            )
            hits = result["hits"]["hits"]
            assert len(hits) == 1
            assert all(hit["_source"]["tenant_namespace"] == names.tenant_namespace for hit in hits)
            assert tenant not in repr(hits)

        # 4、真实BM25命中精确术语，HNSW使用同一预过滤且都只返回当前租户。
        retrieval_query = RetrievalQuery(
            tenant_id=tenants[0],
            text="Steam礼物人工交付",
            filters=RetrievalFilter(
                knowledge_base_ids=("kb-integration",),
                languages=("zh-CN",),
            ),
            candidate_limit=40,
            result_limit=6,
        )
        sparse = await stores[0].sparse_search(retrieval_query)
        dense = await stores[0].dense_search(
            retrieval_query,
            tuple([1.0] + [0.0] * 1023),
            num_candidates=120,
        )
        assert sparse and dense
        assert all(chunk.metadata.tenant_id == tenants[0] for chunk in (*sparse, *dense))
        assert all(chunk.metadata.version == 2 for chunk in (*sparse, *dense))

        # 5、创建新代次后原子切换；旧索引仍保留供窗口期回滚。
        generation_two = await manager.create_generation(tenants[0], generation=2)
        created.append(generation_two.physical_index)
        await manager.switch_generation(tenants[0], generation=2)
        await manager.validate_tenant(tenants[0])
        assert bool(await admin.indices.exists(index=created[0])) is True

        # 6、删除墓碑不允许旧版本回写，关闭并重建连接后别名仍可验证。
        deleted = await stores[1].delete_document(tenants[1], "shared-document", 2)
        stale = await stores[1].replace_document(
            _document(tenants[1], version=1, fingerprint=fingerprint)
        )
        assert deleted.status is IndexStatus.DELETED
        assert stale.status is IndexStatus.SKIPPED
        await online.close()
        online = _client(settings, admin=False)
        reconnect_manager = ElasticsearchIndexManager(
            client=online,
            index_prefix=settings.elasticsearch_index_prefix,
            tenant_hmac_key=settings.rag_chunk_hmac_key.get_secret_value().encode(),
            embedding_fingerprint=fingerprint,
        )
        await reconnect_manager.validate_tenant(tenants[0])

        # 7、错误认证必须失败，不能回退到匿名访问。
        bad = AsyncElasticsearch(
            hosts=[str(node) for node in settings.elasticsearch_node_urls],
            basic_auth=(settings.elasticsearch_username or "", "wrong-password"),
            ca_certs=str(settings.elasticsearch_ca_cert_path),
            verify_certs=True,
        )
        try:
            with pytest.raises(AuthenticationException):
                await bad.info()
        finally:
            await bad.close()

        # 8、伪造CA不能建立TLS连接，证书校验失败不得降级verify_certs=false。
        invalid_ca = tmp_path / "invalid-ca.pem"
        invalid_ca.write_text("not-a-certificate", encoding="utf-8")
        with pytest.raises((TlsError, ssl.SSLError)):
            AsyncElasticsearch(
                hosts=[str(node) for node in settings.elasticsearch_node_urls],
                basic_auth=(
                    settings.elasticsearch_username or "",
                    settings.elasticsearch_password.get_secret_value()
                    if settings.elasticsearch_password
                    else "",
                ),
                ca_certs=str(invalid_ca),
                verify_certs=True,
            )
    finally:
        # 9、只清理本测试明确记录的物理索引，保留生产模板和其他租户数据。
        await online.close()
        for index_name in created:
            if index_name.startswith(f"{settings.elasticsearch_index_prefix}-"):
                await admin.indices.delete(index=index_name, ignore_unavailable=True)
        await admin.close()
