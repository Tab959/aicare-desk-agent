"""验证 ES 文档生命周期的外部版本、Bulk 检查、租户 routing 和安全查询。"""

from __future__ import annotations

from typing import Any

import pytest
from elasticsearch import ConflictError, NotFoundError

from aicare_agent_service.rag.contracts import (
    IndexedDocument,
    IndexStatus,
    KnowledgeChunk,
    KnowledgeMetadata,
)
from aicare_agent_service.rag.elasticsearch_mapping import (
    build_tenant_index_names,
    derive_tenant_namespace_key,
)
from aicare_agent_service.rag.elasticsearch_store import ElasticsearchKnowledgeIndex
from aicare_agent_service.rag.errors import RagError, RagErrorCode

FINGERPRINT = "b" * 64
HMAC_KEY = b"store-test-tenant-hmac-key-32-bytes!!"


def indexed_document(*, version: int = 2, chunks: int = 2) -> IndexedDocument:
    metadata = KnowledgeMetadata(
        tenant_id="tenant-a",
        knowledge_base_id="kb-help",
        document_id="doc-delivery",
        version=version,
        language="zh-CN",
        category="DELIVERY",
        purchase_method="STEAM_GIFT",
    )
    values = tuple(
        KnowledgeChunk(
            metadata=metadata,
            chunk_id=f"chunk-v{version}-{ordinal}",
            title_path=("交付",),
            ordinal=ordinal,
            content=f"Steam礼物交付说明 {ordinal}",
            token_count=10,
            content_checksum=f"{ordinal:064x}",
            embedding=tuple([1.0] + [0.0] * 1023),
        )
        for ordinal in range(1, chunks + 1)
    )
    return IndexedDocument(metadata=metadata, embedding_fingerprint=FINGERPRINT, chunks=values)


class RecordingIndices:
    """记录 alias 与 mapping 校验调用的最小 ES 边界替身。"""

    def __init__(self) -> None:
        self.alias_missing = False
        self.schema_version = 1
        self.fingerprint = FINGERPRINT

    async def get_alias(self, **kwargs: Any) -> dict[str, Any]:
        if self.alias_missing:
            raise NotFoundError("missing", meta=None, body=None)
        return {
            "aicare-kb-index": {
                "aliases": {kwargs["name"]: {"is_write_index": kwargs["name"].endswith("write")}}
            }
        }

    async def get_mapping(self, **_: Any) -> dict[str, Any]:
        return {
            "aicare-kb-index": {
                "mappings": {
                    "_meta": {
                        "aicare_schema_version": self.schema_version,
                        "embedding_fingerprint": self.fingerprint,
                    }
                }
            }
        }


class RecordingClient:
    """模拟 ES 边界响应，并保存写入参数供确定性断言。"""

    def __init__(self) -> None:
        self.indices = RecordingIndices()
        self.index_calls: list[dict[str, Any]] = []
        self.bulk_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.marker: dict[str, Any] | None = None
        self.marker_version = 0
        self.bulk_error = False

    async def get(self, **_: Any) -> dict[str, Any]:
        if self.marker is None:
            raise NotFoundError("missing", meta=None, body=None)
        return {"_source": self.marker, "_version": self.marker_version}

    async def index(self, **kwargs: Any) -> dict[str, Any]:
        self.index_calls.append(kwargs)
        version = int(kwargs["version"])
        if version < self.marker_version:
            raise ConflictError("version conflict", meta=None, body=None)
        self.marker = dict(kwargs["document"])
        self.marker_version = version
        return {"result": "created", "_version": version}

    async def bulk(self, **kwargs: Any) -> dict[str, Any]:
        self.bulk_calls.append(kwargs)
        operations = kwargs["operations"]
        items = []
        for index in range(0, len(operations), 2):
            status = 500 if self.bulk_error and index == 0 else 201
            item: dict[str, Any] = {"status": status}
            if status >= 300:
                item["error"] = {"type": "simulated"}
            items.append({"index": item})
        return {"errors": self.bulk_error, "items": items}

    async def delete_by_query(self, **kwargs: Any) -> dict[str, Any]:
        self.delete_calls.append(kwargs)
        return {"deleted": 1, "version_conflicts": 0, "failures": []}


def store(client: RecordingClient) -> ElasticsearchKnowledgeIndex:
    return ElasticsearchKnowledgeIndex(
        client=client,
        index_prefix="aicare-kb",
        tenant_hmac_key=HMAC_KEY,
        embedding_fingerprint=FINGERPRINT,
    )


@pytest.mark.asyncio
async def test_replace_uses_alias_routing_external_version_and_safe_source() -> None:
    client = RecordingClient()
    result = await store(client).replace_document(indexed_document())
    names = build_tenant_index_names(
        prefix="aicare-kb",
        tenant_id="tenant-a",
        hmac_key=derive_tenant_namespace_key(HMAC_KEY),
        embedding_fingerprint=FINGERPRINT,
        generation=1,
    )

    assert result.status is IndexStatus.INDEXED
    assert result.indexed_chunks == 2
    assert client.index_calls[0]["version_type"] == "external_gte"
    assert client.index_calls[0]["require_alias"] is True
    assert client.index_calls[0]["routing"] == names.routing
    assert client.bulk_calls[0]["index"] == names.write_alias
    assert client.bulk_calls[0]["routing"] == names.routing
    assert client.bulk_calls[0]["require_alias"] is True
    serialized = repr(client.bulk_calls)
    assert "tenant-a" not in serialized
    assert "embedding" in serialized


@pytest.mark.asyncio
async def test_old_or_duplicate_event_cannot_overwrite_marker() -> None:
    client = RecordingClient()
    target = store(client)
    first = await target.replace_document(indexed_document(version=3))
    duplicate = await target.replace_document(indexed_document(version=3))
    old = await target.replace_document(indexed_document(version=2))

    assert first.status is IndexStatus.INDEXED
    assert duplicate.status is IndexStatus.SKIPPED
    assert old.status is IndexStatus.SKIPPED
    assert duplicate.error_code is None
    assert old.error_code == "RAG_INDEX_VERSION_CONFLICT"
    assert len(client.bulk_calls) == 1


@pytest.mark.asyncio
async def test_partial_bulk_failure_is_failed_and_does_not_cleanup_old_chunks() -> None:
    client = RecordingClient()
    client.bulk_error = True

    result = await store(client).replace_document(indexed_document())

    assert result.status is IndexStatus.FAILED
    assert result.error_code == "RAG_INDEX_UNAVAILABLE"
    assert client.delete_calls == []


@pytest.mark.asyncio
async def test_delete_uses_tombstone_and_never_deletes_newer_versions() -> None:
    client = RecordingClient()
    target = store(client)
    await target.replace_document(indexed_document(version=4))

    old = await target.delete_document("tenant-a", "doc-delivery", 3)
    deleted = await target.delete_document("tenant-a", "doc-delivery", 5)

    assert old.status is IndexStatus.SKIPPED
    assert old.error_code == "RAG_INDEX_VERSION_CONFLICT"
    assert deleted.status is IndexStatus.DELETED
    query = client.delete_calls[-1]["query"]
    assert {"range": {"version": {"lte": 5}}} in query["bool"]["filter"]
    assert client.delete_calls[-1]["routing"]


@pytest.mark.asyncio
async def test_missing_alias_and_old_mapping_fail_closed() -> None:
    missing = RecordingClient()
    missing.indices.alias_missing = True
    with pytest.raises(RagError) as missing_error:
        await store(missing).replace_document(indexed_document())
    assert missing_error.value.code is RagErrorCode.INDEX_UNAVAILABLE

    old_schema = RecordingClient()
    old_schema.indices.schema_version = 0
    with pytest.raises(RagError) as schema_error:
        await store(old_schema).replace_document(indexed_document())
    assert schema_error.value.code is RagErrorCode.INDEX_UNAVAILABLE

    wrong_model = RecordingClient()
    wrong_model.indices.fingerprint = "c" * 64
    with pytest.raises(RagError) as model_error:
        await store(wrong_model).replace_document(indexed_document())
    assert model_error.value.code is RagErrorCode.INDEX_UNAVAILABLE
