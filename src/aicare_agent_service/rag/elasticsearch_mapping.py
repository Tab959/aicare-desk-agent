"""定义知识索引的严格 Elasticsearch Mapping、模板和租户安全命名规则。"""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from typing import Any

INDEX_SCHEMA_VERSION = 1
EMBEDDING_DIMENSIONS = 1024
_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_PREFIX_PATTERN = re.compile(r"^[a-z][a-z0-9-]{2,31}$")
_TENANT_NAMESPACE_PURPOSE = b"aicare-rag-index-tenant-namespace-v1"


@dataclass(frozen=True, slots=True)
class TenantIndexNames:
    """保存不暴露原始租户ID的物理索引、读写别名和routing。"""

    tenant_namespace: str
    physical_index: str
    read_alias: str
    write_alias: str
    routing: str


def derive_tenant_namespace_key(root_key: bytes) -> bytes:
    """从RAG根密钥派生只供索引租户命名使用的独立子密钥。"""
    # 1、根密钥至少32字节，不能使用短密码直接参与命名HMAC。
    if len(root_key) < 32:
        raise ValueError("RAG_TENANT_HMAC_KEY_INVALID")
    # 2、固定用途标签实现域分离，避免与Chunk ID HMAC复用同一实际密钥。
    return hmac.new(root_key, _TENANT_NAMESPACE_PURPOSE, hashlib.sha256).digest()


def build_tenant_index_names(
    *,
    prefix: str,
    tenant_id: str,
    hmac_key: bytes,
    embedding_fingerprint: str,
    generation: int,
) -> TenantIndexNames:
    """由租户HMAC、schema、Embedding指纹和代次构造稳定索引名。"""
    # 1、先拒绝非法前缀、空身份、弱密钥、错误指纹和非正代次。
    if (
        not _PREFIX_PATTERN.fullmatch(prefix)
        or not tenant_id.strip()
        or len(hmac_key) < 32
        or not _SHA256_PATTERN.fullmatch(embedding_fingerprint)
        or generation < 1
    ):
        raise ValueError("RAG_INDEX_NAME_INVALID")
    # 2、租户命名空间只使用独立HMAC摘要，索引名和routing均不出现原始tenant ID。
    namespace = hmac.new(hmac_key, tenant_id.encode("utf-8"), hashlib.sha256).hexdigest()[:32]
    stable_stem = f"{prefix}-{namespace}"
    physical = (
        f"{stable_stem}-s{INDEX_SCHEMA_VERSION}-e{embedding_fingerprint[:12]}-{generation:06d}"
    )
    # 3、别名跨重建代次和Embedding升级保持稳定，切换时由ES原子更新。
    return TenantIndexNames(
        tenant_namespace=namespace,
        physical_index=physical,
        read_alias=f"{stable_stem}-read",
        write_alias=f"{stable_stem}-write",
        routing=namespace,
    )


def build_index_template(*, prefix: str, embedding_fingerprint: str) -> dict[str, Any]:
    """构造动态字段拒绝、SmartCN/BM25和1024维HNSW共用的模板。"""
    # 1、复用命名校验规则并验证Embedding指纹，避免创建不可被运行时识别的模板。
    if not _PREFIX_PATTERN.fullmatch(prefix) or not _SHA256_PATTERN.fullmatch(
        embedding_fingerprint
    ):
        raise ValueError("RAG_INDEX_TEMPLATE_INVALID")
    # 2、正文保留SmartCN主字段和standard子字段，过滤字段使用精确类型。
    properties: dict[str, Any] = {
        "doc_kind": {"type": "keyword"},
        "tenant_namespace": {"type": "keyword"},
        "knowledge_base_id": {"type": "keyword"},
        "document_id": {"type": "keyword"},
        "version": {"type": "integer"},
        "status": {"type": "keyword"},
        "language": {"type": "keyword"},
        "category": {"type": "keyword"},
        "game_id": {"type": "keyword"},
        "purchase_method": {"type": "keyword"},
        "issue_type": {"type": "keyword"},
        "chunk_id": {"type": "keyword"},
        "ordinal": {"type": "integer"},
        "title_path": {"type": "keyword"},
        "content": {
            "type": "text",
            "analyzer": "smartcn",
            "fields": {"standard": {"type": "text", "analyzer": "standard"}},
        },
        "source_uri": {"type": "keyword", "index": False, "doc_values": False},
        "content_checksum": {"type": "keyword"},
        "document_checksum": {"type": "keyword"},
        "completed": {"type": "boolean"},
        "embedding_fingerprint": {"type": "keyword"},
        "indexed_at": {"type": "date"},
        "embedding": {
            "type": "dense_vector",
            "dims": EMBEDDING_DIMENSIONS,
            "index": True,
            "similarity": "cosine",
            "index_options": {"type": "int8_hnsw"},
        },
    }
    # 3、模板携带schema和模型指纹，生产启动与每租户访问都可做兼容性门禁。
    return {
        "index_patterns": [f"{prefix}-*"],
        "priority": 500,
        "version": INDEX_SCHEMA_VERSION,
        "_meta": {"managed_by": "aicare-agent-service"},
        "template": {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index.refresh_interval": "1s",
            },
            "mappings": {
                "dynamic": "strict",
                "_meta": {
                    "aicare_schema_version": INDEX_SCHEMA_VERSION,
                    "embedding_fingerprint": embedding_fingerprint,
                },
                "properties": properties,
            },
        },
    }


def index_template_name(prefix: str) -> str:
    """返回受控的组合模板名称。"""
    # 1、模板名与索引前缀一一对应，避免运行服务选择任意模板。
    if not _PREFIX_PATTERN.fullmatch(prefix):
        raise ValueError("RAG_INDEX_PREFIX_INVALID")
    return f"{prefix}-template-v{INDEX_SCHEMA_VERSION}"
