"""锁定 Elasticsearch 知识索引 Mapping、租户命名空间和别名契约。"""

from __future__ import annotations

import re

from aicare_agent_service.rag.elasticsearch_mapping import (
    EMBEDDING_DIMENSIONS,
    INDEX_SCHEMA_VERSION,
    build_index_template,
    build_tenant_index_names,
    derive_tenant_namespace_key,
)

FINGERPRINT = "a" * 64
HMAC_KEY = b"tenant-namespace-test-key-with-32-bytes"


def test_mapping_is_strict_smartcn_and_uses_int8_hnsw() -> None:
    template = build_index_template(prefix="aicare-kb", embedding_fingerprint=FINGERPRINT)
    mappings = template["template"]["mappings"]
    properties = mappings["properties"]

    assert mappings["dynamic"] == "strict"
    assert mappings["_meta"] == {
        "aicare_schema_version": INDEX_SCHEMA_VERSION,
        "embedding_fingerprint": FINGERPRINT,
    }
    assert properties["content"] == {
        "type": "text",
        "analyzer": "smartcn",
        "fields": {"standard": {"type": "text", "analyzer": "standard"}},
    }
    assert properties["embedding"] == {
        "type": "dense_vector",
        "dims": EMBEDDING_DIMENSIONS,
        "index": True,
        "similarity": "cosine",
        "index_options": {"type": "int8_hnsw"},
    }
    assert template["index_patterns"] == ["aicare-kb-*"]


def test_tenant_names_are_hmac_scoped_and_do_not_disclose_tenant() -> None:
    tenant = "tenant-super-secret-canary"
    names = build_tenant_index_names(
        prefix="aicare-kb",
        tenant_id=tenant,
        hmac_key=HMAC_KEY,
        embedding_fingerprint=FINGERPRINT,
        generation=7,
    )

    rendered = repr(names)
    assert tenant not in rendered
    assert names.read_alias.endswith("-read")
    assert names.write_alias.endswith("-write")
    assert names.physical_index.endswith("-000007")
    assert names.routing == names.tenant_namespace
    assert re.fullmatch(r"[a-f0-9]{32}", names.tenant_namespace)


def test_wrong_fingerprint_or_generation_is_rejected() -> None:
    for fingerprint in ("short", "g" * 64):
        try:
            build_tenant_index_names(
                prefix="aicare-kb",
                tenant_id="tenant-a",
                hmac_key=HMAC_KEY,
                embedding_fingerprint=fingerprint,
                generation=1,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid fingerprint must be rejected")

    try:
        build_tenant_index_names(
            prefix="aicare-kb",
            tenant_id="tenant-a",
            hmac_key=HMAC_KEY,
            embedding_fingerprint=FINGERPRINT,
            generation=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-positive generation must be rejected")


def test_tenant_namespace_uses_a_purpose_derived_subkey() -> None:
    derived = derive_tenant_namespace_key(HMAC_KEY)

    assert len(derived) == 32
    assert derived != HMAC_KEY
    assert derived == derive_tenant_namespace_key(HMAC_KEY)
