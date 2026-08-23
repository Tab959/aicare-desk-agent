"""验证生产 RAG 配置完整性及 Elasticsearch 传输安全门禁。"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from aicare_agent_service.config import Environment, Settings, validate_production_settings


def _production_settings(tmp_path: Path, **overrides: object) -> Settings:
    """构造仅缺少测试指定字段的生产配置。"""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    lock_path = tmp_path / "models.lock.json"
    lock_path.write_text('{"schemaVersion": 1, "models": []}', encoding="utf-8")
    ca_path = tmp_path / "http_ca.crt"
    ca_path.write_text("test-ca", encoding="utf-8")
    values: dict[str, object] = {
        "environment": Environment.PRODUCTION,
        "deepseek_api_key": "deepseek-secret",
        "agent_postgres_dsn": "postgresql://agent:password@postgres:5432/aicare_agent",
        "checkpoint_backend": "postgres",
        "agent_redis_url": "redis://agent:password@redis:6379/0",
        "redis_mode": "sentinel",
        "redis_sentinels": "s1:26379,s2:26379,s3:26379",
        "redis_master_name": "agent-master",
        "redis_sentinel_username": "sentinel-user",
        "redis_sentinel_password": "sentinel-secret",
        "checkpoint_encryption_key": "0123456789abcdef0123456789abcdef",
        "java_base_url": "http://platform-api:8080",
        "java_service_token": "java-secret",
        "java_allow_private_http": True,
        "rag_enabled": True,
        "elasticsearch_nodes": "https://es-a:9200,https://es-b:9200",
        "elasticsearch_username": "aicare_agent",
        "elasticsearch_password": "es-secret",
        "elasticsearch_ca_cert_path": ca_path,
        "rag_model_lock_path": lock_path,
        "rag_model_dir": model_dir,
        "bge_embedding_revision": "a" * 40,
        "bge_reranker_revision": "b" * 40,
        "rag_chunk_hmac_key": "c" * 64,
    }
    values.update(overrides)
    return Settings(**values, _env_file=None)


@pytest.mark.parametrize(
    ("field_name", "expected_name"),
    [
        ("elasticsearch_nodes", "AICARE_AGENT_ELASTICSEARCH_NODES"),
        ("elasticsearch_username", "AICARE_AGENT_ELASTICSEARCH_USERNAME"),
        ("elasticsearch_password", "AICARE_AGENT_ELASTICSEARCH_PASSWORD"),
        ("elasticsearch_ca_cert_path", "AICARE_AGENT_ELASTICSEARCH_CA_CERT_PATH"),
        ("rag_model_lock_path", "AICARE_AGENT_RAG_MODEL_LOCK_PATH"),
        ("rag_model_dir", "AICARE_AGENT_RAG_MODEL_DIR"),
        ("bge_embedding_revision", "AICARE_AGENT_BGE_EMBEDDING_REVISION"),
        ("bge_reranker_revision", "AICARE_AGENT_BGE_RERANKER_REVISION"),
        ("rag_chunk_hmac_key", "AICARE_AGENT_RAG_CHUNK_HMAC_KEY"),
    ],
)
def test_production_requires_each_rag_dependency(
    tmp_path: Path, field_name: str, expected_name: str
) -> None:
    settings = _production_settings(tmp_path, **{field_name: None})

    with pytest.raises(ValueError, match=expected_name):
        validate_production_settings(settings)


def test_production_rejects_missing_model_lock_file(tmp_path: Path) -> None:
    settings = _production_settings(tmp_path, rag_model_lock_path=tmp_path / "missing.json")

    with pytest.raises(ValueError, match="模型锁文件不存在"):
        validate_production_settings(settings)


def test_production_rejects_missing_model_directory(tmp_path: Path) -> None:
    settings = _production_settings(tmp_path, rag_model_dir=tmp_path / "missing-models")

    with pytest.raises(ValueError, match="模型目录不存在"):
        validate_production_settings(settings)


@pytest.mark.parametrize(
    "nodes",
    ["http://search.example.com:9200", "https://user:secret@search.example.com:9200"],
)
def test_elasticsearch_nodes_reject_insecure_public_or_embedded_credentials(nodes: str) -> None:
    with pytest.raises(ValidationError):
        Settings(elasticsearch_nodes=nodes, _env_file=None)


def test_elasticsearch_nodes_support_multiple_tls_endpoints() -> None:
    settings = Settings(
        elasticsearch_nodes="https://es-a.internal:9200,https://es-b.internal:9200",
        _env_file=None,
    )

    assert tuple(str(node) for node in settings.elasticsearch_node_urls) == (
        "https://es-a.internal:9200/",
        "https://es-b.internal:9200/",
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("elasticsearch_verify_certs", False),
        ("rag_allow_runtime_model_download", True),
        ("rag_embedding_batch_size", 0),
        ("rag_reranker_batch_size", 0),
        ("rag_model_max_concurrency", 0),
        ("rag_model_deadline_seconds", 0),
    ],
)
def test_rag_security_and_budget_guards_reject_unsafe_values(
    field_name: str, invalid_value: object
) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field_name: invalid_value}, _env_file=None)


def test_safe_rag_boolean_switches_parse_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AICARE_AGENT_ELASTICSEARCH_VERIFY_CERTS", "true")
    monkeypatch.setenv("AICARE_AGENT_RAG_MODEL_USE_FP16", "false")
    monkeypatch.setenv("AICARE_AGENT_RAG_ALLOW_RUNTIME_MODEL_DOWNLOAD", "false")

    settings = Settings(_env_file=None)

    assert settings.elasticsearch_verify_certs is True
    assert settings.rag_model_use_fp16 is False
    assert settings.rag_allow_runtime_model_download is False
