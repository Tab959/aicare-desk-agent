from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from aicare_agent_service.api import lifecycle as api_lifecycle
from aicare_agent_service.api.app import create_app
from aicare_agent_service.config import CheckpointBackend, Environment, RedisMode, Settings
from aicare_agent_service.persistence import lifecycle
from aicare_agent_service.persistence.redis_readiness import (
    RedisReadinessError,
    RedisReadinessReport,
)


def production_settings(tmp_path: Path) -> Settings:
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    lock_path = tmp_path / "models.lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    ca_path = tmp_path / "http_ca.crt"
    ca_path.write_text("ca", encoding="utf-8")
    return Settings(
        environment=Environment.PRODUCTION,
        deepseek_api_key="deepseek-secret",
        agent_postgres_dsn="postgresql://agent:password@postgres/aicare_agent",
        checkpoint_backend=CheckpointBackend.POSTGRES,
        agent_redis_url="redis://:password@redis/1",
        redis_mode=RedisMode.SENTINEL,
        redis_sentinels="sentinel-a:26379,sentinel-b:26380,sentinel-c:26381",
        redis_master_name="aicare-agent-master",
        redis_sentinel_username="aicare_sentinel",
        redis_sentinel_password="sentinel-secret",
        checkpoint_encryption_key="0123456789abcdef0123456789abcdef",
        java_base_url="https://platform-api.example.com",
        java_service_token="java-secret",
        rag_enabled=True,
        elasticsearch_nodes="https://elasticsearch:9200",
        elasticsearch_username="aicare_agent",
        elasticsearch_password="es-secret",
        elasticsearch_ca_cert_path=ca_path,
        rag_model_lock_path=lock_path,
        rag_model_dir=model_dir,
        bge_embedding_revision="a" * 40,
        bge_reranker_revision="b" * 40,
        rag_chunk_hmac_key="c" * 64,
        _env_file=None,
    )


@pytest.fixture(autouse=True)
def isolate_persistence_lifecycle_from_rag(monkeypatch: pytest.MonkeyPatch):
    """本文件只验证持久化顺序，不启动独立RAG外部资源。"""

    @asynccontextmanager
    async def no_op_rag_lifespan(app):
        app.state.rag_resources = None
        yield

    monkeypatch.setattr(api_lifecycle, "rag_lifespan", no_op_rag_lifespan)


def test_default_lifespan_starts_without_external_dependencies() -> None:
    app = create_app(Settings(environment=Environment.TEST, _env_file=None))

    with TestClient(app) as client:
        assert client.get("/health/ready").status_code == 200
        assert app.state.checkpointer is not None
        assert app.state.redis_client is None
        assert app.state.run_store is None
        assert app.state.run_lifecycle is None


def test_lifespan_never_runs_checkpointer_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class ProbeSaver:
        async def setup(self) -> None:
            pytest.fail("普通FastAPI启动不得调用setup")

    @asynccontextmanager
    async def fake_checkpointer_resource(settings: Settings):
        del settings
        events.append("enter")
        yield ProbeSaver()
        events.append("exit")

    monkeypatch.setattr(lifecycle, "checkpointer_resource", fake_checkpointer_resource)
    app = create_app(Settings(environment=Environment.TEST, _env_file=None))

    with TestClient(app):
        assert events == ["enter"]
    assert events == ["enter", "exit"]


def test_production_lifespan_rejects_failed_redis_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []

    class ProbeRedis:
        async def aclose(self) -> None:
            events.append("redis-close")

    def fake_build_client(*args: object, **kwargs: object) -> ProbeRedis:
        return ProbeRedis()

    async def fail_audit(*args: object, **kwargs: object) -> RedisReadinessReport:
        del args, kwargs
        raise RedisReadinessError(RedisReadinessReport(()))

    @asynccontextmanager
    async def fake_checkpointer_resource(settings: Settings):
        del settings
        events.append("checkpointer-enter")
        try:
            yield object()
        finally:
            events.append("checkpointer-exit")

    monkeypatch.setattr(lifecycle, "build_redis_client", fake_build_client)
    monkeypatch.setattr(lifecycle, "enforce_redis_readiness", fail_audit)
    monkeypatch.setattr(lifecycle, "checkpointer_resource", fake_checkpointer_resource)
    app = create_app(production_settings(tmp_path))

    with pytest.raises(RedisReadinessError, match="Redis生产环境门禁未通过"), TestClient(app):
        pass

    assert events == ["checkpointer-enter", "redis-close", "checkpointer-exit"]


def test_lifespan_closes_resources_in_reverse_order_on_normal_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []

    class ProbeSaver:
        async def setup(self) -> None:
            pytest.fail("普通FastAPI启动不得调用setup")

    class ProbeRedis:
        async def aclose(self) -> None:
            events.append("redis-close")

    class ProbeRunStore:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            events.append("run-store-create")

    class ProbeLifecycle:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            events.append("run-lifecycle-create")

    @asynccontextmanager
    async def fake_checkpointer_resource(settings: Settings):
        del settings
        events.append("checkpointer-enter")
        try:
            yield ProbeSaver()
        finally:
            events.append("checkpointer-exit")

    def fake_build_client(*args: object, **kwargs: object) -> ProbeRedis:
        del args, kwargs
        events.append("redis-build")
        return ProbeRedis()

    async def passing_audit(*args: object, **kwargs: object) -> RedisReadinessReport:
        del args, kwargs
        events.append("redis-audit")
        return RedisReadinessReport(())

    monkeypatch.setattr(lifecycle, "checkpointer_resource", fake_checkpointer_resource)
    monkeypatch.setattr(lifecycle, "build_redis_client", fake_build_client)
    monkeypatch.setattr(lifecycle, "enforce_redis_readiness", passing_audit)
    monkeypatch.setattr(lifecycle, "RedisRunStore", ProbeRunStore)
    monkeypatch.setattr(lifecycle, "AgentRunLifecycle", ProbeLifecycle)
    app = create_app(production_settings(tmp_path))

    with TestClient(app):
        assert events == [
            "checkpointer-enter",
            "redis-build",
            "redis-audit",
            "run-store-create",
            "run-lifecycle-create",
        ]

    assert events == [
        "checkpointer-enter",
        "redis-build",
        "redis-audit",
        "run-store-create",
        "run-lifecycle-create",
        "redis-close",
        "checkpointer-exit",
    ]


def test_lifespan_passes_sentinel_discovery_and_acl_to_client_factory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class ProbeRedis:
        async def aclose(self) -> None:
            return None

    class ProbeRunStore:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    class ProbeLifecycle:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    @asynccontextmanager
    async def fake_checkpointer_resource(settings: Settings):
        del settings
        yield object()

    def fake_build_client(*args: object, **kwargs: object) -> ProbeRedis:
        captured.update(kwargs)
        return ProbeRedis()

    async def passing_audit(*args: object, **kwargs: object) -> RedisReadinessReport:
        del args, kwargs
        return RedisReadinessReport(())

    settings = production_settings(tmp_path).model_copy(
        update={
            "redis_mode": RedisMode.SENTINEL,
            "redis_sentinels": "sentinel-a:26379,sentinel-b:26380,sentinel-c:26381",
            "redis_master_name": "aicare-agent-master",
            "redis_sentinel_username": "aicare_sentinel",
            "redis_sentinel_password": SecretStr("sentinel-secret"),
        }
    )
    monkeypatch.setattr(lifecycle, "checkpointer_resource", fake_checkpointer_resource)
    monkeypatch.setattr(lifecycle, "build_redis_client", fake_build_client)
    monkeypatch.setattr(lifecycle, "enforce_redis_readiness", passing_audit)
    monkeypatch.setattr(lifecycle, "RedisRunStore", ProbeRunStore)
    monkeypatch.setattr(lifecycle, "AgentRunLifecycle", ProbeLifecycle)
    app = create_app(settings)

    with TestClient(app):
        pass

    assert captured["redis_mode"] is RedisMode.SENTINEL
    assert captured["sentinel_endpoints"] == (
        ("sentinel-a", 26379),
        ("sentinel-b", 26380),
        ("sentinel-c", 26381),
    )
    assert captured["sentinel_master_name"] == "aicare-agent-master"
    assert captured["sentinel_username"] == "aicare_sentinel"
