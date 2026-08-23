"""验证Java进程级客户端与持久化资源的启动、暴露和逆序关闭。"""

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from aicare_agent_service.api import lifecycle
from aicare_agent_service.api.app import create_app
from aicare_agent_service.config import Environment, Settings


def _settings() -> Settings:
    """返回启用Java Gateway但不连接其他外部资源的测试配置。"""
    return Settings(
        environment=Environment.TEST,
        java_base_url="https://java.internal.example",
        java_service_token="service-secret",
        _env_file=None,
    )


def test_application_lifespan_creates_one_java_client_and_closes_before_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class ProbeHttpClient:
        async def aclose(self) -> None:
            events.append("http-close")

    class ProbeJavaClient:
        def __init__(self, client: ProbeHttpClient, settings: Settings) -> None:
            assert isinstance(client, ProbeHttpClient)
            assert settings.java_service_token is not None
            events.append("java-client-create")

    @asynccontextmanager
    async def fake_persistence_lifespan(app):
        events.append("persistence-enter")
        try:
            yield
        finally:
            events.append("persistence-exit")

    def fake_build_http_client(settings: Settings) -> ProbeHttpClient:
        assert settings.java_base_url is not None
        events.append("http-create")
        return ProbeHttpClient()

    monkeypatch.setattr(lifecycle, "persistence_lifespan", fake_persistence_lifespan)
    monkeypatch.setattr(lifecycle, "build_java_http_client", fake_build_http_client)
    monkeypatch.setattr(lifecycle, "JavaToolClient", ProbeJavaClient)
    app = create_app(_settings())

    with TestClient(app):
        assert app.state.java_http_client is not None
        assert app.state.java_client is not None
        assert events == ["persistence-enter", "http-create", "java-client-create"]

    assert events == [
        "persistence-enter",
        "http-create",
        "java-client-create",
        "http-close",
        "persistence-exit",
    ]


def test_application_lifespan_closes_http_client_when_adapter_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class ProbeHttpClient:
        async def aclose(self) -> None:
            events.append("http-close")

    @asynccontextmanager
    async def fake_persistence_lifespan(app):
        events.append("persistence-enter")
        try:
            yield
        finally:
            events.append("persistence-exit")

    def fail_adapter(*args: object, **kwargs: object) -> object:
        raise RuntimeError("adapter failure")

    monkeypatch.setattr(lifecycle, "persistence_lifespan", fake_persistence_lifespan)
    monkeypatch.setattr(lifecycle, "build_java_http_client", lambda settings: ProbeHttpClient())
    monkeypatch.setattr(lifecycle, "JavaToolClient", fail_adapter)
    app = create_app(_settings())

    with pytest.raises(RuntimeError, match="adapter failure"), TestClient(app):
        pass

    assert events == ["persistence-enter", "http-close", "persistence-exit"]
