"""验证 RAG 资源仅在启用时启动并按逆序可靠关闭。"""

from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from aicare_agent_service.api import lifecycle
from aicare_agent_service.api.app import create_app
from aicare_agent_service.config import Environment, Settings
from aicare_agent_service.rag import model_runtime
from aicare_agent_service.rag.embeddings import BgeM3EmbeddingProvider
from aicare_agent_service.rag.reranker import BgeReranker
from tests.fakes.rag_models import FakeEmbeddingModel, FakeRerankerModel


def test_application_lifespan_starts_rag_before_java_and_closes_it_after_java(monkeypatch) -> None:
    events: list[str] = []

    @asynccontextmanager
    async def fake_persistence(app):
        events.append("persistence-enter")
        try:
            yield
        finally:
            events.append("persistence-exit")

    @asynccontextmanager
    async def fake_rag(app):
        events.append("rag-enter")
        app.state.rag_resources = object()
        try:
            yield
        finally:
            events.append("rag-exit")

    class HttpClient:
        async def aclose(self) -> None:
            events.append("java-exit")

    monkeypatch.setattr(lifecycle, "persistence_lifespan", fake_persistence)
    monkeypatch.setattr(lifecycle, "rag_lifespan", fake_rag)
    monkeypatch.setattr(lifecycle, "build_java_http_client", lambda settings: HttpClient())
    monkeypatch.setattr(lifecycle, "JavaToolClient", lambda client, settings: object())
    settings = Settings(
        environment=Environment.TEST,
        rag_enabled=True,
        java_base_url="https://java.internal.example",
        java_service_token="secret",
        _env_file=None,
    )

    with TestClient(create_app(settings)):
        assert events == ["persistence-enter", "rag-enter"]

    assert events == [
        "persistence-enter",
        "rag-enter",
        "java-exit",
        "rag-exit",
        "persistence-exit",
    ]


def test_disabled_rag_lifespan_exposes_no_runtime_resources(monkeypatch) -> None:
    @asynccontextmanager
    async def fake_persistence(app):
        yield

    monkeypatch.setattr(lifecycle, "persistence_lifespan", fake_persistence)
    settings = Settings(environment=Environment.TEST, rag_enabled=False, _env_file=None)
    app = create_app(settings)

    with TestClient(app):
        assert app.state.rag_resources is None


@pytest.mark.asyncio
async def test_rag_resource_creation_closes_models_then_es_when_warmup_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    events: list[str] = []

    class ProbeElasticsearch:
        async def ping(self) -> bool:
            return True

        async def close(self) -> None:
            events.append("es-close")

    class FailingRuntime:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        async def warm_up(self) -> None:
            raise RuntimeError("warmup failed")

        async def close(self) -> None:
            events.append("model-close")

    monkeypatch.setattr(model_runtime, "AsyncElasticsearch", lambda **kwargs: ProbeElasticsearch())
    monkeypatch.setattr(
        model_runtime,
        "verify_model_lock",
        lambda **kwargs: {"embedding": tmp_path, "reranker": tmp_path},
    )
    monkeypatch.setattr(model_runtime, "_load_bge_models", lambda paths: (object(), object()))
    monkeypatch.setattr(model_runtime, "BgeModelRuntime", FailingRuntime)
    settings = Settings(
        environment=Environment.TEST,
        rag_enabled=True,
        elasticsearch_nodes="https://es.internal:9200",
        elasticsearch_username="agent",
        elasticsearch_password="secret",
        elasticsearch_ca_cert_path=tmp_path / "ca.crt",
        rag_model_lock_path=tmp_path / "models.lock.json",
        rag_model_dir=tmp_path,
        bge_embedding_revision="a" * 40,
        bge_reranker_revision="b" * 40,
        _env_file=None,
    )

    with pytest.raises(RuntimeError, match="warmup failed"):
        await model_runtime.create_rag_resources(settings)

    assert events == ["model-close", "es-close"]


@pytest.mark.asyncio
async def test_rag_resources_expose_only_locked_production_model_adapters(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    class ProbeElasticsearch:
        async def ping(self) -> bool:
            return True

        async def close(self) -> None:
            return None

    monkeypatch.setattr(model_runtime, "AsyncElasticsearch", lambda **kwargs: ProbeElasticsearch())
    monkeypatch.setattr(
        model_runtime,
        "verify_model_lock",
        lambda **kwargs: {"embedding": tmp_path, "reranker": tmp_path},
    )
    monkeypatch.setattr(
        model_runtime,
        "_load_bge_models",
        lambda paths: (FakeEmbeddingModel(), FakeRerankerModel()),
    )
    settings = Settings(
        environment=Environment.TEST,
        rag_enabled=True,
        elasticsearch_nodes="https://es.internal:9200",
        elasticsearch_username="agent",
        elasticsearch_password="secret",
        elasticsearch_ca_cert_path=tmp_path / "ca.crt",
        rag_model_lock_path=tmp_path / "models.lock.json",
        rag_model_dir=tmp_path,
        bge_embedding_revision="a" * 40,
        bge_reranker_revision="b" * 40,
        _env_file=None,
    )

    resources = await model_runtime.create_rag_resources(settings)
    try:
        assert isinstance(resources.embeddings, BgeM3EmbeddingProvider)
        assert isinstance(resources.reranker, BgeReranker)
    finally:
        await resources.models.close()
        await resources.elasticsearch.close()
