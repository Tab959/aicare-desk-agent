import importlib
from contextlib import asynccontextmanager

import pytest
from fastapi.testclient import TestClient

from aicare_agent_service.api import lifecycle
from aicare_agent_service.api.app import create_app
from aicare_agent_service.config import Environment, Settings, get_settings
from aicare_agent_service.rag.readiness import RagReadinessReport


def test_create_app_keeps_legacy_health_routes(test_settings: Settings) -> None:
    client = TestClient(create_app(test_settings))
    expected_response = {
        "status": "UP",
        "service": "aicare-agent-service",
        "version": "0.1.0",
    }

    assert client.get("/health").json() == expected_response
    assert client.get("/api/v1/agent/health").json() == expected_response


def test_readiness_reports_configuration_without_calling_dependencies(
    test_settings: Settings,
) -> None:
    client = TestClient(create_app(test_settings))

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "UP",
        "service": "aicare-agent-service",
        "version": "0.1.0",
        "checks": {
            "configuration": "UP",
            "elasticsearch": "DISABLED",
            "rag_models": "DISABLED",
            "elasticsearch_cluster": "DISABLED",
            "index_template": "DISABLED",
            "aliases_mapping": "DISABLED",
        },
    }


@pytest.mark.parametrize(
    ("ping_result", "status_code", "status"), [(True, 200, "UP"), (False, 503, "DOWN")]
)
def test_readiness_reflects_live_elasticsearch_ping(
    monkeypatch: pytest.MonkeyPatch,
    ping_result: bool,
    status_code: int,
    status: str,
) -> None:
    class ProbeReadiness:
        async def check(self) -> RagReadinessReport:
            value = "UP" if ping_result else "DOWN"
            return RagReadinessReport(
                models=value,
                elasticsearch_cluster=value,
                index_template=value,
                aliases_mapping=value,
            )

    class ProbeResources:
        readiness = ProbeReadiness()

    @asynccontextmanager
    async def fake_rag_lifespan(app):
        app.state.rag_resources = ProbeResources()
        yield

    monkeypatch.setattr(lifecycle, "rag_lifespan", fake_rag_lifespan)
    settings = Settings(environment=Environment.TEST, rag_enabled=True, _env_file=None)

    with TestClient(create_app(settings)) as client:
        response = client.get("/health/ready")

    assert response.status_code == status_code
    assert response.json()["status"] == status
    assert response.json()["checks"]["elasticsearch"] == status


def test_injected_settings_control_application_metadata_and_health() -> None:
    settings = Settings(
        service_name="custom-agent-service",
        service_version="9.8.7",
        _env_file=None,
    )

    app = create_app(settings)
    response = TestClient(app).get("/health")

    assert app.state.settings is settings
    assert app.title == "custom-agent-service"
    assert app.version == "9.8.7"
    assert response.json()["service"] == "custom-agent-service"
    assert response.json()["version"] == "9.8.7"


def test_create_app_rejects_incomplete_production_settings() -> None:
    settings = Settings(environment=Environment.PRODUCTION, _env_file=None)

    with pytest.raises(ValueError, match="生产环境缺少必需配置"):
        create_app(settings)


def test_main_exports_compatible_asgi_application(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AICARE_AGENT_ENVIRONMENT", "test")
    get_settings.cache_clear()

    from aicare_agent_service import main as main_module

    reloaded_main = importlib.reload(main_module)
    response = TestClient(reloaded_main.app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "UP"
