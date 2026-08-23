import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import httpx
import langsmith as ls
import pytest
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client
from pydantic import BaseModel, Field

from aicare_agent_service.config import ModelProviderName, Settings
from aicare_agent_service.models.contracts import ModelPurpose
from aicare_agent_service.models.deepseek import DeepSeekModelProvider

LIVE_FLAG = "AICARE_RUN_LIVE_MODEL_TESTS"
LIVE_RUN_NAMES = {
    "task2-live-structured-output",
    "task2-live-streaming",
}


def _live_tests_enabled() -> bool:
    required_values = (
        os.getenv(LIVE_FLAG) == "1",
        bool(os.getenv("DEEPSEEK_API_KEY", "").strip()),
        bool(os.getenv("LANGSMITH_API_KEY", "").strip()),
    )
    return all(required_values)


pytestmark = pytest.mark.skipif(
    not _live_tests_enabled(),
    reason="真实模型测试需要显式开关以及DeepSeek、LangSmith凭据",
)


class RouteProbe(BaseModel):
    """Task 2真实模型探针，不作为Task 5正式路由契约。"""

    intent: Literal["pre_sales", "after_sales"] = Field(description="客服问题意图")
    confidence: float = Field(ge=0, le=1, description="分类置信度")


@dataclass(frozen=True, slots=True)
class LivePreflight:
    settings: Settings
    langsmith_client: Client
    model_id: str
    project_id: str


@dataclass(frozen=True, slots=True)
class LiveExecution:
    structured_result: RouteProbe
    streamed_text: str
    started_at: datetime


@pytest.fixture(scope="module")
def live_preflight() -> LivePreflight:
    settings = Settings(_env_file=None)
    assert settings.model_provider is ModelProviderName.DEEPSEEK
    assert settings.deepseek_api_key is not None
    assert settings.langsmith_api_key is not None

    models_url = f"{str(settings.deepseek_base_url).rstrip('/')}/models"
    response = httpx.get(
        models_url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {settings.deepseek_api_key.get_secret_value()}",
        },
        timeout=settings.model_timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    available_models = {
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    assert settings.deepseek_model in available_models

    langsmith_client = Client(api_key=settings.langsmith_api_key.get_secret_value())
    projects = list(langsmith_client.list_projects(name=settings.langsmith_project, limit=1))
    assert len(projects) == 1

    return LivePreflight(
        settings=settings,
        langsmith_client=langsmith_client,
        model_id=settings.deepseek_model,
        project_id=str(projects[0].id),
    )


@pytest.fixture(scope="module")
def live_execution(live_preflight: LivePreflight) -> LiveExecution:
    settings = live_preflight.settings
    provider = DeepSeekModelProvider(settings)
    started_at = datetime.now(UTC) - timedelta(seconds=2)
    trace_config = {
        "tags": ["task2-live"],
        "metadata": {
            "task": "2d",
            "data_classification": "synthetic",
        },
    }

    structured_model = provider.create(ModelPurpose.ROUTING).with_structured_output(
        RouteProbe,
        method="function_calling",
    )
    with ls.tracing_context(
        client=live_preflight.langsmith_client,
        project_name=settings.langsmith_project,
        enabled=True,
    ):
        structured_result = structured_model.invoke(
            "虚构测试场景：用户想了解一款游戏当前是否有折扣。请判断为售前或售后问题。",
            config={
                **trace_config,
                "run_name": "task2-live-structured-output",
            },
        )

        answer_model = provider.create(ModelPurpose.ANSWER)
        streamed_text = "".join(
            chunk.text
            for chunk in answer_model.stream(
                "这是虚构连通性测试，请用一句简短中文回复测试已收到。",
                config={
                    **trace_config,
                    "run_name": "task2-live-streaming",
                },
            )
        )

    wait_for_all_tracers()
    assert isinstance(structured_result, RouteProbe)
    return LiveExecution(
        structured_result=structured_result,
        streamed_text=streamed_text,
        started_at=started_at,
    )


def test_live_read_only_preflight_confirms_external_access(
    live_preflight: LivePreflight,
) -> None:
    print(f"DeepSeek预检通过：model={live_preflight.model_id}")
    print("LangSmith预检通过：project_read_limit=1")


def test_deepseek_live_supports_pydantic_structured_output(
    live_execution: LiveExecution,
) -> None:
    assert live_execution.structured_result.intent == "pre_sales"
    assert 0 <= live_execution.structured_result.confidence <= 1


def test_deepseek_live_streams_non_empty_answer(live_execution: LiveExecution) -> None:
    assert live_execution.streamed_text.strip()


@pytest.mark.asyncio
async def test_langsmith_receives_sanitized_task2_traces(
    live_preflight: LivePreflight,
    live_execution: LiveExecution,
) -> None:
    runs = await _wait_for_live_runs(
        live_preflight.langsmith_client,
        live_preflight.project_id,
        live_execution.started_at,
    )
    named_runs = {run.name: run for run in runs if run.name in LIVE_RUN_NAMES}
    assert set(named_runs) == LIVE_RUN_NAMES

    for run in named_runs.values():
        assert "task2-live" in (run.tags or [])
        metadata = run.metadata or {}
        assert metadata.get("task") == "2d"
        assert metadata.get("data_classification") == "synthetic"

    trace_payload = json.dumps(
        [
            {
                "name": run.name,
                "inputs": run.inputs,
                "outputs": run.outputs,
                "error": run.error,
                "extra": run.extra,
                "events": run.events,
            }
            for run in runs
        ],
        ensure_ascii=False,
        default=str,
    )
    _assert_secrets_absent(trace_payload, live_preflight.settings)


async def _wait_for_live_runs(
    client: Client,
    project_id: str,
    started_at: datetime,
) -> list[Any]:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        runs = [
            run
            async for run in client.runs.query(
                project_ids=[project_id],
                is_root=True,
                min_start_time=started_at,
                page_size=10,
                selects=[
                    "NAME",
                    "INPUTS",
                    "OUTPUTS",
                    "ERROR",
                    "EXTRA",
                    "METADATA",
                    "EVENTS",
                    "TAGS",
                ],
            )
        ]
        if LIVE_RUN_NAMES.issubset({run.name for run in runs}):
            return runs
        await asyncio.sleep(1)
    raise AssertionError("LangSmith未在30秒内返回Task 2 Live Trace")


def _assert_secrets_absent(payload: str, settings: Settings) -> None:
    secret_values = [
        secret.get_secret_value()
        for secret in (
            settings.deepseek_api_key,
            settings.langsmith_api_key,
            settings.java_service_token,
        )
        if secret is not None and secret.get_secret_value()
    ]
    if any(secret in payload for secret in secret_values):
        raise AssertionError("LangSmith Trace包含敏感凭据")
