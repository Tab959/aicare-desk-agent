"""使用真实DeepSeek、真实Java工具和LangSmith验证Task 6工具调用闭环。"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import langsmith as ls
import pytest
import pytest_asyncio
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client

from aicare_agent_service.config import Settings
from aicare_agent_service.graph.context import AgentRuntimeContext
from aicare_agent_service.graph.state import AgentIdentity
from aicare_agent_service.models.contracts import ModelPurpose
from aicare_agent_service.models.deepseek import DeepSeekModelProvider
from aicare_agent_service.tools.contracts import ToolName
from aicare_agent_service.tools.java_client import JavaToolClient, build_java_http_client
from aicare_agent_service.tools.registry import READ_ONLY_TOOL_REGISTRY

LIVE_FLAG = "AICARE_RUN_LIVE_TOOL_TESTS"
RUN_NAME = "task6-live-deepseek-java-tool"
TRACE_TAGS = {"task6-live", "java-e2e", "read-only-tool"}


def _live_test_enabled() -> bool:
    return all(
        (
            os.getenv(LIVE_FLAG) == "1",
            bool(os.getenv("DEEPSEEK_API_KEY", "").strip()),
            bool(os.getenv("LANGSMITH_API_KEY", "").strip()),
            os.getenv("AICARE_AGENT_LIVE_JAVA_SUCCESS") == "true",
        )
    )


pytestmark = pytest.mark.skipif(
    not _live_test_enabled(),
    reason="requires explicit live DeepSeek, LangSmith, Java and active-run fixture",
)


@dataclass(frozen=True, slots=True)
class LiveToolExecution:
    """保存一次真实工具Agent执行结果与LangSmith根Trace证据。"""

    settings: Settings
    messages: list[Any]
    root_run: Any
    trace_runs: list[Any]
    trace_url: str


@pytest_asyncio.fixture(scope="module")
async def live_tool_execution() -> LiveToolExecution:
    settings = Settings(
        _env_file=None,
        java_base_url="http://localhost:8080",
        java_allow_private_http=True,
    )
    assert settings.deepseek_model == "deepseek-v4-pro"
    assert settings.langsmith_api_key is not None
    assert settings.java_service_token is not None

    langsmith_client = Client(api_key=settings.langsmith_api_key.get_secret_value())
    projects = list(langsmith_client.list_projects(name=settings.langsmith_project, limit=1))
    assert len(projects) == 1
    project_id = str(projects[0].id)
    provider = DeepSeekModelProvider(settings)
    model = provider.create(ModelPurpose.SPECIALIST)
    search_tool = READ_ONLY_TOOL_REGISTRY[ToolName.SEARCH_GAMES].tool
    agent = create_agent(
        model,
        tools=(search_tool,),
        system_prompt=(
            "你是Steam游戏售前查询验收Agent。用户要求查询商品时必须先调用search_games，"
            "只能依据工具结果回答；不得编造价格、库存或商品。"
        ),
        context_schema=AgentRuntimeContext,
        name="task6_tool_agent",
    )
    identity = AgentIdentity(
        tenant_id="tenant-demo",
        customer_id="user-customer-001",
        conversation_id="conv-tool-live-001",
        run_id="run-tool-live-001",
        trigger_message_id="msg-tool-live-001",
        trigger_sequence=1,
    )
    http_client = build_java_http_client(settings)
    java_client = JavaToolClient(http_client, settings)
    context = AgentRuntimeContext(
        expected_identity=identity,
        java_client=java_client,
        model_provider=provider,
        request_deadline=datetime.now(UTC) + timedelta(seconds=60),
    )
    started_at = datetime.now(UTC) - timedelta(seconds=2)

    try:
        # 1、真实DeepSeek只能看到search_games业务Schema，完整身份和Java客户端由Runtime注入。
        with ls.tracing_context(
            client=langsmith_client,
            project_name=settings.langsmith_project,
            enabled=True,
        ):
            result = await agent.ainvoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "请先调用商品搜索工具，查找100元以内当前可购买的游戏，再简短说明结果。",
                        }
                    ]
                },
                context=context,
                config={
                    "run_name": RUN_NAME,
                    "tags": sorted(TRACE_TAGS),
                    "metadata": {
                        "task": "6g",
                        "model": settings.deepseek_model,
                        "tool": ToolName.SEARCH_GAMES.value,
                        "data_classification": "synthetic",
                    },
                },
            )
    finally:
        await http_client.aclose()

    # 2、等待异步Tracer写入后，读取根run及同一trace的全部模型、工具和Agent节点。
    wait_for_all_tracers()
    root_run = await _wait_for_root_run(langsmith_client, project_id, started_at)
    trace_runs = [
        run
        async for run in langsmith_client.runs.query(
            project_ids=[project_id],
            trace_id=str(root_run.trace_id),
            selects=[
                "ID",
                "NAME",
                "RUN_TYPE",
                "START_TIME",
                "END_TIME",
                "INPUTS",
                "OUTPUTS",
                "ERROR",
                "EXTRA",
                "METADATA",
                "EVENTS",
                "TAGS",
                "TRACE_ID",
            ],
        )
    ]
    trace_url = await langsmith_client.runs.get_url(
        str(root_run.id),
        project_id=project_id,
        trace_id=str(root_run.trace_id),
    )
    return LiveToolExecution(
        settings=settings,
        messages=list(result["messages"]),
        root_run=root_run,
        trace_runs=trace_runs,
        trace_url=trace_url.url,
    )


def test_deepseek_generates_only_business_arguments_and_runtime_executes_java_tool(
    live_tool_execution: LiveToolExecution,
) -> None:
    tool_calls = [
        call
        for message in live_tool_execution.messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    ]
    assert tool_calls
    assert tool_calls[0]["name"] == ToolName.SEARCH_GAMES.value
    assert set(tool_calls[0]["args"]) <= {
        "query",
        "max_price",
        "purchase_methods",
        "limit",
        "cursor",
    }
    assert not {
        "tenant_id",
        "customer_id",
        "conversation_id",
        "run_id",
        "trigger_message_id",
        "service_token",
        "url",
    }.intersection(tool_calls[0]["args"])

    tool_messages = [
        message for message in live_tool_execution.messages if isinstance(message, ToolMessage)
    ]
    assert tool_messages
    assert json.loads(tool_messages[0].content)["kind"] == "GAME_PAGE"
    assert tool_messages[0].artifact.tool_name == ToolName.SEARCH_GAMES.value
    assert tool_messages[0].artifact.status.value == "SUCCESS"
    assert str(live_tool_execution.messages[-1].content).strip()


def test_langsmith_trace_contains_model_tool_latency_and_no_internal_secrets(
    live_tool_execution: LiveToolExecution,
) -> None:
    root = live_tool_execution.root_run
    assert root.name == RUN_NAME
    assert TRACE_TAGS.issubset(set(root.tags or []))
    assert (root.metadata or {}).get("task") == "6g"
    assert (root.metadata or {}).get("model") == "deepseek-v4-pro"
    assert root.end_time is not None

    run_names = {run.name for run in live_tool_execution.trace_runs}
    run_types = {run.run_type for run in live_tool_execution.trace_runs}
    assert ToolName.SEARCH_GAMES.value in run_names
    assert "llm" in run_types
    assert "tool" in run_types
    assert all(run.end_time is not None for run in live_tool_execution.trace_runs)

    trace_payload = json.dumps(
        [
            {
                "name": run.name,
                "run_type": run.run_type,
                "inputs": run.inputs,
                "outputs": run.outputs,
                "error": run.error,
                "extra": run.extra,
                "events": run.events,
                "tags": run.tags,
            }
            for run in live_tool_execution.trace_runs
        ],
        ensure_ascii=False,
        default=str,
    )
    _assert_internal_values_absent(trace_payload, live_tool_execution.settings)
    print(f"Task 6 LangSmith Trace ID: {root.trace_id}")
    print(f"Task 6 LangSmith Trace URL: {live_tool_execution.trace_url}")


async def _wait_for_root_run(client: Client, project_id: str, started_at: datetime) -> Any:
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        runs = [
            run
            async for run in client.runs.query(
                project_ids=[project_id],
                is_root=True,
                min_start_time=started_at,
                selects=[
                    "ID",
                    "NAME",
                    "START_TIME",
                    "END_TIME",
                    "INPUTS",
                    "OUTPUTS",
                    "ERROR",
                    "EXTRA",
                    "METADATA",
                    "TAGS",
                    "TRACE_ID",
                ],
                page_size=20,
            )
        ]
        matching = [run for run in runs if run.name == RUN_NAME and run.end_time is not None]
        if matching:
            return matching[0]
        await asyncio.sleep(1)
    raise AssertionError("LangSmith未在45秒内返回Task 6工具调用根Trace")


def _assert_internal_values_absent(payload: str, settings: Settings) -> None:
    forbidden_values = [
        settings.deepseek_api_key.get_secret_value() if settings.deepseek_api_key else "",
        settings.langsmith_api_key.get_secret_value() if settings.langsmith_api_key else "",
        settings.java_service_token.get_secret_value() if settings.java_service_token else "",
        "tenant-demo",
        "user-customer-001",
        "conv-tool-live-001",
        "run-tool-live-001",
        "msg-tool-live-001",
        "CDK-DEMO-SENSITIVE-CANARY",
    ]
    if any(value and value in payload for value in forbidden_values):
        raise AssertionError("LangSmith Trace包含凭据、完整运行身份或敏感canary")
    assert "contractVersion" not in payload
    assert "X-AICare-Agent-Service-Token" not in payload
    assert "raw_response" not in payload.lower()
    assert "stacktrace" not in payload.lower()
