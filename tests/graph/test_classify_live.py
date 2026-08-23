"""使用真实DeepSeek执行Task 5结构化分类，并回查LangSmith脱敏追踪。"""

import asyncio
import json
import os
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import langsmith as ls
import pytest
from langchain_core.tracers.langchain import wait_for_all_tracers
from langgraph.runtime import Runtime
from langsmith import Client

from aicare_agent_service.config import ModelProviderName, Settings
from aicare_agent_service.contracts.adapters import adapt_run_request
from aicare_agent_service.contracts.decisions import Intent
from aicare_agent_service.models.deepseek import DeepSeekModelProvider
from aicare_agent_service.nodes.classify import classify_node

LIVE_FLAG = "AICARE_RUN_LIVE_MODEL_TESTS"
LIVE_RUN_NAME = "root.route.classify"
PASSWORD_CANARY = "task5-live-password-canary-7f41"


def _live_tests_enabled() -> bool:
    """仅在显式开启且两类外部凭据都已注入时运行真实验收。"""
    # 1、默认跳过会产生费用的网络测试，避免普通pytest意外调用外部服务。
    required_values = (
        os.getenv(LIVE_FLAG) == "1",
        bool(os.getenv("DEEPSEEK_API_KEY", "").strip()),
        bool(os.getenv("LANGSMITH_API_KEY", "").strip()),
    )
    return all(required_values)


pytestmark = pytest.mark.skipif(
    not _live_tests_enabled(),
    reason="真实分类测试需要显式开关以及DeepSeek、LangSmith凭据",
)


@pytest.mark.asyncio
async def test_live_classify_is_traced_and_contains_only_redacted_input() -> None:
    """验证正式分类节点真实调用DeepSeek，并在LangSmith留下安全Trace。"""
    # 1、从进程环境读取真实配置，并确认测试没有切换到Fake Provider。
    settings = Settings(_env_file=None)
    assert settings.model_provider is ModelProviderName.DEEPSEEK
    assert settings.deepseek_api_key is not None
    assert settings.langsmith_api_key is not None

    # 2、先确认目标LangSmith项目存在，防止模型已调用但追踪投递到了错误项目。
    langsmith_client = Client(api_key=settings.langsmith_api_key.get_secret_value())
    projects = list(langsmith_client.list_projects(name=settings.langsmith_project, limit=1))
    assert len(projects) == 1
    project_id = str(projects[0].id)

    # 3、构造明确的售后场景，并让入站适配器在模型调用前脱敏密码探针。
    state = adapt_run_request(
        "1",
        {
            "tenantId": "tenant-task5-live",
            "customerId": "customer-task5-live",
            "conversationId": "conversation-task5-live",
            "runId": "run-task5-live",
            "triggerMessageId": "message-task5-live",
            "triggerSequence": 1,
            "userMessage": (f"密码={PASSWORD_CANARY}，我购买的成品账号无法登录，请帮我处理售后。"),
            "businessContext": {
                "subject": "成品账号无法登录",
                "orderId": "order-task5-live",
                "orderNo": "AD-TASK5-LIVE",
                "orderStatus": "PAID",
                "entitlementId": "entitlement-task5-live",
                "entitlementType": "ACCOUNT",
                "entitlementStatus": "DELIVERED",
            },
        },
        input_max_chars=settings.input_max_chars,
    )
    assert PASSWORD_CANARY not in state["sanitized_user_message"]
    assert "[REDACTED_PASSWORD]" in state["sanitized_user_message"]

    # 4、在LangSmith显式追踪上下文中执行正式分类节点，而不是单独调用模型探针。
    started_at = datetime.now(UTC) - timedelta(seconds=2)
    runtime = Runtime(
        context=SimpleNamespace(model_provider=DeepSeekModelProvider(settings)),
    )
    with ls.tracing_context(
        client=langsmith_client,
        project_name=settings.langsmith_project,
        enabled=True,
    ):
        result = await classify_node(state, runtime)
    wait_for_all_tracers()

    # 5、验证真实结构化结果已转换为代码固定的售后路由目标。
    decision = result["route_decision"]
    assert decision is not None
    assert decision.intent is Intent.AFTER_SALES
    assert decision.route_code.value == "AFTER_SALES"
    assert decision.agent_code.value == "AFTER_SALES_AGENT"
    assert result["classification_failure"] is None

    # 6、轮询LangSmith直到Trace可读，再验证追踪标识和数据分级元数据。
    trace = await _wait_for_classify_trace(langsmith_client, project_id, started_at)
    assert "root" in (trace.tags or [])
    assert "routing" in (trace.tags or [])
    metadata = trace.metadata or {}
    assert metadata.get("node") == "classify"
    assert metadata.get("data_classification") == "redacted"

    # 7、序列化可观测字段，证明原密码和全部服务密钥均未进入LangSmith。
    trace_payload = json.dumps(
        {
            "name": trace.name,
            "inputs": trace.inputs,
            "outputs": trace.outputs,
            "error": trace.error,
            "extra": trace.extra,
            "events": trace.events,
        },
        ensure_ascii=False,
        default=str,
    )
    assert PASSWORD_CANARY not in trace_payload
    assert "[REDACTED_PASSWORD]" in trace_payload
    _assert_secrets_absent(trace_payload, settings)


async def _wait_for_classify_trace(
    client: Client,
    project_id: str,
    started_at: datetime,
) -> Any:
    """等待LangSmith异步索引完成并返回本次正式分类根Trace。"""
    # 1、在有限时间内轮询，避免外部服务异常导致测试永久等待。
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
        matching_runs = [run for run in runs if run.name == LIVE_RUN_NAME]
        if matching_runs:
            return matching_runs[0]
        await asyncio.sleep(1)
    raise AssertionError("LangSmith未在30秒内返回Task 5分类Trace")


def _assert_secrets_absent(payload: str, settings: Settings) -> None:
    """拒绝DeepSeek、LangSmith或Java服务凭据出现在追踪内容中。"""
    # 1、只在内存中比较SecretStr明文，不输出任何密钥或包含密钥的Trace。
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
