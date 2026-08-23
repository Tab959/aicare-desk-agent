"""真实执行DeepSeek、PostgreSQL根图与LangSmith追踪的Task 5整体验收。"""

import asyncio
import json
import os
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import langsmith as ls
import pytest
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client

from aicare_agent_service.config import CheckpointBackend, Environment, Settings
from aicare_agent_service.contracts.adapters import adapt_run_request
from aicare_agent_service.graph.branches import RootBranchDeployment, RootBranches
from aicare_agent_service.graph.builder import build_customer_service_graph
from aicare_agent_service.models.deepseek import DeepSeekModelProvider
from aicare_agent_service.persistence.checkpointer import checkpointer_resource
from tests.persistence.postgres_test_support import prepare_postgres_test_connection

LIVE_FLAG = "AICARE_RUN_LIVE_MODEL_TESTS"
LIVE_RUN_NAME = "task5-live-root-graph"
PASSWORD_CANARY = "task5-root-live-password-canary-61b8"


def _live_tests_enabled() -> bool:
    """仅在显式开启且模型、追踪与PostgreSQL配置完整时运行。"""
    # 1、普通pytest不产生模型费用，也不写外部checkpoint。
    return all(
        (
            os.getenv(LIVE_FLAG) == "1",
            bool(os.getenv("DEEPSEEK_API_KEY", "").strip()),
            bool(os.getenv("LANGSMITH_API_KEY", "").strip()),
            bool(os.getenv("AICARE_AGENT_POSTGRES_DSN", "").strip()),
        )
    )


pytestmark = pytest.mark.skipif(
    not _live_tests_enabled(),
    reason="真实根图测试需要显式开关以及DeepSeek、LangSmith、PostgreSQL配置",
)


class LiveAfterSalesBranch:
    """在Task 7实现前用于根图验收的确定性售后端口测试实现。"""

    deployment_kind = RootBranchDeployment.TEST_ONLY

    async def ainvoke(self, input, config, *, context):
        del input, config, context
        return {"final_answer": "该问题已进入售后处理分支，具体处置需由Java业务接口复核。"}


@pytest.mark.asyncio
async def test_live_root_graph_uses_deepseek_postgres_and_sanitized_langsmith_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证Task 5真实根图完成路由、持久化、终态门禁和脱敏追踪。"""
    # 1、读取真实服务配置，并为Windows psycopg连接隔离密码传递方式。
    source_settings = Settings(_env_file=None)
    assert source_settings.agent_postgres_dsn is not None
    assert source_settings.checkpoint_encryption_key is not None
    assert source_settings.langsmith_api_key is not None
    conninfo, password = prepare_postgres_test_connection(
        source_settings.agent_postgres_dsn.get_secret_value()
    )
    monkeypatch.setenv("PGPASSWORD", password)
    postgres_settings = Settings(
        environment=Environment.TEST,
        checkpoint_backend=CheckpointBackend.POSTGRES,
        agent_postgres_dsn=conninfo,
        checkpoint_encryption_key=source_settings.checkpoint_encryption_key,
        _env_file=None,
    )

    # 2、确认LangSmith目标项目存在，并构造含密码探针的明确售后请求。
    langsmith_client = Client(api_key=source_settings.langsmith_api_key.get_secret_value())
    projects = list(langsmith_client.list_projects(name=source_settings.langsmith_project, limit=1))
    assert len(projects) == 1
    project_id = str(projects[0].id)
    conversation_id = f"task5-live-root-{uuid4().hex}"
    state = adapt_run_request(
        "1",
        {
            "tenantId": "tenant-task5-root-live",
            "customerId": "customer-task5-root-live",
            "conversationId": conversation_id,
            "runId": "run-task5-root-live",
            "triggerMessageId": "message-task5-root-live",
            "triggerSequence": 1,
            "userMessage": f"密码={PASSWORD_CANARY}，购买的成品账号无法登录。",
            "businessContext": {
                "subject": "成品账号无法登录",
                "orderId": "order-task5-root-live",
                "orderNo": "AD-TASK5-ROOT-LIVE",
                "orderStatus": "PAID",
                "entitlementId": "entitlement-task5-root-live",
                "entitlementType": "ACCOUNT",
                "entitlementStatus": "DELIVERED",
            },
        },
        input_max_chars=source_settings.input_max_chars,
    )
    assert PASSWORD_CANARY not in state["sanitized_user_message"]

    # 3、显式注入真实PostgreSQL Saver、DeepSeek Provider和四个受控分支执行完整根图。
    branch = LiveAfterSalesBranch()
    config = {
        "configurable": {"thread_id": conversation_id},
        "run_name": LIVE_RUN_NAME,
        "tags": ["task5-live", "root-graph"],
        "metadata": {"task": "5f", "data_classification": "redacted"},
    }
    runtime_context = SimpleNamespace(
        expected_identity=state["identity"],
        java_client=SimpleNamespace(),
        model_provider=DeepSeekModelProvider(source_settings),
        request_deadline=datetime.now(UTC) + timedelta(seconds=source_settings.run_timeout_seconds),
    )
    started_at = datetime.now(UTC) - timedelta(seconds=2)
    primary_error: BaseException | None = None
    initialized = False
    try:
        async with checkpointer_resource(postgres_settings) as saver:
            graph = build_customer_service_graph(
                branches=RootBranches(branch, branch, branch, branch),
                checkpointer=saver,
                environment=Environment.TEST,
                direct_confidence=source_settings.route_direct_confidence,
                clarify_confidence=source_settings.route_clarify_confidence,
                max_output_chars=source_settings.output_max_chars,
            )
            with ls.tracing_context(
                client=langsmith_client,
                project_name=source_settings.langsmith_project,
                enabled=True,
            ):
                result = await graph.ainvoke(state, config, context=runtime_context)
            initialized = True
            assert result["route_decision"].intent.value == "AFTER_SALES"
            assert result["final_answer"] == (
                "该问题已进入售后处理分支，具体处置需由Java业务接口复核。"
            )
        wait_for_all_tracers()

        # 4、回查根Trace，验证稳定标识、脱敏占位符和全部服务密钥未泄漏。
        trace = await _wait_for_root_trace(langsmith_client, project_id, started_at)
        assert "task5-live" in (trace.tags or [])
        assert "root-graph" in (trace.tags or [])
        metadata = trace.metadata or {}
        assert metadata.get("task") == "5f"
        assert metadata.get("data_classification") == "redacted"
        payload = json.dumps(
            {
                "inputs": trace.inputs,
                "outputs": trace.outputs,
                "error": trace.error,
                "extra": trace.extra,
                "events": trace.events,
            },
            ensure_ascii=False,
            default=str,
        )
        assert PASSWORD_CANARY not in payload
        assert "[REDACTED_PASSWORD]" in payload
        _assert_secrets_absent(payload, source_settings)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        # 5、删除本次唯一测试thread，清理失败不覆盖更早的真实失败。
        if initialized:
            try:
                async with checkpointer_resource(postgres_settings) as cleanup_saver:
                    await cleanup_saver.adelete_thread(conversation_id)
            except Exception:
                if primary_error is None:
                    raise


async def _wait_for_root_trace(client: Client, project_id: str, started_at: datetime) -> Any:
    """在有限时间内等待LangSmith返回本次根图Trace。"""
    # 1、LangSmith异步索引完成前按秒轮询，超时后给出稳定失败。
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
        matches = [run for run in runs if run.name == LIVE_RUN_NAME]
        if matches:
            return matches[0]
        await asyncio.sleep(1)
    raise AssertionError("LangSmith未在30秒内返回Task 5根图Trace")


def _assert_secrets_absent(payload: str, settings: Settings) -> None:
    """拒绝三类服务凭据进入根图Trace。"""
    # 1、只在内存比较SecretStr明文，不输出Trace或密钥。
    secrets = [
        value.get_secret_value()
        for value in (
            settings.deepseek_api_key,
            settings.langsmith_api_key,
            settings.java_service_token,
        )
        if value is not None and value.get_secret_value()
    ]
    if any(secret in payload for secret in secrets):
        raise AssertionError("LangSmith根图Trace包含敏感凭据")
