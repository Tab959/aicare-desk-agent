"""使用真实PostgreSQL验证客服根图同一Java会话跨连接恢复下一轮run。"""

import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from aicare_agent_service.config import CheckpointBackend, Environment, Settings
from aicare_agent_service.contracts.adapters import adapt_run_request
from aicare_agent_service.graph.branches import RootBranchDeployment, RootBranches
from aicare_agent_service.graph.builder import build_customer_service_graph
from aicare_agent_service.models.contracts import ModelPurpose
from aicare_agent_service.models.fake import FakeModelProvider
from aicare_agent_service.persistence.checkpointer import checkpointer_resource
from tests.persistence.postgres_test_support import prepare_postgres_test_connection

pytestmark = pytest.mark.postgres_integration


class PersistentAnswerBranch:
    """为真实持久化测试生成固定、可由Java上下文证明的订单回答。"""

    deployment_kind = RootBranchDeployment.TEST_ONLY

    async def ainvoke(self, input, config, *, context):
        del input, config, context
        return {"final_answer": "已进入订单支持分支，请以Java实时查询结果为准。"}


def state_for(conversation_id: str, run_number: int):
    """构造同一会话中身份递增的Java新run状态。"""
    return adapt_run_request(
        "1",
        {
            "tenantId": "tenant-root-postgres",
            "customerId": "customer-root-postgres",
            "conversationId": conversation_id,
            "runId": f"run-{run_number}",
            "triggerMessageId": f"message-{run_number}",
            "triggerSequence": run_number,
            "userMessage": f"第{run_number}次查询订单状态",
            "businessContext": {
                "subject": "订单查询",
                "orderId": "order-001",
                "orderNo": "AD20260001",
                "orderStatus": "PAID",
                "entitlementId": None,
                "entitlementType": None,
                "entitlementStatus": None,
            },
        },
    )


def runtime_for(state):
    """构造真实根图运行所需的Fake分类上下文。"""
    route = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "RouteClassification",
                "args": {
                    "intent": "ORDER_SUPPORT",
                    "confidence": 0.95,
                    "reason": "订单查询",
                },
                "id": "route-postgres",
                "type": "tool_call",
            }
        ],
    )
    return SimpleNamespace(
        expected_identity=state["identity"],
        java_client=SimpleNamespace(),
        model_provider=FakeModelProvider({ModelPurpose.ROUTING: [route]}),
        request_deadline=datetime.now(UTC) + timedelta(seconds=30),
    )


def branches() -> RootBranches:
    """构造四个满足测试部署等级的确定性分支。"""
    branch = PersistentAnswerBranch()
    return RootBranches(branch, branch, branch, branch)


@pytest.mark.asyncio
async def test_same_java_conversation_restores_next_run_after_postgres_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证conversationId保持thread_id且第二轮替换run身份并保留消息历史。"""
    # 1、普通测试进程不注入外部DSN时跳过；显式集成运行可使用专用或Agent PostgreSQL。
    dsn = os.getenv("AICARE_AGENT_TEST_POSTGRES_DSN") or os.getenv("AICARE_AGENT_POSTGRES_DSN")
    if not dsn:
        pytest.skip("未配置PostgreSQL集成测试连接")
    conninfo, password = prepare_postgres_test_connection(dsn)
    monkeypatch.setenv("PGPASSWORD", password)
    source_settings = Settings(_env_file=None)
    assert source_settings.checkpoint_encryption_key is not None
    settings = Settings(
        environment=Environment.TEST,
        checkpoint_backend=CheckpointBackend.POSTGRES,
        agent_postgres_dsn=conninfo,
        checkpoint_encryption_key=source_settings.checkpoint_encryption_key,
        _env_file=None,
    )

    # 2、第一次连接写入该会话首轮完整根图checkpoint。
    conversation_id = f"root-postgres-{uuid4().hex}"
    config = {"configurable": {"thread_id": conversation_id}}
    first_state = state_for(conversation_id, 1)
    initialized = False
    primary_error: BaseException | None = None
    try:
        async with checkpointer_resource(settings) as first_saver:
            first_graph = build_customer_service_graph(
                branches=branches(),
                checkpointer=first_saver,
                environment=Environment.TEST,
                direct_confidence=0.8,
                clarify_confidence=0.5,
                max_output_chars=500,
            )
            first_result = await first_graph.ainvoke(
                first_state,
                config,
                context=runtime_for(first_state),
            )
            assert first_result["identity"].run_id == "run-1"
            initialized = True

        # 3、关闭首连接后新建Saver，以同一thread提交Java生成的第二轮run身份。
        second_state = state_for(conversation_id, 2)
        async with checkpointer_resource(settings) as second_saver:
            second_graph = build_customer_service_graph(
                branches=branches(),
                checkpointer=second_saver,
                environment=Environment.TEST,
                direct_confidence=0.8,
                clarify_confidence=0.5,
                max_output_chars=500,
            )
            recovered = await second_graph.aget_state(config)
            assert recovered.values["identity"].run_id == "run-1"
            second_result = await second_graph.ainvoke(
                second_state,
                config,
                context=runtime_for(second_state),
            )
            assert second_result["identity"].run_id == "run-2"
            assert second_result["identity"].trigger_sequence == 2
            assert [
                message.id for message in second_result["messages"] if message.type == "human"
            ] == [
                "message-1",
                "message-2",
            ]
    except BaseException as error:
        primary_error = error
        raise
    finally:
        # 4、删除唯一测试thread；清理失败不覆盖更早的真实失败。
        if initialized:
            try:
                async with checkpointer_resource(settings) as cleanup_saver:
                    await cleanup_saver.adelete_thread(conversation_id)
            except Exception:
                if primary_error is None:
                    raise
