"""验证根图在调用模型或子图前校验Java身份、thread和触发消息上下文。"""

import importlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from langgraph.runtime import Runtime

from aicare_agent_service.contracts.adapters import adapt_run_request


def context_sync_module():
    """延迟导入待实现的上下文节点。"""
    return importlib.import_module("aicare_agent_service.nodes.context_sync")


def initial_state():
    """构造已完成输入安全预处理的初始状态。"""
    return adapt_run_request(
        "1",
        {
            "tenantId": "tenant-001",
            "customerId": "customer-001",
            "conversationId": "conversation-001",
            "runId": "run-001",
            "triggerMessageId": "message-001",
            "triggerSequence": 12,
            "userMessage": "查询订单状态",
            "businessContext": {
                "subject": None,
                "orderId": "order-001",
                "orderNo": "AD20260001",
                "orderStatus": "PAID",
                "entitlementId": None,
                "entitlementType": None,
                "entitlementStatus": None,
            },
        },
    )


def runtime_for(state, *, deadline: datetime | None = None):
    """构造仅含5C校验所需字段的LangGraph运行上下文。"""
    context = SimpleNamespace(
        expected_identity=state["identity"],
        request_deadline=deadline or datetime.now(UTC) + timedelta(seconds=30),
    )
    return Runtime(context=context)


def test_context_sync_accepts_consistent_java_context() -> None:
    state = initial_state()

    result = context_sync_module().context_sync_node(
        state,
        {"configurable": {"thread_id": "conversation-001"}},
        runtime_for(state),
    )

    assert result == {
        "route_decision": None,
        "classification_failure": None,
        "citations": [],
        "tool_results": [],
        "handoff_suggestion": None,
        "escalation_suggestion": None,
        "final_answer": None,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", "tenant-other"),
        ("customer_id", "customer-other"),
        ("conversation_id", "conversation-other"),
        ("run_id", "run-other"),
        ("trigger_message_id", "message-other"),
        ("trigger_sequence", 13),
    ],
)
def test_context_sync_rejects_identity_different_from_java_request(
    field: str,
    value: object,
) -> None:
    state = initial_state()
    context = SimpleNamespace(
        expected_identity=state["identity"].model_copy(update={field: value}),
        request_deadline=datetime.now(UTC) + timedelta(seconds=30),
    )

    with pytest.raises(context_sync_module().ContextConsistencyError):
        context_sync_module().context_sync_node(
            state,
            {"configurable": {"thread_id": "conversation-001"}},
            Runtime(context=context),
        )


def test_context_sync_rejects_thread_different_from_java_conversation() -> None:
    state = initial_state()

    with pytest.raises(context_sync_module().ContextConsistencyError):
        context_sync_module().context_sync_node(
            state,
            {"configurable": {"thread_id": "conversation-other"}},
            runtime_for(state),
        )


def test_context_sync_rejects_trigger_message_mismatch() -> None:
    state = initial_state()
    state["safe_history"][0] = state["safe_history"][0].model_copy(
        update={"message_id": "message-other"}
    )

    with pytest.raises(context_sync_module().ContextConsistencyError):
        context_sync_module().context_sync_node(
            state,
            {"configurable": {"thread_id": "conversation-001"}},
            runtime_for(state),
        )


def test_context_sync_rejects_expired_or_timezone_naive_deadline() -> None:
    state = initial_state()
    module = context_sync_module()

    with pytest.raises(module.RequestDeadlineExceededError):
        module.context_sync_node(
            state,
            {"configurable": {"thread_id": "conversation-001"}},
            runtime_for(state, deadline=datetime.now(UTC) - timedelta(seconds=1)),
        )

    with pytest.raises(module.ContextConsistencyError):
        module.context_sync_node(
            state,
            {"configurable": {"thread_id": "conversation-001"}},
            runtime_for(state, deadline=datetime.now(UTC).replace(tzinfo=None)),
        )
