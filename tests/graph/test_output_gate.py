"""验证根图输出门禁只允许安全、可追溯且唯一的终态。"""

import importlib

import pytest
from langchain_core.messages import AIMessage

from aicare_agent_service.contracts.adapters import adapt_run_request
from aicare_agent_service.contracts.decisions import (
    Citation,
    EscalationSuggestion,
    HandoffSuggestion,
    SafeToolResult,
)
from aicare_agent_service.contracts.events import HandoffPriority


def output_gate_module():
    """延迟导入待实现的确定性输出门禁。"""
    return importlib.import_module("aicare_agent_service.nodes.output_gate")


def finalize_module():
    """延迟导入待实现的终态确认节点。"""
    return importlib.import_module("aicare_agent_service.nodes.finalize")


def state_for():
    """构造带Java可信订单上下文的安全状态。"""
    return adapt_run_request(
        "1",
        {
            "tenantId": "tenant-output",
            "customerId": "customer-output",
            "conversationId": "conversation-output",
            "runId": "run-output",
            "triggerMessageId": "message-output",
            "triggerSequence": 1,
            "userMessage": "查询订单状态",
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


def test_output_gate_accepts_one_safe_final_answer_with_supported_fact() -> None:
    state = state_for()
    state["final_answer"] = "订单 AD20260001 当前状态为 PAID。"
    state["citations"] = []
    state["tool_results"] = [
        SafeToolResult(
            tool_name="get_order_detail",
            status="SUCCESS",
            summary="订单状态查询成功",
            facts={"orderNo": "AD20260001", "orderStatus": "PAID"},
        )
    ]

    assert (
        output_gate_module().validate_output_state(
            state,
            expected_identity=state["identity"],
            max_output_chars=200,
        )
        is None
    )
    assert finalize_module().finalize_node(state) == {}


@pytest.mark.parametrize(
    "terminal_updates",
    [
        {},
        {
            "final_answer": "请稍候。",
            "handoff_suggestion": HandoffSuggestion(
                reason="用户请求",
                priority=HandoffPriority.MEDIUM,
                summary="请求人工。",
            ),
        },
        {
            "handoff_suggestion": HandoffSuggestion(
                reason="用户请求",
                priority=HandoffPriority.MEDIUM,
                summary="请求人工。",
            ),
            "escalation_suggestion": EscalationSuggestion(
                issue_type="ACCOUNT_FAILURE",
                reason="账号异常",
                summary="需要Java复核。",
            ),
        },
    ],
)
def test_finalize_rejects_zero_or_multiple_terminals(
    terminal_updates: dict[str, object],
) -> None:
    state = state_for()
    state.update(terminal_updates)

    with pytest.raises(finalize_module().FinalStateValidationError):
        finalize_module().finalize_node(state)


def test_output_gate_rejects_identity_mutation() -> None:
    state = state_for()
    expected = state["identity"]
    state["identity"] = expected.model_copy(update={"customer_id": "customer-other"})
    state["final_answer"] = "无法处理。"

    with pytest.raises(output_gate_module().OutputGateError, match="身份"):
        output_gate_module().validate_output_state(
            state,
            expected_identity=expected,
            max_output_chars=200,
        )


@pytest.mark.parametrize(
    "answer",
    [
        "密码=output-secret-canary-92f1",
        "已为您完成退款。",
        "已经帮您取消订单。",
        "售后工单已为您创建成功。",
        "订单 AD99999999 当前状态为 REFUNDED。",
    ],
)
def test_output_gate_rejects_sensitive_unsupported_or_executed_claims(answer: str) -> None:
    state = state_for()
    state["final_answer"] = answer

    with pytest.raises(output_gate_module().OutputGateError):
        output_gate_module().validate_output_state(
            state,
            expected_identity=state["identity"],
            max_output_chars=200,
        )


def test_output_gate_rejects_oversized_answer() -> None:
    state = state_for()
    state["final_answer"] = "正常内容" * 20

    with pytest.raises(output_gate_module().OutputGateError, match="长度"):
        output_gate_module().validate_output_state(
            state,
            expected_identity=state["identity"],
            max_output_chars=10,
        )


def test_output_gate_rejects_realtime_status_supported_only_by_request_snapshot() -> None:
    state = state_for()
    state["final_answer"] = "订单 AD20260001 当前状态为 PAID。"
    state["tool_results"] = []

    with pytest.raises(output_gate_module().OutputGateError, match="实时工具证据"):
        output_gate_module().validate_output_state(
            state,
            expected_identity=state["identity"],
            max_output_chars=200,
        )


def test_output_gate_rejects_sensitive_text_in_branch_message() -> None:
    state = state_for()
    state["final_answer"] = "暂时无法确认。"
    state["messages"] = [AIMessage(content="密码=branch-output-canary-73a2")]

    with pytest.raises(output_gate_module().OutputGateError, match="消息"):
        output_gate_module().validate_output_state(
            state,
            expected_identity=state["identity"],
            max_output_chars=200,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("citations", [{"document_id": "doc-1"}]),
        ("tool_results", [{"tool_name": "get_order"}]),
    ],
)
def test_output_gate_rejects_unvalidated_citations_or_tool_results(
    field: str,
    invalid_value: list[dict[str, str]],
) -> None:
    state = state_for()
    state["final_answer"] = "暂时无法确认。"
    state[field] = invalid_value

    with pytest.raises(output_gate_module().OutputGateError, match="契约"):
        output_gate_module().validate_output_state(
            state,
            expected_identity=state["identity"],
            max_output_chars=200,
        )


def test_output_gate_accepts_valid_citation_and_safe_tool_result() -> None:
    state = state_for()
    state["final_answer"] = "订单 AD20260001 当前状态为 PAID。"
    state["citations"] = [
        Citation(
            document_id="document-001",
            version=1,
            title_path=("订单帮助",),
            source_uri="https://help.example.test/orders",
        )
    ]
    state["tool_results"] = [
        SafeToolResult(
            tool_name="get_order_detail",
            status="SUCCESS",
            summary="订单状态查询成功",
            facts={"orderNo": "AD20260001", "orderStatus": "PAID"},
        )
    ]

    output_gate_module().validate_output_state(
        state,
        expected_identity=state["identity"],
        max_output_chars=200,
    )
