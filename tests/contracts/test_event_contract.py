from copy import deepcopy

import pytest
from pydantic import TypeAdapter, ValidationError

from aicare_agent_service.contracts.events import (
    AgentEvent,
    AgentEventSequenceError,
    AgentEventSequenceValidator,
    AgentEventType,
    EscalationRequestedEvent,
    FinalMessageEvent,
    HandoffRequestedEvent,
    RouteSelectedEvent,
    RunAcceptedEvent,
    RunFailedEvent,
    RunHeartbeatEvent,
    TokenDeltaEvent,
)

EVENT_ADAPTER = TypeAdapter(AgentEvent)


def base_event(event_type: str, event_index: int = 1) -> dict[str, object]:
    return {
        "type": event_type,
        "runId": "run-001",
        "conversationId": "conversation-001",
        "triggerMessageId": "message-001",
        "triggerSequence": 12,
        "eventIndex": event_index,
    }


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        (base_event("RUN_ACCEPTED"), RunAcceptedEvent),
        (base_event("RUN_HEARTBEAT", 2), RunHeartbeatEvent),
        (
            base_event("ROUTE_SELECTED", 2)
            | {"routeCode": "AFTER_SALES", "agentCode": "CDK_SUPPORT"},
            RouteSelectedEvent,
        ),
        (base_event("TOKEN_DELTA", 2) | {"content": "正在核查"}, TokenDeltaEvent),
        (base_event("FINAL_MESSAGE", 2) | {"content": "请先检查激活区域。"}, FinalMessageEvent),
        (
            base_event("HANDOFF_REQUESTED", 2)
            | {"reason": "用户明确要求人工", "priority": "HIGH", "summary": "已完成基础排查"},
            HandoffRequestedEvent,
        ),
        (
            base_event("ESCALATION_REQUESTED", 2)
            | {"issueType": "CDK_INVALID", "reason": "需要售后处理", "summary": "CDK无法激活"},
            EscalationRequestedEvent,
        ),
        (
            base_event("RUN_FAILED", 2)
            | {
                "errorCode": "MODEL_TIMEOUT",
                "retryable": True,
                "userSafeMessage": "AI客服响应超时，请稍后重试或转人工客服。",
            },
            RunFailedEvent,
        ),
    ],
)
def test_each_event_parses_and_round_trips_camel_case(
    payload: dict[str, object], expected_type: type[object]
) -> None:
    event = EVENT_ADAPTER.validate_python(payload)

    assert isinstance(event, expected_type)
    assert event.model_dump(by_alias=True, mode="json") == payload


@pytest.mark.parametrize(
    ("event_type", "required_field"),
    [
        ("ROUTE_SELECTED", "routeCode"),
        ("ROUTE_SELECTED", "agentCode"),
        ("TOKEN_DELTA", "content"),
        ("FINAL_MESSAGE", "content"),
        ("HANDOFF_REQUESTED", "reason"),
        ("HANDOFF_REQUESTED", "priority"),
        ("HANDOFF_REQUESTED", "summary"),
        ("ESCALATION_REQUESTED", "issueType"),
        ("ESCALATION_REQUESTED", "reason"),
        ("ESCALATION_REQUESTED", "summary"),
        ("RUN_FAILED", "errorCode"),
        ("RUN_FAILED", "retryable"),
        ("RUN_FAILED", "userSafeMessage"),
    ],
)
def test_event_rejects_a_missing_owned_field(event_type: str, required_field: str) -> None:
    fixtures = {
        "ROUTE_SELECTED": {"routeCode": "AFTER_SALES", "agentCode": "CDK_SUPPORT"},
        "TOKEN_DELTA": {"content": "正在核查"},
        "FINAL_MESSAGE": {"content": "请先检查激活区域。"},
        "HANDOFF_REQUESTED": {
            "reason": "用户明确要求人工",
            "priority": "HIGH",
            "summary": "已完成基础排查",
        },
        "ESCALATION_REQUESTED": {
            "issueType": "CDK_INVALID",
            "reason": "需要售后处理",
            "summary": "CDK无法激活",
        },
        "RUN_FAILED": {
            "errorCode": "MODEL_TIMEOUT",
            "retryable": True,
            "userSafeMessage": "AI客服响应超时，请稍后重试或转人工客服。",
        },
    }
    payload = base_event(event_type) | fixtures[event_type]
    del payload[required_field]

    with pytest.raises(ValidationError):
        EVENT_ADAPTER.validate_python(payload)


@pytest.mark.parametrize(
    ("event_type", "foreign_field", "value"),
    [
        ("RUN_ACCEPTED", "content", "不允许"),
        ("RUN_HEARTBEAT", "routeCode", "AFTER_SALES"),
        ("ROUTE_SELECTED", "retryable", True),
        ("TOKEN_DELTA", "priority", "HIGH"),
        ("FINAL_MESSAGE", "issueType", "CDK_INVALID"),
        ("HANDOFF_REQUESTED", "errorCode", "MODEL_TIMEOUT"),
        ("ESCALATION_REQUESTED", "content", "不允许"),
        ("RUN_FAILED", "content", "内部异常"),
    ],
)
def test_event_rejects_fields_owned_by_another_event(
    event_type: str, foreign_field: str, value: object
) -> None:
    valid_payloads = {
        "RUN_ACCEPTED": base_event("RUN_ACCEPTED"),
        "RUN_HEARTBEAT": base_event("RUN_HEARTBEAT"),
        "ROUTE_SELECTED": base_event("ROUTE_SELECTED")
        | {"routeCode": "AFTER_SALES", "agentCode": "CDK_SUPPORT"},
        "TOKEN_DELTA": base_event("TOKEN_DELTA") | {"content": "正在核查"},
        "FINAL_MESSAGE": base_event("FINAL_MESSAGE") | {"content": "处理完成"},
        "HANDOFF_REQUESTED": base_event("HANDOFF_REQUESTED")
        | {"reason": "需要人工", "priority": "HIGH", "summary": "已排查"},
        "ESCALATION_REQUESTED": base_event("ESCALATION_REQUESTED")
        | {"issueType": "CDK_INVALID", "reason": "需要工单", "summary": "无法激活"},
        "RUN_FAILED": base_event("RUN_FAILED")
        | {"errorCode": "MODEL_TIMEOUT", "retryable": True, "userSafeMessage": "请稍后重试。"},
    }
    payload = deepcopy(valid_payloads[event_type])
    payload[foreign_field] = value

    with pytest.raises(ValidationError):
        EVENT_ADAPTER.validate_python(payload)


@pytest.mark.parametrize("field", ["runId", "conversationId", "triggerMessageId"])
@pytest.mark.parametrize("value", ["", "   ", None])
def test_event_rejects_blank_or_missing_identity(field: str, value: object) -> None:
    payload = base_event("RUN_ACCEPTED")
    payload[field] = value

    with pytest.raises(ValidationError):
        EVENT_ADAPTER.validate_python(payload)


@pytest.mark.parametrize("field", ["triggerSequence", "eventIndex"])
@pytest.mark.parametrize("value", [0, -1, True, "2"])
def test_event_rejects_non_strict_positive_sequences(field: str, value: object) -> None:
    payload = base_event("RUN_ACCEPTED")
    payload[field] = value

    with pytest.raises(ValidationError):
        EVENT_ADAPTER.validate_python(payload)


@pytest.mark.parametrize("priority", ["URGENT", "high", ""])
def test_handoff_rejects_unsupported_priority(priority: str) -> None:
    payload = base_event("HANDOFF_REQUESTED") | {
        "reason": "需要人工",
        "priority": priority,
        "summary": "已排查",
    }

    with pytest.raises(ValidationError):
        EVENT_ADAPTER.validate_python(payload)


@pytest.mark.parametrize("error_code", ["model_timeout", "MODEL-TIMEOUT", "MODEL TIMEOUT", ""])
def test_failure_rejects_unstable_error_code(error_code: str) -> None:
    payload = base_event("RUN_FAILED") | {
        "errorCode": error_code,
        "retryable": True,
        "userSafeMessage": "请稍后重试。",
    }

    with pytest.raises(ValidationError):
        EVENT_ADAPTER.validate_python(payload)


def test_sequence_accepts_one_run_from_acceptance_to_one_terminal_event() -> None:
    validator = AgentEventSequenceValidator(
        run_id="run-001",
        conversation_id="conversation-001",
        trigger_message_id="message-001",
        trigger_sequence=12,
    )

    validator.accept(EVENT_ADAPTER.validate_python(base_event("RUN_ACCEPTED", 1)))
    validator.accept(EVENT_ADAPTER.validate_python(base_event("RUN_HEARTBEAT", 2)))
    validator.accept(
        EVENT_ADAPTER.validate_python(base_event("TOKEN_DELTA", 3) | {"content": "正在核查"})
    )
    validator.accept(
        EVENT_ADAPTER.validate_python(base_event("FINAL_MESSAGE", 4) | {"content": "处理完成"})
    )

    assert validator.last_event_index == 4
    assert validator.terminal_type is AgentEventType.FINAL_MESSAGE


@pytest.mark.parametrize(
    "events",
    [
        [base_event("TOKEN_DELTA", 1) | {"content": "缺少接受事件"}],
        [base_event("RUN_ACCEPTED", 2)],
        [base_event("RUN_ACCEPTED", 1), base_event("RUN_ACCEPTED", 2)],
        [base_event("RUN_ACCEPTED", 1), base_event("RUN_HEARTBEAT", 1)],
        [
            base_event("RUN_ACCEPTED", 1),
            base_event("RUN_HEARTBEAT", 3),
            base_event("RUN_HEARTBEAT", 2),
        ],
        [
            base_event("RUN_ACCEPTED", 1),
            base_event("FINAL_MESSAGE", 2) | {"content": "结束"},
            base_event("RUN_HEARTBEAT", 3),
        ],
        [
            base_event("RUN_ACCEPTED", 1),
            base_event("FINAL_MESSAGE", 2) | {"content": "结束"},
            base_event("RUN_FAILED", 3)
            | {"errorCode": "INTERNAL_ERROR", "retryable": False, "userSafeMessage": "失败"},
        ],
    ],
)
def test_sequence_rejects_invalid_order_or_more_than_one_terminal(
    events: list[dict[str, object]],
) -> None:
    validator = AgentEventSequenceValidator(
        run_id="run-001",
        conversation_id="conversation-001",
        trigger_message_id="message-001",
        trigger_sequence=12,
    )

    with pytest.raises(AgentEventSequenceError):
        for payload in events:
            validator.accept(EVENT_ADAPTER.validate_python(payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("runId", "run-other"),
        ("conversationId", "conversation-other"),
        ("triggerMessageId", "message-other"),
        ("triggerSequence", 13),
    ],
)
def test_sequence_rejects_identity_drift(field: str, value: object) -> None:
    validator = AgentEventSequenceValidator(
        run_id="run-001",
        conversation_id="conversation-001",
        trigger_message_id="message-001",
        trigger_sequence=12,
    )
    validator.accept(EVENT_ADAPTER.validate_python(base_event("RUN_ACCEPTED", 1)))
    payload = base_event("RUN_HEARTBEAT", 2)
    payload[field] = value

    with pytest.raises(AgentEventSequenceError):
        validator.accept(EVENT_ADAPTER.validate_python(payload))
