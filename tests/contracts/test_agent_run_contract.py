from copy import deepcopy

import pytest
from pydantic import ValidationError

from aicare_agent_service.contracts.agent_run import AgentRunRequest
from aicare_agent_service.contracts.common import (
    CONTRACT_HEADER_NAME,
    CONTRACT_HEADER_VERSION,
)


@pytest.fixture
def java_payload() -> dict[str, object]:
    return {
        "tenantId": "tenant-demo",
        "customerId": "user-customer-001",
        "conversationId": "conv-001",
        "runId": "run-001",
        "triggerMessageId": "message-001",
        "triggerSequence": 12,
        "userMessage": "CDK 无法激活",
        "businessContext": {
            "subject": "CDK 激活问题",
            "orderId": "order-001",
            "orderNo": "AD2026-001",
            "orderStatus": "PAID",
            "entitlementId": "entitlement-001",
            "entitlementType": "CDK",
            "entitlementStatus": "DELIVERED",
        },
    }


def test_contract_version_is_transport_metadata() -> None:
    assert CONTRACT_HEADER_NAME == "X-Contract-Version"
    assert CONTRACT_HEADER_VERSION == "1"


def test_java_payload_round_trips_without_field_drift(java_payload: dict[str, object]) -> None:
    request = AgentRunRequest.model_validate(java_payload)

    assert request.conversation_id == "conv-001"
    assert request.user_message == "CDK 无法激活"
    assert request.model_dump(by_alias=True, mode="json") == java_payload


@pytest.mark.parametrize(
    "field",
    [
        "tenantId",
        "customerId",
        "conversationId",
        "runId",
        "triggerMessageId",
        "triggerSequence",
        "userMessage",
        "businessContext",
    ],
)
def test_required_request_fields_cannot_be_missing(
    java_payload: dict[str, object], field: str
) -> None:
    payload = deepcopy(java_payload)
    payload.pop(field)

    with pytest.raises(ValidationError):
        AgentRunRequest.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    ["tenantId", "customerId", "conversationId", "runId", "triggerMessageId", "userMessage"],
)
@pytest.mark.parametrize("value", ["", "   ", None])
def test_identity_and_message_fields_reject_blank_values(
    java_payload: dict[str, object], field: str, value: object
) -> None:
    payload = deepcopy(java_payload)
    payload[field] = value

    with pytest.raises(ValidationError):
        AgentRunRequest.model_validate(payload)


@pytest.mark.parametrize("value", [0, -1, True, "12"])
def test_trigger_sequence_is_a_strict_positive_integer(
    java_payload: dict[str, object], value: object
) -> None:
    payload = deepcopy(java_payload)
    payload["triggerSequence"] = value

    with pytest.raises(ValidationError):
        AgentRunRequest.model_validate(payload)


@pytest.mark.parametrize("field", ["threadId", "contractVersion", "recentMessages", "content"])
def test_request_rejects_fields_outside_v1(java_payload: dict[str, object], field: str) -> None:
    payload = deepcopy(java_payload)
    payload[field] = "unexpected"

    with pytest.raises(ValidationError):
        AgentRunRequest.model_validate(payload)


def test_wire_model_rejects_python_field_names(java_payload: dict[str, object]) -> None:
    payload = deepcopy(java_payload)
    payload["conversation_id"] = payload.pop("conversationId")

    with pytest.raises(ValidationError):
        AgentRunRequest.model_validate(payload)


@pytest.mark.parametrize("field", ["cdk", "password", "downloadUrl", "token", "balance"])
def test_business_context_rejects_sensitive_or_unknown_fields(
    java_payload: dict[str, object], field: str
) -> None:
    payload = deepcopy(java_payload)
    context = payload["businessContext"]
    assert isinstance(context, dict)
    context[field] = "must-not-enter-agent"

    with pytest.raises(ValidationError):
        AgentRunRequest.model_validate(payload)


def test_business_context_requires_all_seven_nullable_fields(
    java_payload: dict[str, object],
) -> None:
    payload = deepcopy(java_payload)
    context = payload["businessContext"]
    assert isinstance(context, dict)
    context["subject"] = None
    context["orderId"] = None
    context["orderNo"] = None
    context["orderStatus"] = None
    context["entitlementId"] = None
    context["entitlementType"] = None
    context["entitlementStatus"] = None

    request = AgentRunRequest.model_validate(payload)
    assert request.business_context.model_dump(by_alias=True) == context

    context.pop("orderId")
    with pytest.raises(ValidationError):
        AgentRunRequest.model_validate(payload)


def test_wire_models_trim_text_and_are_frozen(java_payload: dict[str, object]) -> None:
    payload = deepcopy(java_payload)
    payload["tenantId"] = "  tenant-demo  "
    request = AgentRunRequest.model_validate(payload)

    assert request.tenant_id == "tenant-demo"
    with pytest.raises(ValidationError):
        request.tenant_id = "other-tenant"
