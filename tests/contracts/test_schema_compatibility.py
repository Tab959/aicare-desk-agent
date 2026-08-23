import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage
from pydantic import TypeAdapter, ValidationError

from aicare_agent_service.contracts.adapters import (
    UnsupportedContractVersionError,
    adapt_run_request,
)
from aicare_agent_service.contracts.agent_run import AgentRunRequest
from aicare_agent_service.contracts.events import AgentEvent, AgentEventType

FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "agent_internal_v1.json"
EVENT_ADAPTER = TypeAdapter(AgentEvent)


def load_fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_shared_v1_fixture_validates_request_and_all_event_variants() -> None:
    fixture = load_fixture()

    request = AgentRunRequest.model_validate(fixture["request"])
    events = [EVENT_ADAPTER.validate_python(payload) for payload in fixture["events"]]

    assert fixture["contractVersion"] == "1"
    assert request.conversation_id == "conversation-contract-001"
    assert [event.type for event in events] == list(AgentEventType)


def test_shared_v1_fixture_rejects_the_same_invalid_events() -> None:
    fixture = load_fixture()

    for case in fixture["invalidEvents"]:
        with pytest.raises(ValidationError):
            EVENT_ADAPTER.validate_python(case["payload"])


def test_request_schema_uses_only_required_camel_case_wire_fields() -> None:
    schema = AgentRunRequest.model_json_schema(by_alias=True)

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "tenantId",
        "customerId",
        "conversationId",
        "runId",
        "triggerMessageId",
        "triggerSequence",
        "userMessage",
        "businessContext",
    }
    assert set(schema["properties"]) == set(schema["required"])
    assert "threadId" not in schema["properties"]
    business_schema = schema["$defs"]["AgentBusinessContext"]
    assert business_schema["additionalProperties"] is False
    assert set(business_schema["required"]) == {
        "subject",
        "orderId",
        "orderNo",
        "orderStatus",
        "entitlementId",
        "entitlementType",
        "entitlementStatus",
    }


def test_event_schema_is_a_closed_discriminated_union_with_fixed_enums() -> None:
    schema = EVENT_ADAPTER.json_schema(by_alias=True)

    assert schema["discriminator"]["propertyName"] == "type"
    assert set(schema["discriminator"]["mapping"]) == {event.value for event in AgentEventType}
    for definition_name, definition in schema["$defs"].items():
        if definition_name.endswith("Event"):
            assert definition["additionalProperties"] is False
            assert {
                "type",
                "runId",
                "conversationId",
                "triggerMessageId",
                "triggerSequence",
                "eventIndex",
            }.issubset(definition["required"])


def test_v1_adapter_maps_conversation_to_state_without_transport_leakage() -> None:
    fixture = load_fixture()

    state = adapt_run_request("1", fixture["request"])

    assert state["identity"].conversation_id == "conversation-contract-001"
    assert state["identity"].run_id == "run-contract-001"
    assert len(state["messages"]) == 1
    assert isinstance(state["messages"][0], HumanMessage)
    assert state["messages"][0].content == "CDK 无法激活"
    assert state["business_context"].order_id == "order-contract-001"
    assert "thread_id" not in state
    assert "contract_version" not in state
    assert "raw_request" not in state


def test_adapter_rejects_an_unknown_version_instead_of_silently_accepting_it() -> None:
    with pytest.raises(UnsupportedContractVersionError):
        adapt_run_request("2", load_fixture()["request"])
