import hashlib
import json
from copy import deepcopy

import pytest

from aicare_agent_service.contracts.agent_run import AgentRunRequest
from aicare_agent_service.contracts.common import CONTRACT_HEADER_VERSION
from aicare_agent_service.persistence.identity import (
    build_thread_config,
    canonical_request_digest,
)


@pytest.fixture
def java_payload() -> dict[str, object]:
    return {
        "tenantId": "tenant-demo",
        "customerId": "customer-001",
        "conversationId": "conversation-java-001",
        "runId": "run-java-001",
        "triggerMessageId": "message-java-001",
        "triggerSequence": 7,
        "userMessage": "这个订单为什么还没有交付？",
        "businessContext": {
            "subject": "订单交付",
            "orderId": "order-001",
            "orderNo": "AD2026-001",
            "orderStatus": "PAID",
            "entitlementId": "entitlement-001",
            "entitlementType": "CDK",
            "entitlementStatus": "PENDING",
        },
    }


def test_java_conversation_id_is_the_only_langgraph_thread_id(
    java_payload: dict[str, object],
) -> None:
    request = AgentRunRequest.model_validate(java_payload)

    config = build_thread_config(request)

    assert config == {"configurable": {"thread_id": "conversation-java-001"}}
    assert request.run_id not in json.dumps(config)


def test_canonical_digest_matches_sorted_compact_utf8_json(
    java_payload: dict[str, object],
) -> None:
    request = AgentRunRequest.model_validate(java_payload)
    canonical_payload = {
        "contractVersion": CONTRACT_HEADER_VERSION,
        "request": request.model_dump(by_alias=True, mode="json"),
    }
    expected = hashlib.sha256(
        json.dumps(
            canonical_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    assert canonical_request_digest(CONTRACT_HEADER_VERSION, request) == expected


def test_canonical_digest_is_stable_across_input_key_order(
    java_payload: dict[str, object],
) -> None:
    reversed_payload = dict(reversed(list(java_payload.items())))

    first = canonical_request_digest(
        CONTRACT_HEADER_VERSION,
        AgentRunRequest.model_validate(java_payload),
    )
    second = canonical_request_digest(
        CONTRACT_HEADER_VERSION,
        AgentRunRequest.model_validate(reversed_payload),
    )

    assert first == second


def test_canonical_digest_changes_when_business_input_changes(
    java_payload: dict[str, object],
) -> None:
    changed_payload = deepcopy(java_payload)
    changed_payload["userMessage"] = "我要查询另一个订单"

    original = canonical_request_digest(
        CONTRACT_HEADER_VERSION,
        AgentRunRequest.model_validate(java_payload),
    )
    changed = canonical_request_digest(
        CONTRACT_HEADER_VERSION,
        AgentRunRequest.model_validate(changed_payload),
    )

    assert original != changed


@pytest.mark.parametrize("contract_version", ["", "   "])
def test_canonical_digest_rejects_blank_contract_version(
    java_payload: dict[str, object], contract_version: str
) -> None:
    request = AgentRunRequest.model_validate(java_payload)

    with pytest.raises(ValueError, match="契约版本不能为空"):
        canonical_request_digest(contract_version, request)
