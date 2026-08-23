"""验证20个LangChain工具注册、Runtime隐藏和最小权限能力包。"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast

import pytest

from aicare_agent_service.contracts.decisions import SafeToolResult, ToolResultStatus
from aicare_agent_service.tools import registry as registry_module
from aicare_agent_service.tools.contracts import AgentToolInvokeResponse, ToolName
from aicare_agent_service.tools.registry import (
    READ_ONLY_TOOL_REGISTRY,
    CapabilityPackage,
    tools_for_capability,
)


def test_registry_contains_exactly_twenty_frozen_read_only_tools() -> None:
    assert frozenset(READ_ONLY_TOOL_REGISTRY) == frozenset(ToolName)
    assert len(READ_ONLY_TOOL_REGISTRY) == 20
    assert all(
        registration.risk.value == "READ_ONLY" for registration in READ_ONLY_TOOL_REGISTRY.values()
    )


def test_registry_builder_rejects_duplicate_or_missing_contract() -> None:
    first = registry_module._SPECS[0]
    with pytest.raises(ValueError, match="重复注册"):
        registry_module._build_registry((first, first))

    incomplete = replace(first, response_model=cast(Any, None))
    with pytest.raises(ValueError, match="缺少实现或契约"):
        registry_module._build_registry((incomplete,))


def test_model_visible_schemas_never_expose_runtime_or_internal_connection_fields() -> None:
    forbidden = {"runtime", "user_id", "tenant_id", "url", "token", "http_method"}

    for registration in READ_ONLY_TOOL_REGISTRY.values():
        properties = registration.tool.tool_call_schema.model_json_schema().get("properties", {})
        assert forbidden.isdisjoint(properties)


def test_internal_tool_schema_accepts_runtime_while_model_schema_hides_it() -> None:
    registration = READ_ONLY_TOOL_REGISTRY[ToolName.SEARCH_GAMES]

    assert "runtime" in registration.tool.args_schema.model_fields
    assert "runtime" not in registration.tool.tool_call_schema.model_fields


def test_capability_packages_are_immutable_and_least_privilege() -> None:
    presales = frozenset(tool.name for tool in tools_for_capability(CapabilityPackage.PRE_SALES))
    orders = frozenset(tool.name for tool in tools_for_capability(CapabilityPackage.ORDER_SUPPORT))
    after_sales = frozenset(
        tool.name for tool in tools_for_capability(CapabilityPackage.AFTER_SALES)
    )

    assert presales == frozenset(tool.value for tool in tuple(ToolName)[:10])
    assert orders == {
        ToolName.LIST_ORDERS.value,
        ToolName.GET_ORDER_DETAIL.value,
        ToolName.GET_WALLET.value,
        ToolName.LIST_WALLET_TRANSACTIONS.value,
        ToolName.INSPECT_ENTITLEMENT_STATUS.value,
    }
    assert after_sales == {
        ToolName.LIST_ORDERS.value,
        ToolName.GET_ORDER_DETAIL.value,
        ToolName.INSPECT_ENTITLEMENT_STATUS.value,
    }
    assert ToolName.GET_WALLET.value not in after_sales


def test_trace_metadata_is_low_cardinality_and_contains_no_identity() -> None:
    for registration in READ_ONLY_TOOL_REGISTRY.values():
        assert registration.tool.metadata == {
            "tool_name": registration.name.value,
            "domain": registration.domain.value,
            "risk": "READ_ONLY",
        }
        assert set(registration.tool.metadata or {}).isdisjoint(
            {"customer_id", "conversation_id", "run_id", "tool_call_id"}
        )


@pytest.mark.asyncio
async def test_tool_wrapper_uses_runtime_identity_and_returns_safe_content_and_evidence() -> None:
    captured: dict[str, object] = {}

    class ProbeJavaClient:
        async def execute_tool(self, **kwargs: object) -> AgentToolInvokeResponse:
            captured.update(kwargs)
            return AgentToolInvokeResponse.model_validate(
                {
                    "contractVersion": "1.0",
                    "requestId": "request-1",
                    "toolCallId": "call-1",
                    "toolName": "get_wallet",
                    "status": "SUCCESS",
                    "observedAt": "2026-08-16T00:00:00Z",
                    "data": {
                        "kind": "WALLET",
                        "availableBalance": "100.00",
                        "currency": "CNY",
                    },
                }
            )

    identity = SimpleNamespace(
        tenant_id="tenant-1",
        customer_id="customer-1",
        conversation_id="conversation-1",
        run_id="run-1",
        trigger_message_id="message-1",
        trigger_sequence=1,
    )
    deadline = datetime.now(UTC) + timedelta(seconds=5)
    runtime = SimpleNamespace(
        tool_call_id="call-1",
        context=SimpleNamespace(
            java_client=ProbeJavaClient(),
            expected_identity=identity,
            request_deadline=deadline,
        ),
    )
    coroutine = READ_ONLY_TOOL_REGISTRY[ToolName.GET_WALLET].tool.coroutine
    assert coroutine is not None

    content, artifact = await coroutine(runtime=cast(Any, runtime))

    assert content == '{"availableBalance":"100.00","currency":"CNY","kind":"WALLET"}'
    assert isinstance(artifact, SafeToolResult)
    assert artifact.status is ToolResultStatus.SUCCESS
    assert artifact.facts == {
        "kind": "WALLET",
        "observed_at": "2026-08-16T00:00:00+00:00",
    }
    assert captured["identity"] is identity
    assert captured["tool_call_id"] == "call-1"
    assert captured["deadline"] == deadline


def test_evidence_projection_keeps_only_whitelisted_scalar_quantity_and_status() -> None:
    response = AgentToolInvokeResponse.model_validate(
        {
            "contractVersion": "1.0",
            "requestId": "request-1",
            "toolCallId": "call-1",
            "toolName": "preview_checkout",
            "status": "SUCCESS",
            "observedAt": "2026-08-16T00:00:00Z",
            "data": {
                "kind": "CHECKOUT_PREVIEW",
                "itemCount": 2,
                "totalAmount": "80.00",
                "currency": "CNY",
                "available": True,
            },
        }
    )

    artifact = registry_module._project_evidence(ToolName.PREVIEW_CHECKOUT, response)

    assert artifact.facts == {
        "kind": "CHECKOUT_PREVIEW",
        "observed_at": "2026-08-16T00:00:00+00:00",
        "item_count": 2,
        "available": True,
    }
    assert "total_amount" not in artifact.facts
