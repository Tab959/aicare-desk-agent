"""验证 Java 只读工具线契约拒绝漂移、越权参数和越界数据。"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from aicare_agent_service.tools.contracts import (
    MAX_TOOL_PAYLOAD_BYTES,
    TOOL_ARGUMENT_MODELS,
    TOOL_CONTRACT_VERSION,
    AgentToolIdentity,
    AgentToolInvokeRequest,
    AgentToolInvokeResponse,
    GamePageData,
    GameSummary,
    SearchGamesArguments,
    ToolName,
    WalletTransaction,
    validate_tool_arguments,
    validate_tool_payload_limits,
)


def _identity_payload() -> dict[str, object]:
    return {
        "tenantId": "tenant-demo",
        "customerId": "user-customer-001",
        "conversationId": "conversation-001",
        "runId": "run-001",
        "triggerMessageId": "message-001",
        "triggerSequence": 7,
    }


def test_tool_catalog_contains_exactly_the_twenty_read_only_capabilities() -> None:
    assert len(ToolName) == 20
    assert set(TOOL_ARGUMENT_MODELS) == set(ToolName)
    assert "reveal_entitlement_delivery" not in {item.value for item in ToolName}


def test_search_arguments_reject_unknown_identity_and_oversized_page() -> None:
    with pytest.raises(ValidationError):
        SearchGamesArguments.model_validate(
            {"query": "动作游戏", "limit": 21, "customerId": "forged-user"}
        )


def test_cursor_is_opaque_bounded_text_instead_of_page_number() -> None:
    with pytest.raises(ValidationError):
        SearchGamesArguments.model_validate({"query": "动作游戏", "cursor": "x" * 513})

    arguments = SearchGamesArguments.model_validate({"query": "动作游戏", "cursor": "opaque-v1"})
    assert arguments.cursor == "opaque-v1"


def test_invoke_request_requires_camel_case_complete_java_identity() -> None:
    request = AgentToolInvokeRequest.model_validate(
        {
            "contractVersion": TOOL_CONTRACT_VERSION,
            "toolCallId": "call-001",
            "identity": _identity_payload(),
            "arguments": {"query": "动作游戏", "limit": 5},
        }
    )
    assert request.identity == AgentToolIdentity.model_validate(_identity_payload())

    with pytest.raises(ValidationError):
        AgentToolInvokeRequest.model_validate(
            {
                "contractVersion": TOOL_CONTRACT_VERSION,
                "toolCallId": "call-001",
                "identity": {**_identity_payload(), "unexpected": "blocked"},
                "arguments": {},
            }
        )


def test_each_tool_arguments_are_validated_by_its_fixed_schema() -> None:
    with pytest.raises(ValidationError):
        validate_tool_arguments(ToolName.GET_GAME_DETAIL, {"gameId": "game-1", "url": "/admin"})

    parsed = validate_tool_arguments(ToolName.GET_GAME_DETAIL, {"gameId": "game-1"})
    assert parsed.model_dump(by_alias=True) == {"gameId": "game-1"}


def test_response_rejects_data_shape_that_does_not_match_tool_name() -> None:
    data = GamePageData(
        kind="GAME_PAGE",
        items=(
            GameSummary(
                game_id="game-1",
                name="示例游戏",
                price=Decimal("99.00"),
                currency="CNY",
                purchase_methods=("CDK",),
                available=True,
            ),
        ),
        next_cursor=None,
    )
    response = AgentToolInvokeResponse(
        contract_version=TOOL_CONTRACT_VERSION,
        request_id="request-001",
        tool_call_id="call-001",
        tool_name=ToolName.SEARCH_GAMES,
        status="SUCCESS",
        observed_at=datetime(2026, 8, 15, tzinfo=UTC),
        data=data,
    )
    assert response.data.kind == "GAME_PAGE"

    with pytest.raises(ValidationError):
        AgentToolInvokeResponse(
            contract_version=TOOL_CONTRACT_VERSION,
            request_id="request-001",
            tool_call_id="call-001",
            tool_name=ToolName.GET_WALLET,
            status="SUCCESS",
            observed_at=datetime(2026, 8, 15, tzinfo=UTC),
            data=data,
        )


def test_wallet_transaction_allows_debit_but_balance_remains_non_negative() -> None:
    transaction = WalletTransaction.model_validate(
        {
            "transactionId": "transaction-001",
            "transactionType": "ORDER_PAYMENT",
            "amount": "-99.00",
            "balanceAfter": "201.00",
            "occurredAt": "2026-08-15T10:00:00Z",
            "orderId": "order-001",
        }
    )
    assert transaction.amount == Decimal("-99.00")

    with pytest.raises(ValidationError):
        WalletTransaction.model_validate(
            {
                "transactionId": "transaction-001",
                "transactionType": "ORDER_PAYMENT",
                "amount": "-99.00",
                "balanceAfter": "-1.00",
                "occurredAt": "2026-08-15T10:00:00Z",
                "orderId": "order-001",
            }
        )


def test_payload_limits_reject_oversized_or_deep_cross_service_data() -> None:
    with pytest.raises(ValueError, match="大小"):
        validate_tool_payload_limits({"value": "x" * MAX_TOOL_PAYLOAD_BYTES})

    with pytest.raises(ValueError, match="嵌套"):
        validate_tool_payload_limits(
            {"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": 1}}}}}}}}}
        )
