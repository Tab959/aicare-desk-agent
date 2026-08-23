"""逐项验证共享OpenAPI与Python的20工具请求、响应和关键拒绝规则。"""

import json
from pathlib import Path

import jsonschema_rs
import pytest
from pydantic import ValidationError

from aicare_agent_service.tools.contracts import (
    TOOL_CONTRACT_VERSION,
    AgentToolInvokeRequest,
    AgentToolInvokeResponse,
    ToolName,
    validate_tool_arguments,
)
from tests.fixtures.java_tool_responses import TOOL_SUCCESS_CASES


def _contract_document() -> dict[str, object]:
    contract_path = (
        Path(__file__).resolve().parents[2] / "resources" / "docs" / "api" / "agent-tools-v1.yaml"
    )
    return json.loads(contract_path.read_text(encoding="utf-8"))


def _schema(document: dict[str, object], name: str) -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": f"#/components/schemas/{name}",
        "components": document["components"],
    }


def _identity() -> dict[str, object]:
    return {
        "tenantId": "tenant-demo",
        "customerId": "user-customer-001",
        "conversationId": "conversation-001",
        "runId": "run-001",
        "triggerMessageId": "message-001",
        "triggerSequence": 1,
    }


@pytest.mark.parametrize("tool_name", list(ToolName), ids=lambda value: value.value)
def test_each_tool_success_case_matches_shared_openapi_and_python_models(
    tool_name: ToolName,
) -> None:
    document = _contract_document()
    argument_schema, arguments, data = TOOL_SUCCESS_CASES[tool_name]

    # 1、同一组业务参数必须同时通过工具专属OpenAPI Schema与Python严格模型。
    jsonschema_rs.validate(_schema(document, argument_schema), arguments)
    validated_arguments = validate_tool_arguments(tool_name, arguments)

    # 2、完整请求信封必须符合共享OpenAPI，且身份只来自固定运行时样例。
    request = {
        "contractVersion": TOOL_CONTRACT_VERSION,
        "toolCallId": f"call-{tool_name.value}",
        "identity": _identity(),
        "arguments": validated_arguments.model_dump(mode="json", by_alias=True),
    }
    jsonschema_rs.validate(_schema(document, "AgentToolInvokeRequest"), request)
    AgentToolInvokeRequest.model_validate(request)

    # 3、工具名、响应种类和成功信封必须同时通过OpenAPI与Python交叉校验。
    response = {
        "contractVersion": TOOL_CONTRACT_VERSION,
        "requestId": f"request-{tool_name.value}",
        "toolCallId": f"call-{tool_name.value}",
        "toolName": tool_name.value,
        "status": "SUCCESS",
        "observedAt": "2026-08-16T06:00:00Z",
        "data": data,
    }
    jsonschema_rs.validate(_schema(document, "AgentToolInvokeResponse"), response)
    AgentToolInvokeResponse.model_validate(response)


def test_contract_matrix_is_complete_and_rejects_critical_drift() -> None:
    assert set(TOOL_SUCCESS_CASES) == set(ToolName)
    assert len(TOOL_SUCCESS_CASES) == 20

    with pytest.raises((jsonschema_rs.ValidationError, ValidationError)):
        AgentToolInvokeRequest.model_validate(
            {
                "contractVersion": "2.0",
                "toolCallId": "call-invalid",
                "identity": {**_identity(), "serviceToken": "blocked"},
                "arguments": {},
            }
        )

    with pytest.raises(ValidationError):
        validate_tool_arguments(ToolName.GET_WALLET, {"customerId": "forged"})

    _, _, wrong_data = TOOL_SUCCESS_CASES[ToolName.GET_WALLET]
    with pytest.raises(ValidationError):
        AgentToolInvokeResponse.model_validate(
            {
                "contractVersion": TOOL_CONTRACT_VERSION,
                "requestId": "request-invalid",
                "toolCallId": "call-invalid",
                "toolName": ToolName.SEARCH_GAMES.value,
                "status": "SUCCESS",
                "observedAt": "2026-08-16T06:00:00Z",
                "data": wrong_data,
            }
        )
