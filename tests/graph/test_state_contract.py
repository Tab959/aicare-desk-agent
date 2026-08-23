from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import ValidationError

from aicare_agent_service.contracts import AgentBusinessContext, AgentRunRequest, SafeToolResult
from aicare_agent_service.graph.context import AgentRuntimeContext, JavaBusinessClient
from aicare_agent_service.graph.state import (
    AgentIdentity,
    AgentIdentityMutationError,
    CustomerServiceState,
    preserve_identity,
)
from aicare_agent_service.models import FakeModelProvider


def run_request() -> AgentRunRequest:
    return AgentRunRequest.model_validate(
        {
            "tenantId": "tenant-001",
            "customerId": "customer-001",
            "conversationId": "conversation-001",
            "runId": "run-001",
            "triggerMessageId": "message-001",
            "triggerSequence": 12,
            "userMessage": "CDK 无法激活",
            "businessContext": {
                "subject": "CDK激活问题",
                "orderId": "order-001",
                "orderNo": "AD20260001",
                "orderStatus": "PAID",
                "entitlementId": "entitlement-001",
                "entitlementType": "CDK",
                "entitlementStatus": "DELIVERED",
            },
        }
    )


def identity() -> AgentIdentity:
    return AgentIdentity.from_request(run_request())


def test_agent_identity_is_derived_from_java_request_and_frozen() -> None:
    value = identity()

    assert value.model_dump(mode="json") == {
        "tenant_id": "tenant-001",
        "customer_id": "customer-001",
        "conversation_id": "conversation-001",
        "run_id": "run-001",
        "trigger_message_id": "message-001",
        "trigger_sequence": 12,
    }
    with pytest.raises(ValidationError):
        value.run_id = "run-other"


def test_preserve_identity_accepts_same_value_without_replacing_it() -> None:
    current = identity()
    same_value = identity()

    result = preserve_identity(current, same_value)

    assert result is current


def test_preserve_identity_accepts_next_java_run_in_same_conversation() -> None:
    current = identity()
    next_run = current.model_copy(
        update={
            "run_id": "run-002",
            "trigger_message_id": "message-002",
            "trigger_sequence": 13,
        }
    )

    result = preserve_identity(current, next_run)

    assert result is next_run


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
def test_preserve_identity_rejects_each_java_owned_field_change(field: str, value: object) -> None:
    current = identity()
    changed = current.model_copy(update={field: value})

    with pytest.raises(AgentIdentityMutationError):
        preserve_identity(current, changed)


def test_add_messages_reducer_appends_node_output() -> None:
    def answer(_: CustomerServiceState) -> dict[str, list[AIMessage]]:
        return {"messages": [AIMessage(content="请先确认激活区域。", id="ai-001")]}

    builder = StateGraph(CustomerServiceState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile()

    result = graph.invoke(
        {
            "identity": identity(),
            "messages": [HumanMessage(content="CDK 无法激活", id="customer-001")],
        }
    )

    assert [(message.type, message.content) for message in result["messages"]] == [
        ("human", "CDK 无法激活"),
        ("ai", "请先确认激活区域。"),
    ]


def test_identity_reducer_blocks_a_graph_node_from_switching_run() -> None:
    def corrupt_identity(state: CustomerServiceState) -> dict[str, AgentIdentity]:
        return {"identity": state["identity"].model_copy(update={"run_id": "run-other"})}

    builder = StateGraph(CustomerServiceState)
    builder.add_node("corrupt_identity", corrupt_identity)
    builder.add_edge(START, "corrupt_identity")
    builder.add_edge("corrupt_identity", END)
    graph = builder.compile()

    with pytest.raises(AgentIdentityMutationError):
        graph.invoke({"identity": identity(), "messages": []})


class ProbeJavaClient:
    def __init__(self, secret_marker: str) -> None:
        self.secret_marker = secret_marker

    async def execute_tool(self, tool_name: str, arguments: dict[str, object]) -> SafeToolResult:
        del arguments
        return SafeToolResult(
            tool_name=tool_name,
            status="SUCCESS",
            summary="测试结果",
            facts={},
        )


def test_runtime_context_is_readable_but_absent_from_serialized_state() -> None:
    secret_marker = "service-token-must-not-enter-state"
    java_client = ProbeJavaClient(secret_marker)
    provider = FakeModelProvider()
    context = AgentRuntimeContext(
        expected_identity=identity(),
        java_client=cast(JavaBusinessClient, java_client),
        model_provider=provider,
        request_deadline=datetime.now(UTC) + timedelta(seconds=30),
    )
    observed_contexts: list[AgentRuntimeContext] = []

    def inspect_context(
        _: CustomerServiceState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, list[AIMessage]]:
        observed_contexts.append(runtime.context)
        assert runtime.context.java_client is java_client
        assert runtime.context.model_provider is provider
        return {"messages": [AIMessage(content="context已读取", id="ai-context")]}

    builder = StateGraph(CustomerServiceState, context_schema=AgentRuntimeContext)
    builder.add_node("inspect_context", inspect_context)
    builder.add_edge(START, "inspect_context")
    builder.add_edge("inspect_context", END)
    graph = builder.compile()

    result = graph.invoke(
        {
            "identity": identity(),
            "messages": [HumanMessage(content="测试上下文", id="customer-context")],
            "business_context": AgentBusinessContext(
                subject=None,
                orderId=None,
                orderNo=None,
                orderStatus=None,
                entitlementId=None,
                entitlementType=None,
                entitlementStatus=None,
            ),
        },
        context=context,
    )
    serialized_type, serialized_state = JsonPlusSerializer().dumps_typed(result)

    assert observed_contexts == [context]
    assert set(result) == {"identity", "messages", "business_context"}
    assert serialized_type == "msgpack"
    assert secret_marker.encode() not in serialized_state


def test_customer_service_state_has_no_runtime_dependency_channels() -> None:
    runtime_only_names = {
        "java_client",
        "model_provider",
        "service_token",
        "database_dsn",
        "raw_response",
    }

    assert runtime_only_names.isdisjoint(CustomerServiceState.__annotations__)
