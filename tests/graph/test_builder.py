"""验证正式根图只能显式注入专业分支和Checkpointer后编译。"""

import importlib
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from aicare_agent_service.config import Environment
from aicare_agent_service.contracts.adapters import adapt_run_request
from aicare_agent_service.graph.branches import RootBranchDeployment, RootBranches
from aicare_agent_service.models.contracts import ModelPurpose
from aicare_agent_service.models.fake import FakeModelProvider


def builder_module():
    """延迟导入待实现的正式根图builder。"""
    return importlib.import_module("aicare_agent_service.graph.builder")


class AnswerBranch:
    """返回唯一安全最终回答的确定性专业分支。"""

    deployment_kind = RootBranchDeployment.TEST_ONLY

    async def ainvoke(self, input, config, *, context):
        del input, config, context
        return {"final_answer": "已进入订单支持分支，请以Java实时查询结果为准。"}


def state_for():
    """构造可由Fake路由到订单分支的初始状态。"""
    return adapt_run_request(
        "1",
        {
            "tenantId": "tenant-builder",
            "customerId": "customer-builder",
            "conversationId": "conversation-builder",
            "runId": "run-builder",
            "triggerMessageId": "message-builder",
            "triggerSequence": 1,
            "userMessage": "查询订单",
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


def runtime_for(state):
    """构造根图所需的不可持久化运行上下文。"""
    route = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "RouteClassification",
                "args": {
                    "intent": "ORDER_SUPPORT",
                    "confidence": 0.95,
                    "reason": "用户查询订单",
                },
                "id": "route-builder",
                "type": "tool_call",
            }
        ],
    )
    return SimpleNamespace(
        expected_identity=state["identity"],
        java_client=SimpleNamespace(),
        model_provider=FakeModelProvider({ModelPurpose.ROUTING: [route]}),
        request_deadline=datetime.now(UTC) + timedelta(seconds=30),
    )


@pytest.mark.asyncio
async def test_builder_runs_one_branch_through_output_gate_and_finalize() -> None:
    branch = AnswerBranch()
    branches = RootBranches(branch, branch, branch, branch)
    graph = builder_module().build_customer_service_graph(
        branches=branches,
        checkpointer=InMemorySaver(),
        environment=Environment.TEST,
        direct_confidence=0.8,
        clarify_confidence=0.5,
        max_output_chars=500,
    )
    state = state_for()

    result = await graph.ainvoke(
        state,
        {"configurable": {"thread_id": "conversation-builder"}},
        context=runtime_for(state),
    )

    assert result["final_answer"] == "已进入订单支持分支，请以Java实时查询结果为准。"
    assert result["handoff_suggestion"] is None
    assert result["escalation_suggestion"] is None


def test_builder_rejects_memory_checkpointer_in_production() -> None:
    branch = AnswerBranch()
    branches = RootBranches(branch, branch, branch, branch)

    with pytest.raises(builder_module().RootGraphConfigurationError, match="PostgreSQL"):
        builder_module().build_customer_service_graph(
            branches=branches,
            checkpointer=InMemorySaver(),
            environment=Environment.PRODUCTION,
            direct_confidence=0.8,
            clarify_confidence=0.5,
            max_output_chars=500,
        )


def test_builder_requires_checkpointer_and_all_branches() -> None:
    branch = AnswerBranch()
    branches = RootBranches(branch, branch, branch, branch)

    with pytest.raises(builder_module().RootGraphConfigurationError, match="Checkpointer"):
        builder_module().build_customer_service_graph(
            branches=branches,
            checkpointer=None,
            environment=Environment.TEST,
            direct_confidence=0.8,
            clarify_confidence=0.5,
            max_output_chars=500,
        )


def test_agent_server_builder_compiles_without_application_owned_checkpointer() -> None:
    branch = AnswerBranch()

    graph = builder_module().build_customer_service_server_graph(
        branches=RootBranches(branch, branch, branch, branch),
        environment=Environment.TEST,
        direct_confidence=0.8,
        clarify_confidence=0.5,
        max_output_chars=500,
    )

    assert graph.checkpointer is None
    assert "finalize" in graph.get_graph().nodes


def test_root_topology_has_no_loop_or_shared_tool_node_and_all_success_paths_finalize() -> None:
    branch = AnswerBranch()
    graph = (
        builder_module()
        .build_customer_service_graph(
            branches=RootBranches(branch, branch, branch, branch),
            checkpointer=InMemorySaver(),
            environment=Environment.TEST,
            direct_confidence=0.8,
            clarify_confidence=0.5,
            max_output_chars=500,
        )
        .get_graph()
    )
    edges = {(edge.source, edge.target) for edge in graph.edges}
    branch_names = {"pre_sales", "order_support", "after_sales", "knowledge"}

    assert "tool_node" not in graph.nodes
    assert "tool_approval" not in graph.nodes
    assert all(target != "__start__" for _, target in edges)
    assert all((name, "output_gate") in edges for name in branch_names)
    assert all(target == "output_gate" for source, target in edges if source in branch_names)
    assert ("output_gate", "finalize") in edges
    assert ("finalize", "__end__") in edges


def test_root_nodes_expose_only_stable_low_cardinality_metadata() -> None:
    branch = AnswerBranch()
    graph = (
        builder_module()
        .build_customer_service_graph(
            branches=RootBranches(branch, branch, branch, branch),
            checkpointer=InMemorySaver(),
            environment=Environment.TEST,
            direct_confidence=0.8,
            clarify_confidence=0.5,
            max_output_chars=500,
        )
        .get_graph()
    )

    for name, node in graph.nodes.items():
        if name in {"__start__", "__end__"}:
            continue
        assert node.metadata == {
            "layer": "root",
            "node": name,
            "data_classification": "redacted",
        }


class FailingBranch(AnswerBranch):
    """抛出编程错误以验证根图不把异常伪装为客服回复。"""

    async def ainvoke(self, input, config, *, context):
        del input, config, context
        raise RuntimeError("branch-programming-error")


@pytest.mark.asyncio
async def test_unexpected_branch_error_bubbles_to_run_lifecycle() -> None:
    answer = AnswerBranch()
    graph = builder_module().build_customer_service_graph(
        branches=RootBranches(answer, FailingBranch(), answer, answer),
        checkpointer=InMemorySaver(),
        environment=Environment.TEST,
        direct_confidence=0.8,
        clarify_confidence=0.5,
        max_output_chars=500,
    )
    state = state_for()

    with pytest.raises(RuntimeError, match="branch-programming-error"):
        await graph.ainvoke(
            state,
            {"configurable": {"thread_id": "conversation-builder"}},
            context=runtime_for(state),
        )
