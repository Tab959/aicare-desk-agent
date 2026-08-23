"""装配客服根图的固定拓扑，并要求显式注入专业子图与Checkpointer。"""

from collections.abc import Callable
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from aicare_agent_service.config import Environment
from aicare_agent_service.graph.branches import (
    RootBranch,
    RootBranches,
    invoke_root_branch,
    validate_root_branches,
)
from aicare_agent_service.graph.context import AgentRuntimeContext
from aicare_agent_service.graph.routes import (
    RootRoute,
    select_classification_route,
    select_entry_route,
)
from aicare_agent_service.graph.state import CustomerServiceState
from aicare_agent_service.nodes.classify import (
    build_classification_terminal,
    classify_node,
)
from aicare_agent_service.nodes.context_sync import context_sync_node
from aicare_agent_service.nodes.finalize import finalize_node
from aicare_agent_service.nodes.input_guard import build_input_terminal
from aicare_agent_service.nodes.output_gate import validate_output_state


class RootGraphConfigurationError(ValueError):
    """根图缺少生产依赖或收到非法阈值。"""


def build_customer_service_graph(
    *,
    branches: RootBranches,
    checkpointer: BaseCheckpointSaver[Any] | None,
    environment: Environment,
    direct_confidence: float,
    clarify_confidence: float,
    max_output_chars: int,
) -> CompiledStateGraph:
    """构建并编译唯一客服根图；调用方拥有Checkpointer资源生命周期。"""
    # 1、普通FastAPI部署必须显式拥有Checkpointer；生产拒绝进程内实现。
    if checkpointer is None:
        raise RootGraphConfigurationError("根图必须显式注入Checkpointer")
    if environment is Environment.PRODUCTION and isinstance(checkpointer, InMemorySaver):
        raise RootGraphConfigurationError("生产根图必须使用PostgreSQL Checkpointer")
    # 2、构建固定拓扑后使用调用方持有的Saver编译。
    return _build_customer_service_workflow(
        branches=branches,
        environment=environment,
        direct_confidence=direct_confidence,
        clarify_confidence=clarify_confidence,
        max_output_chars=max_output_chars,
    ).compile(checkpointer=checkpointer)


def build_customer_service_server_graph(
    *,
    branches: RootBranches,
    environment: Environment,
    direct_confidence: float,
    clarify_confidence: float,
    max_output_chars: int,
) -> CompiledStateGraph:
    """为Agent Server编译根图；PostgreSQL Checkpointer由Server统一注入。"""
    # 1、按照Agent Server约定不在应用图中绑定Saver，避免出现双重持久化所有权。
    return _build_customer_service_workflow(
        branches=branches,
        environment=environment,
        direct_confidence=direct_confidence,
        clarify_confidence=clarify_confidence,
        max_output_chars=max_output_chars,
    ).compile()


def _build_customer_service_workflow(
    *,
    branches: RootBranches,
    environment: Environment,
    direct_confidence: float,
    clarify_confidence: float,
    max_output_chars: int,
) -> StateGraph:
    """校验根图依赖并构造未编译的唯一固定工作流。"""
    # 1、四个专业分支必须齐全，阈值必须形成互斥区间。
    validate_root_branches(branches, environment)
    if not 0 <= clarify_confidence < direct_confidence <= 1:
        raise RootGraphConfigurationError("分类置信度阈值非法")
    if max_output_chars <= 0:
        raise RootGraphConfigurationError("输出长度上限必须为正数")

    # 2、注册根图拥有的上下文、安全、分类、门禁和终态节点。
    builder = StateGraph(CustomerServiceState, context_schema=AgentRuntimeContext)
    # node context_sync : 校验Java上下文并同步到状态。
    _add_root_node(builder, "context_sync", context_sync_node)
    # node classify : 结构化分类后根据阈值和固定映射选择专业或确定性节点。
    _add_root_node(builder, "classify", classify_node)
    # node security_block : 安全处置节点，根据状态路由到其他节点。
    _add_root_node(builder, "security_block", build_input_terminal)
    # node human_handoff : 人工转接节点，根据状态路由到其他节点。
    _add_root_node(builder, "human_handoff", build_input_terminal)
    # node input_clarify : 模型澄清节点，根据状态路由到其他节点。
    _add_root_node(builder, "input_clarify", build_input_terminal)
    # node classification_fallback : 分类失败节点，根据状态路由到其他节点。
    _add_root_node(
        builder,
        "classification_fallback",
        _classification_terminal(RootRoute.CLASSIFICATION_FALLBACK),
    )
    # node clarify : 模型澄清节点，根据状态路由到其他节点。
    _add_root_node(builder, "clarify", _classification_terminal(RootRoute.CLARIFY))
    # node unsupported : 不支持请求节点，根据状态路由到其他节点。
    _add_root_node(builder, "unsupported", _classification_terminal(RootRoute.UNSUPPORTED))

    # 3、四个专业节点只调用对应注入子图，并在端口层限制局部更新字段。
    _add_root_node(builder, "pre_sales", _branch_node(branches.pre_sales))
    _add_root_node(builder, "order_support", _branch_node(branches.order_support))
    _add_root_node(builder, "after_sales", _branch_node(branches.after_sales))
    _add_root_node(builder, "knowledge", _branch_node(branches.knowledge_rag))
    _add_root_node(builder, "output_gate", _output_gate_node(max_output_chars))
    # node finalize : 最终节点，根据状态路由到其他节点。
    _add_root_node(builder, "finalize", finalize_node)
    # 4、入口先校验Java上下文，再按图前安全处置选择唯一分支。
    builder.add_edge(START, "context_sync")
    builder.add_conditional_edges(
        "context_sync",
        select_entry_route,
        {
            RootRoute.SECURITY_BLOCK: "security_block",
            RootRoute.HUMAN_HANDOFF: "human_handoff",
            RootRoute.CLARIFY: "input_clarify",
            RootRoute.CLASSIFY: "classify",
        },
    )

    # 5、结构化分类后由代码阈值和固定映射进入一个专业或确定性节点。
    builder.add_conditional_edges(
        "classify",
        lambda state: select_classification_route(
            state,
            direct_confidence,
            clarify_confidence,
        ),
        {
            RootRoute.CLASSIFICATION_FALLBACK: "classification_fallback",
            RootRoute.CLARIFY: "clarify",
            RootRoute.PRE_SALES: "pre_sales",
            RootRoute.ORDER_SUPPORT: "order_support",
            RootRoute.AFTER_SALES: "after_sales",
            RootRoute.KNOWLEDGE: "knowledge",
            RootRoute.HUMAN_HANDOFF: "human_handoff",
            RootRoute.UNSUPPORTED: "unsupported",
        },
    )

    # 6、所有成功路径汇入同一输出门禁和finalize，不返回START也不形成分支循环。
    terminal_sources = (
        "security_block",
        "human_handoff",
        "input_clarify",
        "classification_fallback",
        "clarify",
        "unsupported",
        "pre_sales",
        "order_support",
        "after_sales",
        "knowledge",
    )
    for source in terminal_sources:
        builder.add_edge(source, "output_gate")
    builder.add_edge("output_gate", "finalize")
    builder.add_edge("finalize", END)

    # 7、返回未编译工作流，让FastAPI或Agent Server选择唯一持久化所有者。
    return builder


def _add_root_node(builder: StateGraph, name: str, action: Callable[..., object]) -> None:
    """用稳定节点名和低基数元数据注册一个根图节点。"""
    # 1、节点元数据不包含用户、会话、正文或密钥。
    builder.add_node(
        name,
        action,
        metadata={"layer": "root", "node": name, "data_classification": "redacted"},
    )


def _classification_terminal(
    route: RootRoute,
) -> Callable[[CustomerServiceState], dict[str, object]]:
    """为固定分类终止路由创建无模型节点函数。"""

    # 1、闭包只保存低基数路由枚举，不捕获请求状态。
    def terminal(_: CustomerServiceState) -> dict[str, object]:
        return build_classification_terminal(route)

    return terminal


def _branch_node(branch: RootBranch) -> Callable[..., object]:
    """把统一专业分支端口适配为LangGraph异步节点签名。"""

    # 1、运行时把图状态、配置和不可持久化上下文交给指定子图。
    async def invoke(
        state: CustomerServiceState,
        config: RunnableConfig,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, object]:
        return await invoke_root_branch(branch, state, config, context=runtime.context)

    return invoke


def _output_gate_node(max_output_chars: int) -> Callable[..., object]:
    """创建绑定固定长度上限的确定性输出门禁节点。"""

    # 1、最终身份只和运行时Java期望身份比较，不相信模型或子图返回值。
    def gate(
        state: CustomerServiceState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, object]:
        validate_output_state(
            state,
            expected_identity=runtime.context.expected_identity,
            max_output_chars=max_output_chars,
        )
        return {}

    return cast(Callable[..., object], gate)
