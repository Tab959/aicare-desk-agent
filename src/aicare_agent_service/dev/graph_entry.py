"""组装Agent Server可检查的正式根图；未完成的专业子图保持显式失败。"""

from collections.abc import Mapping

from langchain_core.runnables import RunnableConfig

from aicare_agent_service.config import Environment, get_settings
from aicare_agent_service.graph.branches import (
    RootBranchConfigurationError,
    RootBranchDeployment,
    RootBranches,
)
from aicare_agent_service.graph.builder import build_customer_service_server_graph
from aicare_agent_service.graph.context import AgentRuntimeContext
from aicare_agent_service.graph.state import CustomerServiceState


class UnavailableSpecialistBranch:
    """Task 7/8尚未装配专业子图时使用的开发入口失败边界。"""

    deployment_kind = RootBranchDeployment.DEVELOPMENT_ONLY

    async def ainvoke(
        self,
        input: CustomerServiceState,
        config: RunnableConfig,
        *,
        context: AgentRuntimeContext,
    ) -> Mapping[str, object]:
        """拒绝伪造专业回复，不提供内存、Mock或静默回退业务能力。"""
        # 1、显式消费端口参数后向调用层传播稳定装配错误。
        del input, config, context
        raise RootBranchConfigurationError("Task 7/8专业子图尚未装配")


# 1、Agent Server从.env读取阈值和输出上限，模型与Java客户端仍由每次run上下文注入。
settings = get_settings()
# 2、四个未完成端口只会失败，不产生任何演示业务答案；Task 7/8将替换为真实子图。
unavailable_branch = UnavailableSpecialistBranch()
branches = RootBranches(
    unavailable_branch,
    unavailable_branch,
    unavailable_branch,
    unavailable_branch,
)
# 3、图本身不绑定Saver；langgraph.json注册的Server资源统一注入PostgreSQL Checkpointer。
graph = build_customer_service_server_graph(
    branches=branches,
    environment=Environment.DEVELOPMENT,
    direct_confidence=settings.route_direct_confidence,
    clarify_confidence=settings.route_clarify_confidence,
    max_output_chars=settings.output_max_chars,
)
