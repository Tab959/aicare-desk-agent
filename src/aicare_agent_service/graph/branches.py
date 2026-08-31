"""定义根图调用四类专业子图的统一端口、部署等级和局部更新边界。"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import Protocol, runtime_checkable

from langchain_core.runnables import RunnableConfig

from aicare_agent_service.config import Environment
from aicare_agent_service.contracts.decisions import Citation
from aicare_agent_service.graph.context import AgentRuntimeContext
from aicare_agent_service.graph.state import CustomerServiceState


class RootBranchDeployment(StrEnum):
    """专业子图实现允许使用的部署范围。"""

    PRODUCTION = "production"
    DEVELOPMENT_ONLY = "development_only"
    TEST_ONLY = "test_only"


@runtime_checkable
class RootBranch(Protocol):
    """根图可调用的专业子图最小异步接口。"""

    deployment_kind: RootBranchDeployment

    async def ainvoke(
        self,
        input: CustomerServiceState,
        config: RunnableConfig,
        *,
        context: AgentRuntimeContext,
    ) -> Mapping[str, object]:
        """接收完整脱敏状态并返回约定的局部状态更新。"""
        ...


class RootBranchConfigurationError(ValueError):
    """专业子图缺失，或部署等级不允许当前运行环境使用。"""


class RootBranchContractError(ValueError):
    """专业子图返回了根图拥有的字段或未知状态字段。"""


@dataclass(frozen=True, slots=True)
class RootBranches:
    """保存售前、订单、售后和知识库四个必需专业子图。"""

    pre_sales: RootBranch
    order_support: RootBranch
    after_sales: RootBranch
    knowledge_rag: RootBranch

    def __post_init__(self) -> None:
        """拒绝任一专业能力缺失。"""
        # 1、四个字段都是生产根图的必需依赖，不能用None占位。
        if any(getattr(self, field.name) is None for field in fields(self)):
            raise RootBranchConfigurationError("根图缺少必需专业子图")


_ALLOWED_BRANCH_UPDATES = frozenset(
    {
        "messages",
        "citations",
        "tool_results",
        "handoff_suggestion",
        "escalation_suggestion",
        "final_answer",
    }
)


def validate_root_branches(branches: RootBranches, environment: Environment) -> None:
    """确认所有子图声明合法部署等级，并阻止生产使用调试实现。"""
    # 1、逐个读取固定字段，拒绝不满足端口或未声明部署等级的对象。
    for field in fields(branches):
        branch = getattr(branches, field.name)
        if not isinstance(branch, RootBranch):
            raise RootBranchConfigurationError("专业子图不满足RootBranch端口")
        # 2、生产环境只接受明确标记为生产实现的专业子图。
        if (
            environment is Environment.PRODUCTION
            and branch.deployment_kind is not RootBranchDeployment.PRODUCTION
        ):
            raise RootBranchConfigurationError("生产环境禁止使用调试专业子图")


async def invoke_root_branch(
    branch: RootBranch,
    state: CustomerServiceState,
    config: RunnableConfig,
    *,
    context: AgentRuntimeContext,
) -> dict[str, object]:
    """调用一个专业子图并校验它只能返回允许的局部字段。"""
    # 1、把当前脱敏状态、LangGraph配置和非持久化依赖传给唯一目标子图。
    raw_update = await branch.ainvoke(state, config, context=context)
    if not isinstance(raw_update, Mapping):
        raise RootBranchContractError("专业子图必须返回状态映射")
    # 2、拒绝identity、route_decision等根图拥有字段以及未知字段。
    unexpected = set(raw_update).difference(_ALLOWED_BRANCH_UPDATES)
    if unexpected:
        raise RootBranchContractError("专业子图返回了禁止字段")
    # 3、知识引用必须是有限的安全契约实例，不能把原始检索响应带回父图。
    citations = raw_update.get("citations")
    if citations is not None and (
        not isinstance(citations, list)
        or len(citations) > 6
        or not all(isinstance(item, Citation) for item in citations)
    ):
        raise RootBranchContractError("专业子图返回了非法引用")
    # 4、复制为普通字典，避免子图在返回后继续修改原映射。
    return dict(raw_update)
