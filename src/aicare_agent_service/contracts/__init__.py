"""Java/Python 内部通信契约包的统一公开入口。

该文件把调用方最常使用的请求、事件、结构化决策和适配函数重新导出。
具体字段仍分别定义在职责单一的模块中，避免所有契约堆在一个大文件里。
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # 静态检查器可以看到公共类型；运行时延迟导入，避免contracts与graph.state形成环。
    from aicare_agent_service.contracts.adapters import (
        UnsupportedContractVersionError,
        adapt_run_request,
    )

# Java 发起一次生成任务的顶层请求模型。
from aicare_agent_service.contracts.agent_run import AgentRunRequest

# 请求中嵌套的最小业务上下文模型。
from aicare_agent_service.contracts.business_context import AgentBusinessContext

# 允许进入内部状态的常用结构化模型。
from aicare_agent_service.contracts.decisions import (
    # RAG 引用。
    Citation,
    # 升级工单建议。
    EscalationSuggestion,
    # 转人工建议。
    HandoffSuggestion,
    # 路由决策。
    RouteDecision,
    # 安全会话历史项。
    SafeConversationMessage,
    # 脱敏工具结果。
    SafeToolResult,
)

# 对外事件联合类型和事件类型枚举。
from aicare_agent_service.contracts.events import AgentEvent, AgentEventType

# ``__all__`` 是 contracts 包承诺的稳定公共名字清单；未列出的内部细节仍可在子模块中使用。
__all__ = [
    # 业务上下文模型。
    "AgentBusinessContext",
    # NDJSON 判别联合类型。
    "AgentEvent",
    # 事件类型枚举。
    "AgentEventType",
    # Agent run 请求模型。
    "AgentRunRequest",
    # RAG 引用模型。
    "Citation",
    # 升级工单建议模型。
    "EscalationSuggestion",
    # 转人工建议模型。
    "HandoffSuggestion",
    # 路由决策模型。
    "RouteDecision",
    # 安全历史模型。
    "SafeConversationMessage",
    # 脱敏工具结果模型。
    "SafeToolResult",
    # 版本异常。
    "UnsupportedContractVersionError",
    # 请求适配函数。
    "adapt_run_request",
]


def __getattr__(name: str) -> Any:
    """按需公开依赖图状态的适配器，避免包初始化阶段形成循环导入。"""
    # 1、只有两个稳定公共适配器允许延迟解析，其他未知名称保持标准AttributeError语义。
    if name not in {"UnsupportedContractVersionError", "adapt_run_request"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # 2、基础契约与graph.state完成初始化后再加载适配模块。
    from aicare_agent_service.contracts import adapters

    # 3、缓存解析结果，后续访问不再重复执行动态查找。
    value = getattr(adapters, name)
    globals()[name] = value
    return value
