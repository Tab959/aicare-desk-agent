"""客服 LangGraph 核心类型的统一公开入口。

外部模块可从 ``aicare_agent_service.graph`` 一处导入运行时上下文、状态、身份模型
和身份 reducer，无需了解它们分别位于哪个内部文件。本文件本身不构建或执行图。
"""

# 导出不会进入 checkpoint 的运行时依赖上下文，以及 Java 业务客户端协议。
from aicare_agent_service.graph.context import AgentRuntimeContext, JavaBusinessClient

# 从 state 子模块集中导入会进入图状态的类型与身份保护逻辑。
from aicare_agent_service.graph.state import (
    AgentIdentity,
    AgentIdentityMutationError,
    CustomerServiceState,
    preserve_identity,
)

# ``__all__`` 明确此包承诺给调用者使用的公共名称，未列出的内部名称不属于稳定接口。
__all__ = [
    "AgentIdentity",
    "AgentIdentityMutationError",
    "AgentRuntimeContext",
    "CustomerServiceState",
    "JavaBusinessClient",
    "preserve_identity",
]
