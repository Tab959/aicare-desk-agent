"""定义 LangGraph 单次运行所需、但不得持久化的依赖上下文。

图状态只保存可安全 checkpoint 的数据；Java 客户端、模型 Provider 和请求截止时间
通过 ``AgentRuntimeContext`` 在运行时注入，避免连接对象或密钥进入持久化状态。
"""

# dataclass 自动生成初始化、比较等样板方法，让纯数据容器更简洁。
from dataclasses import dataclass

# datetime 用来记录单次请求必须完成的绝对截止时间。
from datetime import datetime

# Protocol 使用结构化子类型：对象只要实现规定方法，就可视为符合该协议。
from typing import Protocol

from aicare_agent_service.graph.state import AgentIdentity

# ChatModelProvider 是按用途创建 LangChain 聊天模型的统一接口。
from aicare_agent_service.models.contracts import ChatModelProvider
from aicare_agent_service.tools.contracts import (
    ToolArguments,
    ToolInvocationResult,
    ToolName,
)


# Protocol 类不提供真实实现，只声明后续 Java 客户端适配器必须具备的形状。
class JavaBusinessClient(Protocol):
    """Task 6将实现的受控Java只读工具客户端边界。"""

    # ``async def`` 表示真实工具调用可能涉及网络 I/O，调用处需要使用 await。
    async def execute_tool(
        # self 指向具体客户端实例，由 Python 在调用实例方法时自动传入。
        self,
        *,
        identity: AgentIdentity,
        tool_name: ToolName,
        tool_call_id: str,
        arguments: ToolArguments,
        deadline: datetime,
    ) -> ToolInvocationResult:
        """使用运行时身份和截止时间执行一个固定Java只读工具。"""
        # ``...`` 是 Ellipsis 字面量，在 Protocol 中表示这里只定义接口，不实现方法体。
        ...


# frozen=True 禁止初始化后改字段；slots=True 不创建 __dict__，可减少内存并阻止随意加属性。
@dataclass(frozen=True, slots=True)
class AgentRuntimeContext:
    """单次图调用依赖；通过context_schema注入且不得进入checkpoint。"""

    # 当前Java请求派生的完整身份，根图在入口和终态门禁与状态逐项核对。
    expected_identity: AgentIdentity
    # 受控 Java 工具客户端；图节点只能经此边界查询业务事实。
    java_client: JavaBusinessClient
    # 模型提供者；节点按用途请求模型，而不是直接读取密钥或模型名。
    model_provider: ChatModelProvider
    # 请求绝对截止时间；节点和客户端可据此拒绝已超时的工作。
    request_deadline: datetime
