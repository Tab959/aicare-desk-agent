"""定义 Java 调用 Python 时的一次 Agent 生成请求。

Java 负责生成所有身份字段并传入 Python。``conversationId`` 对应 LangGraph ``thread_id``，
``runId`` 只标识本次生成；Python 不允许模型自行提供或修改这些身份。
"""

# 业务上下文是请求中的嵌套模型。
from aicare_agent_service.contracts.business_context import AgentBusinessContext

# 导入公共约束类型和所有线模型必须继承的基类。
from aicare_agent_service.contracts.common import NonEmptyText, PositiveSequence, WireContractModel


class AgentRunRequest(WireContractModel):
    """Java 创建的一次 Agent 生成请求；实例冻结且只接受 camelCase JSON。"""

    # 租户 ID，用于多租户身份隔离；必须是去空白后的非空字符串。
    tenant_id: NonEmptyText
    # 当前 C 端顾客 ID，由 Java 鉴权上下文确定。
    customer_id: NonEmptyText
    # Java 会话 ID，同时作为 LangGraph thread_id 跨轮复用。
    conversation_id: NonEmptyText
    # 本次 AI 生成任务 ID，用于幂等、取消和事件关联，不能替代 conversation_id。
    run_id: NonEmptyText
    # 触发本次生成的 Java 持久化消息 ID。
    trigger_message_id: NonEmptyText
    # 触发消息在会话中的严格递增序号；必须是大于等于 1 的严格整数。
    trigger_sequence: PositiveSequence
    # 用户本轮输入；不能为空。后续日志和 Redis ledger 不得保存该正文。
    user_message: NonEmptyText
    # Java 给出的最小业务上下文嵌套对象。
    business_context: AgentBusinessContext
