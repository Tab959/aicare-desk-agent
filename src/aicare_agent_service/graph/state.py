"""定义客服 LangGraph 可持久化状态及不可变运行身份。

``AgentIdentity`` 保存 Java 分配的会话/run 标识，``preserve_identity`` 作为 reducer
阻止节点篡改这些标识；``CustomerServiceState`` 列出可写入 checkpoint 的脱敏字段。
运行时连接和模型对象不在这里，而是在 ``graph.context`` 中注入。
"""

# Annotated 可为类型附加 reducer 元数据；TypedDict 用字段声明描述普通字典的静态结构。
from typing import Annotated, TypedDict

# AnyMessage 是 LangChain 支持的任意标准消息联合类型。
from langchain_core.messages import AnyMessage

# add_messages 是 LangGraph 专用消息 reducer，可按消息 ID 追加或替换消息。
from langgraph.graph.message import add_messages

# BaseModel 提供运行时校验；ConfigDict 用于集中配置模型行为。
from pydantic import BaseModel, ConfigDict

# AgentRunRequest 是 Java 发起单次 AI run 时提交的完整入站契约。
from aicare_agent_service.contracts.agent_run import AgentRunRequest

# AgentBusinessContext 是 Java 提供的脱敏业务上下文快照。
from aicare_agent_service.contracts.business_context import AgentBusinessContext

# 复用公共强约束文本和正整数序号类型，避免在状态层重复校验规则。
from aicare_agent_service.contracts.common import NonEmptyText, PositiveSequence

# 导入路由、引用、工具结果、转人工等可 checkpoint 的结构化决策类型。
from aicare_agent_service.contracts.decisions import (
    Citation,
    EscalationSuggestion,
    HandoffSuggestion,
    RouteDecision,
    SafeConversationMessage,
    SafeToolResult,
)
from aicare_agent_service.security.contracts import (
    InputSafetyAssessment,
    RouteClassificationFailure,
)


# 继承 ValueError 表明问题来自“不允许的身份字段值变化”。
class AgentIdentityMutationError(ValueError):
    """图节点尝试改写Java拥有的run身份。"""


# BaseModel 让身份对象在创建时接受 Pydantic 的类型和约束校验。
class AgentIdentity(BaseModel):
    """由Java请求派生且在一次图运行中不可变化的身份。"""

    # extra="forbid" 拒绝未知字段；frozen=True 使实例创建后不可赋值修改。
    model_config = ConfigDict(extra="forbid", frozen=True)

    # 租户 ID，用于隔离不同租户的数据和工具访问范围。
    tenant_id: NonEmptyText
    # 当前顾客 ID，身份来源是 Java 鉴权上下文而不是模型输出。
    customer_id: NonEmptyText
    # Java 生成的会话 ID，同时作为 LangGraph thread_id 使用。
    conversation_id: NonEmptyText
    # Java 为本轮 AI 生成任务创建的 run ID，用于去重和单飞校验。
    run_id: NonEmptyText
    # 触发本轮生成的 Java 消息 ID，最终结果回写前需再次核对。
    trigger_message_id: NonEmptyText
    # 触发消息在会话中的严格递增序号，必须是正整数。
    trigger_sequence: PositiveSequence

    # classmethod 把类本身作为 cls 传入，因此子类调用时也能构建正确的子类实例。
    @classmethod
    def from_request(cls, request: AgentRunRequest) -> "AgentIdentity":
        """从已校验的 AgentRunRequest 复制 Java 拥有的身份字段。

        ``request`` 是入站请求；返回新的不可变 AgentIdentity。字符串形式的返回类型
        避免在类体尚未完全定义时直接引用类本身。
        """
        # 调用 cls(...) 会触发 Pydantic 再次验证所有字段约束。
        return cls(
            # 以下每个值都直接来自 Java 请求，不做猜测或重写。
            tenant_id=request.tenant_id,
            customer_id=request.customer_id,
            conversation_id=request.conversation_id,
            run_id=request.run_id,
            trigger_message_id=request.trigger_message_id,
            trigger_sequence=request.trigger_sequence,
        )


def preserve_identity(
    # current 是 checkpoint 中已有身份；首次写入时还不存在，所以允许 None。
    current: AgentIdentity | None,
    # update 是某节点本次尝试写入 identity 字段的新值。
    update: AgentIdentity,
) -> AgentIdentity:
    """接受同一会话的初始、相同或更高序号Java run，拒绝非法身份切换。"""
    # 1、第一次建立thread状态时接受Java请求派生的完整身份。
    if current is None:
        return update
    # 2、同一run的重复合并复用旧对象，避免无意义替换。
    if current == update:
        return current
    # 3、同租户、用户和会话可接续Java生成的更高消息序号新run。
    same_conversation = (
        current.tenant_id == update.tenant_id
        and current.customer_id == update.customer_id
        and current.conversation_id == update.conversation_id
    )
    complete_next_run = (
        update.trigger_sequence > current.trigger_sequence
        and update.run_id != current.run_id
        and update.trigger_message_id != current.trigger_message_id
    )
    if same_conversation and complete_next_run:
        return update
    # 4、跨身份、倒退序号或只修改部分run字段都立即阻断。
    raise AgentIdentityMutationError("图节点不得非法修改Java拥有的Agent身份")


# total=False 表示 TypedDict 的所有字段在静态类型层面都可缺省，适合逐节点增量构建状态。
class CustomerServiceState(TypedDict, total=False):
    """只包含可checkpoint的脱敏会话状态。"""

    # Annotated 的第二个参数指定 reducer：连续更新必须经过身份保护函数合并。
    identity: Annotated[AgentIdentity, preserve_identity]
    # 消息列表使用 add_messages 合并，避免普通列表更新直接覆盖历史消息。
    messages: Annotated[list[AnyMessage], add_messages]
    # Java 提供的当前订单、权益等脱敏业务快照；实时事实仍须通过工具重新查询。
    business_context: AgentBusinessContext
    # Java 提供的安全历史消息，不包含不应发送给模型的敏感字段。
    safe_history: list[SafeConversationMessage]
    # 长会话压缩后的摘要；None 表示尚未生成摘要。
    conversation_summary: str | None
    # 售前、售后等结构化路由结果；分类节点执行前可以为 None。
    route_decision: RouteDecision | None
    # 图前安全预处理产生的脱敏当前用户正文。
    sanitized_user_message: str
    # 输入安全标签、脱敏计数、处置和稳定原因码。
    input_safety_assessment: InputSafetyAssessment
    # 分类失败时保存稳定代码，不保存模型原始响应。
    classification_failure: RouteClassificationFailure | None
    # 最终回答所引用的知识来源列表。
    citations: list[Citation]
    # 经脱敏和字段白名单过滤后的 Java 工具调用结果。
    tool_results: list[SafeToolResult]
    # 请求转人工的结构化建议；Python 只建议，由 Java 决定是否变更会话状态。
    handoff_suggestion: HandoffSuggestion | None
    # 升级售后工单的结构化建议；Python 不直接创建或修改工单。
    escalation_suggestion: EscalationSuggestion | None
    # 图完成后准备交给 Java 的完整 AI 文本；流式 token 不直接持久化。
    final_answer: str | None
