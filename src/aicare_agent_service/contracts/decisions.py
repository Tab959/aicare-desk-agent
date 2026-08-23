"""定义可安全进入 LangGraph 状态的结构化决策、历史、引用和工具结果。

这些模型用于 Python 内部编排，不直接等同于 Java 业务动作。它们通过冻结、未知字段拒绝、
严格类型和敏感字段名检查，限制模型输出能够影响的范围，并为后续路由/RAG/工具节点提供稳定结构。
"""

# ``StrEnum`` 的成员既是枚举又是字符串，便于和 JSON、日志中的固定代码互操作。
from enum import StrEnum

# ``Annotated`` 用来给基础类型附加 Pydantic 字段约束。
from typing import Annotated

# 使用圆括号进行多行导入；这些对象分别负责模型、配置、字段约束、严格类型和验证器。
from pydantic import (
    # 所有内部数据模型的基础类。
    BaseModel,
    # 声明模型级配置。
    ConfigDict,
    # 给字段添加严格范围、长度等约束。
    Field,
    # 只接受真正的 bool，不把 0/1 或字符串自动转为布尔值。
    StrictBool,
    # 只接受真正的 float。
    StrictFloat,
    # 只接受真正的 int；由于 bool 是 int 的子类，严格类型可避免混淆。
    StrictInt,
    # 只接受真正的 str。
    StrictStr,
    # 声明单字段验证函数。
    field_validator,
    # 声明需要观察整个模型的验证函数。
    model_validator,
)

# 复用非空文本和正序号约束。
from aicare_agent_service.contracts.common import NonEmptyText, PositiveSequence

# 人工转接优先级和稳定错误码在事件契约中也会使用，因此从统一位置导入。
from aicare_agent_service.contracts.events import HandoffPriority, StableErrorCode


class SafeInternalModel(BaseModel):
    """可进入 Agent 内部状态的公共冻结模型。"""

    # 禁止未知字段，并禁止构造后修改，减少模型输出“夹带字段”或后续意外篡改。
    model_config = ConfigDict(extra="forbid", frozen=True)


class MessageRole(StrEnum):
    """统一会话参与者角色；值必须与 Java 角色语义一致。"""

    # C 端顾客消息。
    CUSTOMER = "CUSTOMER"
    # Python Agent 生成的 AI 消息。
    AI = "AI"
    # 人工客服消息。
    STAFF = "STAFF"
    # Java 或平台产生的系统消息。
    SYSTEM = "SYSTEM"


class Intent(StrEnum):
    """路由阶段允许识别的有限意图集合。"""

    # 用户明确要求人工客服。
    HUMAN_REQUEST = "HUMAN_REQUEST"
    # 退款、补发、权益异常等售后问题。
    AFTER_SALES = "AFTER_SALES"
    # 查询或解释既有订单。
    ORDER_SUPPORT = "ORDER_SUPPORT"
    # 商品推荐、对比、优惠咨询等售前问题。
    PRE_SALES = "PRE_SALES"
    # 平台规则、商品说明等可由知识库回答的问题。
    KNOWLEDGE = "KNOWLEDGE"
    # 当前系统明确不支持或无法安全判断的请求。
    UNSUPPORTED = "UNSUPPORTED"


class RouteCode(StrEnum):
    """根图条件分支使用的固定路由代码。"""

    # 进入转人工建议节点。
    HUMAN_HANDOFF = "HUMAN_HANDOFF"
    # 进入售后专业分支。
    AFTER_SALES = "AFTER_SALES"
    # 进入订单支持分支。
    ORDER_SUPPORT = "ORDER_SUPPORT"
    # 进入售前导购分支。
    PRE_SALES = "PRE_SALES"
    # 进入可复用知识库RAG子图。
    KNOWLEDGE = "KNOWLEDGE"
    # 进入安全兜底分支。
    UNSUPPORTED = "UNSUPPORTED"


class AgentCode(StrEnum):
    """路由选中的职责单一 Agent 或确定性处理器代码。"""

    # 转人工处理器；它只生成建议，不修改 Java 会话状态。
    HUMAN_HANDOFF = "HUMAN_HANDOFF"
    # 售后专用 Agent。
    AFTER_SALES_AGENT = "AFTER_SALES_AGENT"
    # 订单查询与解释 Agent。
    ORDER_SUPPORT_AGENT = "ORDER_SUPPORT_AGENT"
    # 售前导购 Agent。
    PRE_SALES_AGENT = "PRE_SALES_AGENT"
    # 可复用知识库RAG子图。
    KNOWLEDGE_RAG = "KNOWLEDGE_RAG"
    # 无法处理时的安全回复节点。
    SAFE_FALLBACK = "SAFE_FALLBACK"


class RouteClassification(SafeInternalModel):
    """模型可生成的最小分类结果，不允许模型提供节点名或Agent代码。"""

    # 从固定枚举中选择的单一意图。
    intent: Intent
    # 模型对分类结果的置信度，代码再依据配置阈值决定是否直接路由。
    confidence: Annotated[float, Field(strict=True, ge=0, le=1)]
    # 简短且不含敏感值的分类依据。
    reason: Annotated[NonEmptyText, Field(max_length=240)]


class ToolResultStatus(StrEnum):
    """受控 Java 工具调用结果的有限状态。"""

    # 调用成功并取得允许进入状态的结构化事实。
    SUCCESS = "SUCCESS"
    # 目标资源不存在。
    NOT_FOUND = "NOT_FOUND"
    # Java 因权限、状态或业务规则拒绝操作。
    REJECTED = "REJECTED"
    # Java 或下游服务暂时不可用。
    UNAVAILABLE = "UNAVAILABLE"


class SafeConversationMessage(SafeInternalModel):
    """允许进入 checkpoint 的单条安全会话历史。"""

    # Java 持久化消息 ID。
    message_id: NonEmptyText
    # Java 会话内严格递增序号。
    sequence: PositiveSequence
    # 消息发送者角色，只能取 MessageRole 枚举值。
    role: MessageRole
    # 经策略允许进入 Agent 上下文的非空文本。
    content: NonEmptyText


# 泛型注解说明字典的 key 是 Intent，value 是二元 tuple（RouteCode, AgentCode）。
# 该映射是确定性业务表，不由大模型决定，以保证同一 intent 只能进入一个固定目标。
ROUTE_TARGETS: dict[Intent, tuple[RouteCode, AgentCode]] = {
    # 人工请求固定映射到人工转接处理器。
    Intent.HUMAN_REQUEST: (RouteCode.HUMAN_HANDOFF, AgentCode.HUMAN_HANDOFF),
    # 售后意图固定映射到售后 Agent。
    Intent.AFTER_SALES: (RouteCode.AFTER_SALES, AgentCode.AFTER_SALES_AGENT),
    # 订单意图固定映射到订单支持 Agent。
    Intent.ORDER_SUPPORT: (RouteCode.ORDER_SUPPORT, AgentCode.ORDER_SUPPORT_AGENT),
    # 售前意图固定映射到售前 Agent。
    Intent.PRE_SALES: (RouteCode.PRE_SALES, AgentCode.PRE_SALES_AGENT),
    # 知识意图固定映射到知识库RAG子图。
    Intent.KNOWLEDGE: (RouteCode.KNOWLEDGE, AgentCode.KNOWLEDGE_RAG),
    # 不支持意图固定映射到安全兜底。
    Intent.UNSUPPORTED: (RouteCode.UNSUPPORTED, AgentCode.SAFE_FALLBACK),
}


class RouteDecision(SafeInternalModel):
    """模型给出的结构化路由判断，并由代码再次校验固定映射。"""

    # 模型识别出的有限意图。
    intent: Intent
    # 根图使用的分支代码，必须与 intent 对应。
    route_code: RouteCode
    # 负责处理该请求的 Agent 代码，必须与 intent 对应。
    agent_code: AgentCode
    # 严格浮点置信度，范围是闭区间 [0, 1]。
    confidence: Annotated[float, Field(strict=True, ge=0, le=1)]
    # 不为空的简短判断依据；不得写入敏感原始数据。
    reason: NonEmptyText

    # 装饰器要求 Pydantic 在所有字段完成校验后调用此实例方法。
    @model_validator(mode="after")
    def require_matching_target(self) -> "RouteDecision":
        """保证模型不能为某个 intent 自行选择另一个路由或 Agent。"""
        # 圆括号创建 tuple；字典索引根据 intent 取得代码侧固定的目标 tuple。
        if (self.route_code, self.agent_code) != ROUTE_TARGETS[self.intent]:
            # 抛出 ValueError 后，Pydantic 会把它包装进结构化 ValidationError。
            raise ValueError("route_code和agent_code必须与intent的固定目标一致")
        # after 验证器必须返回通过校验的实例；这里没有修改冻结模型。
        return self


class Citation(SafeInternalModel):
    """RAG 回答引用的可追溯文档位置。"""

    # Java 知识文档 ID。
    document_id: NonEmptyText
    # 文档发布版本，必须是正整数。
    version: PositiveSequence
    # 非空 tuple 表示从文档标题到小节标题的路径；``...`` 表示元素数量可变。
    title_path: Annotated[tuple[NonEmptyText, ...], Field(min_length=1)]
    # 可回溯的来源 URI；不允许把下载凭证放入这里。
    source_uri: NonEmptyText


# 工具 facts 只允许严格标量或 None，不允许嵌套任意响应、列表或对象。
SafeFactValue = StrictStr | StrictInt | StrictFloat | StrictBool | None

# ``frozenset`` 是不可修改集合，适合作为模块级安全规则；查找成员也比 list 更直接。
_FORBIDDEN_FACT_KEY_FRAGMENTS = frozenset(
    {
        # 禁止原始下游响应。
        "rawresponse",
        # 禁止任何 token 字段。
        "token",
        # 禁止密码字段。
        "password",
        # 禁止通用凭据字段。
        "credential",
        # 禁止 secret 字段。
        "secret",
        # 禁止 CDK 及其变体。
        "cdk",
        "cdkcode",
        # 禁止许可证密钥。
        "licensekey",
        # 禁止可能携带访问凭据的下载链接。
        "downloadurl",
        # 禁止异常堆栈。
        "stacktrace",
        # 禁止异常正文。
        "exception",
    }
)


class SafeToolResult(SafeInternalModel):
    """允许进入图状态和 checkpoint 的脱敏工具结果。"""

    # 调用的受控工具名称。
    tool_name: NonEmptyText
    # 工具的有限结果状态。
    status: ToolResultStatus
    # 面向编排和回答的安全摘要。
    summary: NonEmptyText
    # 只允许非空键和 SafeFactValue 标量值的事实字典。
    facts: dict[NonEmptyText, SafeFactValue]

    # 指定只验证 facts 字段；装饰器执行顺序是 Pydantic 先完成基础字典校验，再调用本方法。
    @field_validator("facts")
    # ``classmethod`` 表示第一个参数是类 cls，而不是模型实例 self；字段验证阶段实例尚未完整构造。
    @classmethod
    def reject_sensitive_fact_keys(
        cls, facts: dict[str, SafeFactValue]
    ) -> dict[str, SafeFactValue]:
        """拒绝名称暗示凭据、密钥、原始响应或堆栈的 facts 字段。"""
        # 直接迭代字典会依次得到 key，而不是 value。
        for key in facts:
            # 生成器表达式逐字符处理；只保留字母数字并转小写，使 snake/camel/kebab 写法统一。
            normalized = "".join(character for character in key.lower() if character.isalnum())
            # ``any`` 只要任一禁止片段出现就返回 True，并会短路停止后续检查。
            if any(fragment in normalized for fragment in _FORBIDDEN_FACT_KEY_FRAGMENTS):
                # 错误只指出字段名，不包含敏感字段值。
                raise ValueError(f"facts包含禁止保存的敏感字段：{key}")
        # 验证通过后原样返回字典，Pydantic 会把它保存到模型中。
        return facts


class HandoffSuggestion(SafeInternalModel):
    """Python 向 Java 提出的转人工建议；不代表会话状态已经改变。"""

    # 建议转人工的结构化原因文本。
    reason: NonEmptyText
    # 建议优先级；最终排队规则由 Java 决定。
    priority: HandoffPriority
    # 提供给人工客服的安全上下文摘要。
    summary: NonEmptyText


class EscalationSuggestion(SafeInternalModel):
    """Python 向 Java 提出的升级工单建议；实际工单仍由 Java 校验和创建。"""

    # 只允许大写稳定代码格式的问题类型。
    issue_type: StableErrorCode
    # 建议升级的原因。
    reason: NonEmptyText
    # 提供给工单的安全摘要，不包含敏感权益明文。
    summary: NonEmptyText
