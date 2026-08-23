"""定义 Python 流向 Java 的 NDJSON 事件契约及单次 run 的事件序列校验器。

每个事件是一行完整 JSON，对外使用 camelCase 字段。所有事件都携带 Java 生成的 run/会话/触发消息身份，
并通过 ``type`` 判别具体模型。序列校验器保证首事件、递增 eventIndex、身份一致和终态互斥。
"""

# StrEnum 的成员可直接作为字符串序列化；适合稳定事件代码。
from enum import StrEnum

# Annotated 附加 Pydantic 元数据；Literal 把字段限制为一个精确字面量。
from typing import Annotated, Literal

# Field 提供判别字段配置；严格布尔避免字符串自动转换；其余对象用于约束和联合类型校验。
from pydantic import Field, StrictBool, StringConstraints, TypeAdapter

# 复用非空文本、正序号和统一 camelCase 线模型规则。
from aicare_agent_service.contracts.common import NonEmptyText, PositiveSequence, WireContractModel

# 稳定错误码仍是 str，但必须去空白、非空，并匹配“大写字母开头，只含大写字母/数字/下划线”。
StableErrorCode = Annotated[
    # 基础 Python 类型。
    str,
    # Pydantic 字符串约束元数据。
    StringConstraints(
        # 校验前移除首尾空白。
        strip_whitespace=True,
        # 至少保留一个字符。
        min_length=1,
        # 正则表达式：``^`` 和 ``$`` 锚定整个字符串，字符组限定可用字符。
        pattern=r"^[A-Z][A-Z0-9_]*$",
    ),
]


class AgentEventType(StrEnum):
    """NDJSON 线上允许出现的全部事件类型。"""

    # Python 已接受 run，Java 收到后不再自动重放整图。
    RUN_ACCEPTED = "RUN_ACCEPTED"
    # 长任务期间维持 Java 空闲超时的临时心跳。
    RUN_HEARTBEAT = "RUN_HEARTBEAT"
    # 路由决策已完成。
    ROUTE_SELECTED = "ROUTE_SELECTED"
    # AI 回复的临时流式文本片段，不由 Java 作为完整消息持久化。
    TOKEN_DELTA = "TOKEN_DELTA"
    # 可由 Java 复核后持久化的完整 AI 回复终态。
    FINAL_MESSAGE = "FINAL_MESSAGE"
    # 请求 Java 执行转人工状态变化的建议终态。
    HANDOFF_REQUESTED = "HANDOFF_REQUESTED"
    # 请求 Java 创建/升级工单的建议终态。
    ESCALATION_REQUESTED = "ESCALATION_REQUESTED"
    # 本次 run 失败终态。
    RUN_FAILED = "RUN_FAILED"


class HandoffPriority(StrEnum):
    """转人工建议的有限优先级。"""

    # 低优先级。
    LOW = "LOW"
    # 普通优先级。
    MEDIUM = "MEDIUM"
    # 高优先级；Java 仍需按自身规则校验。
    HIGH = "HIGH"


class AgentEventEnvelope(WireContractModel):
    """所有 Java/Python NDJSON 事件共享的身份信封。"""

    # 判别具体事件子类的固定类型代码。
    type: AgentEventType
    # Java 为本次生成任务创建的 run ID。
    run_id: NonEmptyText
    # Java 会话 ID，与 LangGraph thread_id 相同。
    conversation_id: NonEmptyText
    # 触发本次 run 的 Java 消息 ID。
    trigger_message_id: NonEmptyText
    # 触发消息的会话序号。
    trigger_sequence: PositiveSequence
    # 本次 run 内严格递增的临时事件序号，从 1 开始。
    event_index: PositiveSequence


class RunAcceptedEvent(AgentEventEnvelope):
    """run 已被 Python 接受；没有额外业务字段。"""

    # Literal 让该子类只接受 RUN_ACCEPTED，不能误装成其他事件。
    type: Literal[AgentEventType.RUN_ACCEPTED]


class RunHeartbeatEvent(AgentEventEnvelope):
    """运行中临时心跳；不会成为用户消息。"""

    # 固定判别值。
    type: Literal[AgentEventType.RUN_HEARTBEAT]


class RouteSelectedEvent(AgentEventEnvelope):
    """Agent 根图已完成路由选择。"""

    # 固定判别值。
    type: Literal[AgentEventType.ROUTE_SELECTED]
    # 选择的根图路由代码。
    route_code: NonEmptyText
    # 选择的职责单一 Agent/处理器代码。
    agent_code: NonEmptyText


class TokenDeltaEvent(AgentEventEnvelope):
    """流式回复中的单个非空文本片段。"""

    # 固定判别值。
    type: Literal[AgentEventType.TOKEN_DELTA]
    # 本次增量文本；仅临时转发给 Java WebSocket。
    content: NonEmptyText


class FinalMessageEvent(AgentEventEnvelope):
    """完整 AI 回复终态。"""

    # 固定判别值。
    type: Literal[AgentEventType.FINAL_MESSAGE]
    # 完整回复正文；Java 保存前仍需复核当前 run 与会话状态。
    content: NonEmptyText


class HandoffRequestedEvent(AgentEventEnvelope):
    """请求 Java 转人工的结构化建议终态。"""

    # 固定判别值。
    type: Literal[AgentEventType.HANDOFF_REQUESTED]
    # 转人工原因。
    reason: NonEmptyText
    # 建议优先级。
    priority: HandoffPriority
    # 供人工客服查看的安全摘要。
    summary: NonEmptyText


class EscalationRequestedEvent(AgentEventEnvelope):
    """请求 Java 升级工单的结构化建议终态。"""

    # 固定判别值。
    type: Literal[AgentEventType.ESCALATION_REQUESTED]
    # 结构化问题类型。
    issue_type: NonEmptyText
    # 升级原因。
    reason: NonEmptyText
    # 工单安全摘要。
    summary: NonEmptyText


class RunFailedEvent(AgentEventEnvelope):
    """run 失败终态；不携带原始异常或堆栈。"""

    # 固定判别值。
    type: Literal[AgentEventType.RUN_FAILED]
    # 可供 Java 分支处理的稳定错误代码。
    error_code: StableErrorCode
    # 严格布尔值，说明 Java 是否可以创建新 run 重试。
    retryable: StrictBool
    # 可安全展示给用户的中文错误信息。
    user_safe_message: NonEmptyText


# ``|`` 把八个模型组成联合类型；外层 Annotated 再告诉 Pydantic 使用 type 字段作 discriminator。
AgentEvent = Annotated[
    # 每一行是联合类型的一个可能成员。
    RunAcceptedEvent
    | RunHeartbeatEvent
    | RouteSelectedEvent
    | TokenDeltaEvent
    | FinalMessageEvent
    | HandoffRequestedEvent
    | EscalationRequestedEvent
    | RunFailedEvent,
    # 判别联合会先读取 type，再只校验对应子模型，错误更明确、性能也更稳定。
    Field(discriminator="type"),
]

# TypeAdapter 让不是 BaseModel 子类的联合类型也能直接执行 validate_python/validate_json。
AGENT_EVENT_ADAPTER = TypeAdapter(AgentEvent)

# 终态事件集合不可修改；一旦出现其中任一类型，本 run 不能再发送事件。
TERMINAL_EVENT_TYPES = frozenset(
    {
        # 正常最终回复。
        AgentEventType.FINAL_MESSAGE,
        # 转人工建议。
        AgentEventType.HANDOFF_REQUESTED,
        # 升级工单建议。
        AgentEventType.ESCALATION_REQUESTED,
        # 失败终态。
        AgentEventType.RUN_FAILED,
    }
)


class AgentEventSequenceError(ValueError):
    """单次 run 的事件身份、顺序或生命周期不符合共享契约。"""


class AgentEventSequenceValidator:
    """在内存中逐条校验单次 run 事件；不负责持久化。"""

    def __init__(
        self,
        *,
        run_id: str,
        conversation_id: str,
        trigger_message_id: str,
        trigger_sequence: int,
    ) -> None:
        """建立绑定到一个 Java run 身份的空序列校验器。

        ``*`` 后面的参数只能按名字传入，例如 ``run_id="..."``，可避免四个相似字符串传错位置。
        """
        # tuple 按固定顺序保存四个身份字段，后续每个事件都必须完全相同。
        self._identity = (run_id, conversation_id, trigger_message_id, trigger_sequence)
        # 0 表示当前还没有接受任何事件；合法线上 eventIndex 从 1 开始。
        self._last_event_index = 0
        # None 表示尚未到达终态；``AgentEventType | None`` 是联合类型。
        self._terminal_type: AgentEventType | None = None

    # property 让调用方用只读属性语法访问方法结果：``validator.last_event_index``。
    @property
    def last_event_index(self) -> int:
        """返回最后一个已接受的 eventIndex；尚无事件时为 0。"""
        # 返回整数副本，不暴露内部可变对象。
        return self._last_event_index

    @property
    def terminal_type(self) -> AgentEventType | None:
        """返回已接受的终态类型；尚未终态时返回 None。"""
        # 枚举成员不可变，可以安全返回。
        return self._terminal_type

    def accept(self, event: AgentEventEnvelope) -> None:
        """校验并记录一条事件；失败时不推进内部序列状态。

        Args:
            event: 已通过具体事件 Schema 校验的公共事件对象。

        Returns:
            ``None``；成功通过内部字段更新来记录进度。

        Raises:
            AgentEventSequenceError: 身份不一致、首事件错误、重复 accepted、序号不递增或终态后继续发送。
        """
        # 从事件取出与构造函数相同顺序的身份 tuple。
        identity = (
            # 本次生成任务身份。
            event.run_id,
            # 会话身份。
            event.conversation_id,
            # 触发消息身份。
            event.trigger_message_id,
            # 触发消息序号。
            event.trigger_sequence,
        )
        # tuple 比较会逐项比较；任一项不同都拒绝，防止跨 run 串流。
        if identity != self._identity:
            raise AgentEventSequenceError("事件身份与当前run不一致")
        # ``is not None`` 明确区分“已有终态”和“还没有终态”。
        if self._terminal_type is not None:
            raise AgentEventSequenceError("终态后不允许继续发送事件")
        # 第一个事件进入专门规则。
        if self._last_event_index == 0:
            # ``or`` 任一条件为真即失败；``is not`` 比较的是同一个枚举成员身份。
            if event.type is not AgentEventType.RUN_ACCEPTED or event.event_index != 1:
                raise AgentEventSequenceError("首事件必须是eventIndex=1的RUN_ACCEPTED")
        # ``elif`` 只在前一个 if 不成立（已经有事件）时检查，禁止再次 accepted。
        elif event.type is AgentEventType.RUN_ACCEPTED:
            raise AgentEventSequenceError("每个run只能发送一次RUN_ACCEPTED")
        # 当前序号必须严格大于已接受序号；相等和倒退都非法。
        if event.event_index <= self._last_event_index:
            raise AgentEventSequenceError("eventIndex必须严格递增")

        # 所有校验通过后才推进内部序号，保证失败不会污染状态。
        self._last_event_index = event.event_index
        # 成员查询 ``in`` 用集合进行判断；只有四种终态会进入分支。
        if event.type in TERMINAL_EVENT_TYPES:
            # 保存终态类型；下一次 accept 会在前面的终态检查中拒绝任何事件。
            self._terminal_type = event.type
