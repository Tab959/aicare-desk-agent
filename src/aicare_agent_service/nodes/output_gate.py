"""在根图结束前确定性校验身份、契约对象、敏感内容和事实证据边界。"""

import re
from collections.abc import Iterable

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from aicare_agent_service.contracts.decisions import (
    Citation,
    EscalationSuggestion,
    HandoffSuggestion,
    SafeToolResult,
)
from aicare_agent_service.graph.state import AgentIdentity, CustomerServiceState
from aicare_agent_service.security.redaction import redact_sensitive_input


class OutputGateError(ValueError):
    """最终状态违反可机械验证的安全、身份或事实边界。"""


_EXECUTED_ACTION_PATTERN = re.compile(
    r"(?:已为您|已经帮您|已替您).{0,12}(?:退款|取消订单|修改订单|创建工单|补发)|"
    r"(?:退款|取消订单|工单|补发).{0,8}(?:操作成功|处理完成|创建成功)"
)
_ORDER_REFERENCE_PATTERN = re.compile(r"\bAD[A-Z0-9-]{4,}\b", re.IGNORECASE)
_STATUS_REFERENCE_PATTERN = re.compile(
    r"(?:状态(?:为|是|：|:)\s*)([A-Z][A-Z0-9_]{2,})",
    re.IGNORECASE,
)


def validate_output_state(
    state: CustomerServiceState,
    *,
    expected_identity: AgentIdentity,
    max_output_chars: int,
) -> None:
    """验证根图最终输出，不调用模型、不修改状态也不执行Java业务动作。"""
    # 1、终态身份必须仍等于Java为当前run注入的身份。
    if state.get("identity") != expected_identity:
        raise OutputGateError("最终状态身份与Java请求不一致")
    if max_output_chars <= 0:
        raise OutputGateError("输出长度上限必须为正数")

    # 2、引用、工具结果和建议必须已经通过项目安全契约模型校验。
    citations = state.get("citations", [])
    tool_results = state.get("tool_results", [])
    if not isinstance(citations, list) or not all(isinstance(item, Citation) for item in citations):
        raise OutputGateError("引用没有通过安全契约校验")
    if not isinstance(tool_results, list) or not all(
        isinstance(item, SafeToolResult) for item in tool_results
    ):
        raise OutputGateError("工具结果没有通过安全契约校验")
    _validate_suggestion_contracts(state)

    # 3、对三类终态的全部用户可见文本统一执行长度与敏感模式检查。
    output_texts = tuple(_terminal_texts(state))
    if sum(len(text) for text in output_texts) > max_output_chars:
        raise OutputGateError("最终输出超过允许长度")
    for text in output_texts:
        if redact_sensitive_input(text).sanitized_text != text:
            raise OutputGateError("最终输出包含未脱敏敏感内容")
        if _EXECUTED_ACTION_PATTERN.search(text):
            raise OutputGateError("Python不得声称已完成Java业务动作")

    # 4、专业分支追加的消息正文也属于模型输出，必须通过同一敏感模式检查。
    messages = state.get("messages", [])
    if not isinstance(messages, list) or not all(
        isinstance(message, BaseMessage) for message in messages
    ):
        raise OutputGateError("消息没有通过LangChain契约校验")
    for message in messages:
        for text in _nested_text_values(message.content):
            if redact_sensitive_input(text).sanitized_text != text:
                raise OutputGateError("消息包含未脱敏敏感内容")

    # 5、订单号可由Java上下文定位，实时状态必须来自本轮安全工具结果。
    final_answer = state.get("final_answer")
    if isinstance(final_answer, str):
        order_evidence = _order_reference_evidence(state)
        order_references = set(_ORDER_REFERENCE_PATTERN.findall(final_answer))
        if any(reference.lower() not in order_evidence for reference in order_references):
            raise OutputGateError("最终回答包含缺少证据的结构化业务事实")
        tool_evidence = _tool_fact_evidence(state)
        status_references = set(_STATUS_REFERENCE_PATTERN.findall(final_answer))
        if any(reference.lower() not in tool_evidence for reference in status_references):
            raise OutputGateError("最终回答的状态缺少实时工具证据")

    # 6、引用和工具安全模型的全部字符串字段也必须保持脱敏。
    for model in (*citations, *tool_results):
        for text in _model_text_values(model):
            if redact_sensitive_input(text).sanitized_text != text:
                raise OutputGateError("证据契约包含未脱敏敏感内容")


def _validate_suggestion_contracts(state: CustomerServiceState) -> None:
    """验证可选转人工与工单建议只能使用冻结安全模型。"""
    # 1、None表示当前没有该建议；存在时必须是对应契约实例。
    handoff = state.get("handoff_suggestion")
    escalation = state.get("escalation_suggestion")
    if handoff is not None and not isinstance(handoff, HandoffSuggestion):
        raise OutputGateError("转人工建议没有通过安全契约校验")
    if escalation is not None and not isinstance(escalation, EscalationSuggestion):
        raise OutputGateError("升级建议没有通过安全契约校验")


def _terminal_texts(state: CustomerServiceState) -> Iterable[str]:
    """按终态类型枚举所有将向Java传递的用户可见文本。"""
    # 1、最终回答存在时必须为非空字符串。
    answer = state.get("final_answer")
    if answer is not None:
        if not isinstance(answer, str) or not answer.strip():
            raise OutputGateError("最终回答必须是非空文本")
        yield answer
    # 2、结构化建议只暴露原因和摘要，不检查枚举或代码字段文本。
    handoff = state.get("handoff_suggestion")
    if isinstance(handoff, HandoffSuggestion):
        yield handoff.reason
        yield handoff.summary
    escalation = state.get("escalation_suggestion")
    if isinstance(escalation, EscalationSuggestion):
        yield escalation.reason
        yield escalation.summary


def _order_reference_evidence(state: CustomerServiceState) -> set[str]:
    """收集Java上下文与安全工具结果中可用于定位订单的标量。"""
    # 1、Java上下文允许证明关联标识，但其中状态快照不能证明当前实时状态。
    values = {
        str(value).lower()
        for value in state["business_context"].model_dump(mode="python").values()
        if value is not None
    }
    values.update(_tool_fact_evidence(state))
    return values


def _tool_fact_evidence(state: CustomerServiceState) -> set[str]:
    """只收集本轮安全Java工具结果中的实时标量事实。"""
    # 1、SafeToolResult已限制字段名称和值类型，可直接形成大小写无关证据集合。
    values: set[str] = set()
    for result in state.get("tool_results", []):
        values.update(str(value).lower() for value in result.facts.values() if value is not None)
    return values


def _model_text_values(model: BaseModel) -> Iterable[str]:
    """递归枚举安全契约模型中的字符串值供敏感模式检查。"""
    # 1、使用JSON模式转换枚举和嵌套模型，再遍历字典、列表和标量。
    yield from _nested_text_values(model.model_dump(mode="json"))


def _nested_text_values(value: object) -> Iterable[str]:
    """递归枚举字符串、字典和序列中的全部文本值。"""
    # 1、使用显式栈避免深层结构递归调用占用Python调用栈。
    pending: list[object] = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            yield current
        elif isinstance(current, dict):
            pending.extend(current.values())
        elif isinstance(current, list | tuple):
            pending.extend(current)
