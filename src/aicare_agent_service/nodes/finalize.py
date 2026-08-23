"""确认根图恰有一个终态；事件持久化和Java业务状态仍由生命周期层负责。"""

from enum import StrEnum

from aicare_agent_service.contracts.decisions import (
    EscalationSuggestion,
    HandoffSuggestion,
)
from aicare_agent_service.graph.state import CustomerServiceState


class FinalStateValidationError(ValueError):
    """根图没有终态、存在冲突终态或终态类型非法。"""


class TerminalKind(StrEnum):
    """根图和生命周期层共享的三种合法终态类型。"""

    FINAL_ANSWER = "final_answer"
    HANDOFF = "handoff"
    ESCALATION = "escalation"


def finalize_node(state: CustomerServiceState) -> dict[str, object]:
    """确定性确认唯一终态并返回空更新，不调用模型或Java。"""
    # 1、复用生命周期事件封装前也会调用的严格终态选择器。
    validate_terminal_state(state)
    # 2、finalize不生成新内容、不写Java，也不改变已经校验的状态。
    return {}


def validate_terminal_state(state: CustomerServiceState) -> TerminalKind:
    """严格选择唯一终态，供根图finalize和生命周期事件封装共同复用。"""
    # 1、分别按严格类型判断最终消息、转人工建议和升级建议候选。
    answer = state.get("final_answer")
    handoff = state.get("handoff_suggestion")
    escalation = state.get("escalation_suggestion")
    candidates: list[TerminalKind] = []
    if isinstance(answer, str) and answer.strip():
        candidates.append(TerminalKind.FINAL_ANSWER)
    elif answer is not None:
        raise FinalStateValidationError("最终回答类型非法")
    if isinstance(handoff, HandoffSuggestion):
        candidates.append(TerminalKind.HANDOFF)
    elif handoff is not None:
        raise FinalStateValidationError("转人工建议类型非法")
    if isinstance(escalation, EscalationSuggestion):
        candidates.append(TerminalKind.ESCALATION)
    elif escalation is not None:
        raise FinalStateValidationError("升级建议类型非法")
    # 2、零个或多个候选都不能作为生命周期COMPLETED终态。
    if len(candidates) != 1:
        raise FinalStateValidationError("根图必须且只能生成一个终态")
    # 3、返回固定枚举，调用方不再各自重复推断终态类型。
    return candidates[0]
