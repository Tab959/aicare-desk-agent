"""把图前安全判定转换为固定答复或主动转人工建议。"""

from aicare_agent_service.contracts.decisions import HandoffSuggestion
from aicare_agent_service.contracts.events import HandoffPriority
from aicare_agent_service.graph.state import CustomerServiceState
from aicare_agent_service.security.contracts import SafetyDisposition
from aicare_agent_service.security.policy import build_safety_response


def build_input_terminal(state: CustomerServiceState) -> dict[str, object]:
    """为阻断、空输入澄清或明确人工请求构造唯一终态。"""
    # 1、阻断与安全澄清使用策略原因对应的固定文本，不调用模型。
    assessment = state["input_safety_assessment"]
    if assessment.disposition in {SafetyDisposition.BLOCK, SafetyDisposition.CLARIFY}:
        return {
            "final_answer": build_safety_response(assessment),
            "handoff_suggestion": None,
            "escalation_suggestion": None,
        }
    # 2、只有明确用户请求人工时生成建议，Java决定是否真正切换会话状态。
    if assessment.disposition is SafetyDisposition.HUMAN_HANDOFF:
        return {
            "final_answer": None,
            "handoff_suggestion": HandoffSuggestion(
                reason="用户明确要求人工客服",
                priority=HandoffPriority.MEDIUM,
                summary="用户请求转接人工客服。",
            ),
            "escalation_suggestion": None,
        }
    # 3、允许分类的输入调用此节点属于拓扑错误，直接向生命周期层传播。
    raise ValueError("允许分类的输入不能生成安全终态")
