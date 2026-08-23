"""把安全判定和结构化分类转换为根图中存在的固定节点代码。"""

from enum import StrEnum

from aicare_agent_service.contracts.decisions import RouteCode
from aicare_agent_service.graph.state import CustomerServiceState
from aicare_agent_service.security.contracts import SafetyDisposition


class RootRoute(StrEnum):
    """根图条件边允许返回的固定节点代码。"""

    SECURITY_BLOCK = "security_block"
    HUMAN_HANDOFF = "human_handoff"
    CLASSIFY = "classify"
    CLASSIFICATION_FALLBACK = "classification_fallback"
    CLARIFY = "clarify"
    PRE_SALES = "pre_sales"
    ORDER_SUPPORT = "order_support"
    AFTER_SALES = "after_sales"
    KNOWLEDGE = "knowledge"
    UNSUPPORTED = "unsupported"


class RootRoutingError(ValueError):
    """根图缺少合法决策或收到非法阈值。"""


_ROUTE_CODE_TARGETS: dict[RouteCode, RootRoute] = {
    RouteCode.HUMAN_HANDOFF: RootRoute.HUMAN_HANDOFF,
    RouteCode.AFTER_SALES: RootRoute.AFTER_SALES,
    RouteCode.ORDER_SUPPORT: RootRoute.ORDER_SUPPORT,
    RouteCode.PRE_SALES: RootRoute.PRE_SALES,
    RouteCode.KNOWLEDGE: RootRoute.KNOWLEDGE,
    RouteCode.UNSUPPORTED: RootRoute.UNSUPPORTED,
}


def select_entry_route(state: CustomerServiceState) -> RootRoute:
    """按阻断、主动人工、澄清、允许分类的顺序选择入口路径。"""
    # 1、安全处置已经在状态创建前确定，根图不再调用模型重判。
    assessment = state["input_safety_assessment"]
    if assessment.disposition is SafetyDisposition.BLOCK:
        return RootRoute.SECURITY_BLOCK
    if assessment.disposition is SafetyDisposition.HUMAN_HANDOFF:
        return RootRoute.HUMAN_HANDOFF
    if assessment.disposition is SafetyDisposition.CLARIFY:
        return RootRoute.CLARIFY
    if assessment.disposition is SafetyDisposition.ALLOW:
        return RootRoute.CLASSIFY
    raise RootRoutingError("不支持的输入安全处置")


def select_classification_route(
    state: CustomerServiceState,
    direct_confidence: float,
    clarify_confidence: float,
) -> RootRoute:
    """用配置阈值和代码路由表决定分类后的唯一下一节点。"""
    # 1、阈值必须形成互斥区间，禁止调用方绕过Settings关系校验。
    if not 0 <= clarify_confidence < direct_confidence <= 1:
        raise RootRoutingError("分类置信度阈值非法")
    # 2、可预期分类失败优先进入固定兜底。
    if state.get("classification_failure") is not None:
        return RootRoute.CLASSIFICATION_FALLBACK
    decision = state.get("route_decision")
    if decision is None:
        raise RootRoutingError("缺少结构化路由决定")
    # 3、低置信度固定兜底，中间区间固定澄清。
    if decision.confidence < clarify_confidence:
        return RootRoute.CLASSIFICATION_FALLBACK
    if decision.confidence < direct_confidence:
        return RootRoute.CLARIFY
    # 4、达到直接路由阈值后只查代码映射，不接受模型提供节点名。
    try:
        return _ROUTE_CODE_TARGETS[decision.route_code]
    except KeyError as exc:
        raise RootRoutingError("路由代码没有固定根图目标") from exc
