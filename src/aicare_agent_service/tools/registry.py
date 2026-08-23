"""注册20个只读LangChain工具，并按专业Agent提供不可变最小权限能力包。"""

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from pydantic import BaseModel, ConfigDict, create_model

from aicare_agent_service.contracts.decisions import (
    SafeToolResult,
    ToolResultStatus,
)
from aicare_agent_service.graph.context import AgentRuntimeContext
from aicare_agent_service.tools.contracts import (
    TOOL_ARGUMENT_MODELS,
    CartData,
    CheckoutData,
    CurrentUserData,
    EntitlementStatusData,
    FlashSaleTimelineData,
    GameDetailData,
    GameFiltersData,
    GamePageData,
    OrderDetailData,
    OrderPageData,
    ProfileData,
    ToolName,
    WalletData,
    WalletTransactionPageData,
    validate_tool_arguments,
)
from aicare_agent_service.tools.java_client import JavaToolClientError
from aicare_agent_service.tools.sanitizer import sanitize_payload


class ToolDomain(StrEnum):
    """工具所属的稳定业务领域，用于能力隔离和低基数追踪。"""

    CATALOG = "CATALOG"
    PROMOTION = "PROMOTION"
    CUSTOMER = "CUSTOMER"
    ORDER = "ORDER"
    WALLET = "WALLET"
    ENTITLEMENT = "ENTITLEMENT"


class ToolRisk(StrEnum):
    """Task 6只允许注册无副作用的只读风险级别。"""

    READ_ONLY = "READ_ONLY"


class CapabilityPackage(StrEnum):
    """Task 7三个专业Agent可取得的最小工具集合。"""

    PRE_SALES = "PRE_SALES"
    ORDER_SUPPORT = "ORDER_SUPPORT"
    AFTER_SALES = "AFTER_SALES"


@dataclass(frozen=True, slots=True)
class ToolRegistration:
    """单个工具的固定名称、领域、风险、输入输出模型和LangChain实现。"""

    name: ToolName
    domain: ToolDomain
    risk: ToolRisk
    arguments_model: type[BaseModel]
    response_model: type[BaseModel]
    description: str
    tool: BaseTool


@dataclass(frozen=True, slots=True)
class _ToolSpec:
    """构建注册表前的不可变工具声明。"""

    name: ToolName
    domain: ToolDomain
    response_model: type[BaseModel]
    description: str


class _RuntimeArguments(BaseModel):
    """仅供LangChain注入ToolRuntime的内部参数基类，不进入模型可见Schema。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    runtime: ToolRuntime[Any]


_DESCRIPTIONS: dict[ToolName, str] = {
    ToolName.SEARCH_GAMES: "按关键词、价格、版本和购买方式搜索可售Steam游戏。",
    ToolName.GET_GAME_FILTERS: "读取当前游戏搜索可用的版本与购买方式筛选项。",
    ToolName.GET_GAME_DETAIL: "读取指定游戏的价格、购买方式、交付方式和退款规则摘要。",
    ToolName.GET_RELATED_GAMES: "读取与指定游戏相关的可售游戏。",
    ToolName.GET_RECOMMENDATIONS: "读取基于当前顾客行为生成的个性化游戏推荐。",
    ToolName.GET_FEATURED_GAMES: "读取平台人工精选游戏。",
    ToolName.GET_SALES_RANKING: "读取最近24小时游戏销量榜。",
    ToolName.GET_DEALS: "读取当前处于优惠状态的游戏。",
    ToolName.GET_FLASH_SALE_TIMELINE: "读取上一场、当前场和下一场秒杀时间摘要。",
    ToolName.GET_FLASH_SALE_GAMES: "读取指定秒杀场次中的安全商品列表。",
    ToolName.GET_CURRENT_USER: "读取当前活动会话顾客的最小身份摘要。",
    ToolName.GET_PROFILE: "读取当前顾客用于个性化称呼的最小资料。",
    ToolName.LIST_FAVORITES: "读取当前顾客收藏的游戏。",
    ToolName.GET_CART: "读取当前顾客购物车与已选商品合计。",
    ToolName.PREVIEW_CHECKOUT: "只读预览结算金额与库存，不创建订单。",
    ToolName.LIST_ORDERS: "按可选状态读取当前顾客订单列表。",
    ToolName.GET_ORDER_DETAIL: "读取当前顾客拥有的订单及安全权益摘要。",
    ToolName.GET_WALLET: "读取当前顾客钱包可用余额。",
    ToolName.LIST_WALLET_TRANSACTIONS: "读取当前顾客钱包流水。",
    ToolName.INSPECT_ENTITLEMENT_STATUS: "只读取权益类型和交付状态，不读取任何交付凭据。",
}

_RESPONSE_MODELS: dict[ToolName, type[BaseModel]] = {
    ToolName.SEARCH_GAMES: GamePageData,
    ToolName.GET_GAME_FILTERS: GameFiltersData,
    ToolName.GET_GAME_DETAIL: GameDetailData,
    ToolName.GET_RELATED_GAMES: GamePageData,
    ToolName.GET_RECOMMENDATIONS: GamePageData,
    ToolName.GET_FEATURED_GAMES: GamePageData,
    ToolName.GET_SALES_RANKING: GamePageData,
    ToolName.GET_DEALS: GamePageData,
    ToolName.GET_FLASH_SALE_TIMELINE: FlashSaleTimelineData,
    ToolName.GET_FLASH_SALE_GAMES: GamePageData,
    ToolName.GET_CURRENT_USER: CurrentUserData,
    ToolName.GET_PROFILE: ProfileData,
    ToolName.LIST_FAVORITES: GamePageData,
    ToolName.GET_CART: CartData,
    ToolName.PREVIEW_CHECKOUT: CheckoutData,
    ToolName.LIST_ORDERS: OrderPageData,
    ToolName.GET_ORDER_DETAIL: OrderDetailData,
    ToolName.GET_WALLET: WalletData,
    ToolName.LIST_WALLET_TRANSACTIONS: WalletTransactionPageData,
    ToolName.INSPECT_ENTITLEMENT_STATUS: EntitlementStatusData,
}

_PROMOTION_TOOLS = frozenset(
    {
        ToolName.GET_DEALS,
        ToolName.GET_FLASH_SALE_TIMELINE,
        ToolName.GET_FLASH_SALE_GAMES,
    }
)
_CUSTOMER_TOOLS = frozenset(
    {ToolName.GET_CURRENT_USER, ToolName.GET_PROFILE, ToolName.LIST_FAVORITES}
)
_ORDER_TOOLS = frozenset(
    {ToolName.GET_CART, ToolName.PREVIEW_CHECKOUT, ToolName.LIST_ORDERS, ToolName.GET_ORDER_DETAIL}
)


def _domain(tool_name: ToolName) -> ToolDomain:
    """根据固定工具名返回唯一业务领域。"""
    if tool_name in _PROMOTION_TOOLS:
        return ToolDomain.PROMOTION
    if tool_name in _CUSTOMER_TOOLS:
        return ToolDomain.CUSTOMER
    if tool_name in _ORDER_TOOLS:
        return ToolDomain.ORDER
    if tool_name in {ToolName.GET_WALLET, ToolName.LIST_WALLET_TRANSACTIONS}:
        return ToolDomain.WALLET
    if tool_name is ToolName.INSPECT_ENTITLEMENT_STATUS:
        return ToolDomain.ENTITLEMENT
    return ToolDomain.CATALOG


def _build_tool(spec: _ToolSpec) -> BaseTool:
    """把单个固定声明包装成Runtime身份不可见的异步LangChain工具。"""

    async def invoke(
        runtime: ToolRuntime[AgentRuntimeContext],
        **model_arguments: Any,
    ) -> tuple[str, SafeToolResult]:
        """执行严格参数校验、Java调用和安全ToolMessage投影。"""
        # 1、Runtime由LangChain注入且不进入模型Schema；工具调用ID必须存在。
        if not runtime.tool_call_id:
            raise ToolException("工具调用缺少关联ID")
        arguments = validate_tool_arguments(spec.name, model_arguments)
        # 2、身份、Java客户端和deadline只从不可变运行时上下文取得。
        try:
            result = await runtime.context.java_client.execute_tool(
                identity=runtime.context.expected_identity,
                tool_name=spec.name,
                tool_call_id=runtime.tool_call_id,
                arguments=arguments,
                deadline=runtime.context.request_deadline,
            )
        except JavaToolClientError as exc:
            status = _error_status(exc.code)
            artifact = SafeToolResult(
                tool_name=spec.name.value,
                status=status,
                summary=exc.safe_message,
                facts={"error_code": exc.code},
            )
            return sanitize_payload({"code": exc.code, "message": exc.safe_message}), artifact
        # 3、模型只看到经递归门禁的紧凑JSON；checkpoint只保存标量证据。
        return sanitize_payload(result.data), _project_evidence(spec.name, result)

    # 4、LangChain先向内部参数模型注入runtime，再用tool_call_schema自动排除该字段。
    arguments_model = TOOL_ARGUMENT_MODELS[spec.name]
    runtime_arguments_model = create_model(
        f"{arguments_model.__name__}WithRuntime",
        __base__=(arguments_model, _RuntimeArguments),
    )
    return StructuredTool.from_function(
        coroutine=invoke,
        name=spec.name.value,
        description=spec.description,
        args_schema=runtime_arguments_model,
        response_format="content_and_artifact",
        metadata={
            "tool_name": spec.name.value,
            "domain": spec.domain.value,
            "risk": ToolRisk.READ_ONLY.value,
        },
    )


def _project_evidence(tool_name: ToolName, result: Any) -> SafeToolResult:
    """把完整安全业务结果缩减为可checkpoint的标量证据。"""
    # 1、固定保存工具种类与观测时间，不保存完整Java响应。
    data = result.data.model_dump(mode="json", by_alias=False)
    facts: dict[str, str | int | float | bool | None] = {
        "kind": result.data.kind,
        "observed_at": result.observed_at.isoformat(),
    }
    # 2、列表只保存数量；实体ID和状态只从明确白名单字段投影。
    items = data.get("items")
    if isinstance(items, list):
        facts["item_count"] = len(items)
    for field in (
        "game_id",
        "customer_id",
        "order_id",
        "entitlement_id",
        "status",
        "delivery_status",
        "requires_manual_handling",
        "item_count",
        "quantity",
        "available",
    ):
        value = data.get(field)
        if field in data and (isinstance(value, (str, int, float, bool)) or value is None):
            facts[field] = value
    # 嵌套商品或订单只投影其ID与状态，不保存嵌套对象。
    for container_name in ("game", "order"):
        container = data.get(container_name)
        if not isinstance(container, dict):
            continue
        for field in ("game_id", "order_id", "status", "available"):
            value = container.get(field)
            if field in container and (isinstance(value, (str, int, float, bool)) or value is None):
                facts[field] = value
    # 多权益与交付类型只保存数量，不保存ID列表或业务正文。
    for field, fact_name in (
        ("entitlement_ids", "entitlement_count"),
        ("delivery_types", "delivery_type_count"),
    ):
        values = data.get(field)
        if isinstance(values, list):
            facts[fact_name] = len(values)
    # 3、摘要仅描述成功种类，不复制模型可见正文。
    return SafeToolResult(
        tool_name=tool_name.value,
        status=ToolResultStatus.SUCCESS,
        summary=f"{result.data.kind}查询成功",
        facts=facts,
    )


def _error_status(code: str) -> ToolResultStatus:
    """把客户端稳定错误码映射为有限工具结果状态。"""
    if code == "TOOL_NOT_FOUND":
        return ToolResultStatus.NOT_FOUND
    if code == "TOOL_ACCESS_DENIED":
        return ToolResultStatus.REJECTED
    return ToolResultStatus.UNAVAILABLE


def _build_registry(specs: tuple[_ToolSpec, ...]) -> MappingProxyType:
    """构建不可变注册表，并拒绝重名、缺参数模型或缺输出模型。"""
    # 1、逐项验证固定依赖存在，避免启动后才发现工具半注册。
    registrations: dict[ToolName, ToolRegistration] = {}
    for spec in specs:
        if spec.name in registrations:
            raise ValueError(f"工具重复注册：{spec.name.value}")
        if spec.name not in TOOL_ARGUMENT_MODELS or spec.response_model is None:
            raise ValueError(f"工具缺少实现或契约：{spec.name.value}")
        tool = _build_tool(spec)
        registrations[spec.name] = ToolRegistration(
            name=spec.name,
            domain=spec.domain,
            risk=ToolRisk.READ_ONLY,
            arguments_model=TOOL_ARGUMENT_MODELS[spec.name],
            response_model=spec.response_model,
            description=spec.description,
            tool=tool,
        )
    # 2、固定首批必须覆盖全部20个ToolName，不允许静默漏注册。
    if frozenset(registrations) != frozenset(ToolName):
        raise ValueError("只读工具注册表必须完整覆盖20个ToolName")
    return MappingProxyType(registrations)


_SPECS = tuple(
    _ToolSpec(
        name=name,
        domain=_domain(name),
        response_model=_RESPONSE_MODELS[name],
        description=_DESCRIPTIONS[name],
    )
    for name in ToolName
)

READ_ONLY_TOOL_REGISTRY = _build_registry(_SPECS)

_CAPABILITIES = MappingProxyType(
    {
        CapabilityPackage.PRE_SALES: frozenset(tuple(ToolName)[:10]),
        CapabilityPackage.ORDER_SUPPORT: frozenset(
            {
                ToolName.LIST_ORDERS,
                ToolName.GET_ORDER_DETAIL,
                ToolName.GET_WALLET,
                ToolName.LIST_WALLET_TRANSACTIONS,
                ToolName.INSPECT_ENTITLEMENT_STATUS,
            }
        ),
        CapabilityPackage.AFTER_SALES: frozenset(
            {
                ToolName.LIST_ORDERS,
                ToolName.GET_ORDER_DETAIL,
                ToolName.INSPECT_ENTITLEMENT_STATUS,
            }
        ),
    }
)


def tools_for_capability(capability: CapabilityPackage) -> tuple[BaseTool, ...]:
    """按ToolName声明顺序返回某专业Agent的不可变最小工具元组。"""
    # 1、能力包只保存固定枚举集合，调用方无法在原映射上增删工具。
    allowed = _CAPABILITIES[capability]
    # 2、按枚举顺序构建新tuple，使Prompt工具顺序稳定且不可修改。
    return tuple(READ_ONLY_TOOL_REGISTRY[name].tool for name in ToolName if name in allowed)
