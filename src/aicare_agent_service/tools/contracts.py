"""定义 Python 与 Java 内部只读工具 API 的严格版本化契约。

本文件只描述允许跨服务传输的身份、工具参数和最小安全业务数据，不包含网络调用、
工具循环或业务规则。所有模型拒绝未知字段，防止 Java 响应或模型参数静默扩张。
"""

import json
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic.alias_generators import to_camel

TOOL_CONTRACT_VERSION: Literal["1.0"] = "1.0"
MAX_PAGE_SIZE = 20
MAX_CURSOR_LENGTH = 512
MAX_TOOL_PAYLOAD_BYTES = 65_536
MAX_TOOL_PAYLOAD_DEPTH = 8

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240)]
Cursor = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_CURSOR_LENGTH)
]
Money = Annotated[Decimal, Field(ge=0, max_digits=12, decimal_places=2)]
SignedMoney = Annotated[Decimal, Field(max_digits=12, decimal_places=2)]
PageSize = Annotated[int, Field(strict=True, ge=1, le=MAX_PAGE_SIZE)]


class ToolWireModel(BaseModel):
    """所有内部工具 JSON 模型的公共严格基类。"""

    model_config = ConfigDict(
        alias_generator=to_camel,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_by_alias=True,
        validate_by_name=True,
    )


class ToolName(StrEnum):
    """首批允许跨服务调用的20个只读工具名。"""

    # 搜索游戏工具
    SEARCH_GAMES = "search_games"
    # 获取游戏筛选项工具
    GET_GAME_FILTERS = "get_game_filters"
    # 获取游戏详情工具
    GET_GAME_DETAIL = "get_game_detail"
    # 获取相关游戏工具
    GET_RELATED_GAMES = "get_related_games"
    # 获取推荐游戏工具
    GET_RECOMMENDATIONS = "get_recommendations"
    # 获取特色游戏工具
    GET_FEATURED_GAMES = "get_featured_games"
    # 获取销售排名工具
    GET_SALES_RANKING = "get_sales_ranking"
    # 获取优惠游戏工具
    GET_DEALS = "get_deals"
    # 获取秒杀时间线工具
    GET_FLASH_SALE_TIMELINE = "get_flash_sale_timeline"
    # 获取秒杀游戏工具
    GET_FLASH_SALE_GAMES = "get_flash_sale_games"
    # 获取当前用户工具
    GET_CURRENT_USER = "get_current_user"
    # 获取用户个人信息工具
    GET_PROFILE = "get_profile"
    # 获取用户收藏游戏列表工具
    LIST_FAVORITES = "list_favorites"
    # 获取用户购物车工具
    GET_CART = "get_cart"
    # 预览结账工具
    PREVIEW_CHECKOUT = "preview_checkout"
    # 获取用户订单列表工具
    LIST_ORDERS = "list_orders"
    # 获取用户订单详情工具
    GET_ORDER_DETAIL = "get_order_detail"
    # 获取用户钱包工具
    GET_WALLET = "get_wallet"
    # 获取用户钱包交易记录工具
    LIST_WALLET_TRANSACTIONS = "list_wallet_transactions"
    # 检查用户权益状态工具
    INSPECT_ENTITLEMENT_STATUS = "inspect_entitlement_status"


class AgentToolIdentity(ToolWireModel):
    """Java拥有的完整活动run身份，Python只能从运行时上下文复制。"""

    tenant_id: Identifier
    customer_id: Identifier
    conversation_id: Identifier
    run_id: Identifier
    trigger_message_id: Identifier
    trigger_sequence: Annotated[int, Field(strict=True, ge=1)]


class EmptyArguments(ToolWireModel):
    """没有模型可见参数的工具输入；任何额外字段都会被拒绝。"""


class CursorArguments(ToolWireModel):
    """使用Java返回的不透明游标读取有界列表。"""

    limit: PageSize = 10
    cursor: Cursor | None = None


class SearchGamesArguments(CursorArguments):
    """按自然语言和安全商品筛选条件搜索游戏。"""

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
    max_price: Money | None = None
    purchase_methods: Annotated[tuple[ShortText, ...], Field(max_length=8)] = ()


class GetGameFiltersArguments(EmptyArguments):
    """读取Java提供的可用游戏筛选项。"""


class GameIdArguments(ToolWireModel):
    """按单一游戏ID查询数据。"""

    game_id: Identifier


class GetGameDetailArguments(GameIdArguments):
    """读取单个游戏的安全详情。"""


class GetRelatedGamesArguments(GameIdArguments):
    """读取某个游戏的相关推荐。"""

    limit: PageSize = 6


class GetRecommendationsArguments(CursorArguments):
    """读取当前顾客的个性化推荐。"""


class GetFeaturedGamesArguments(CursorArguments):
    """读取平台人工精选游戏。"""


class GetSalesRankingArguments(CursorArguments):
    """读取24小时销量榜。"""


class GetDealsArguments(CursorArguments):
    """读取当前优惠游戏。"""


class GetFlashSaleTimelineArguments(EmptyArguments):
    """读取上一场、当前场和下一场秒杀摘要。"""


class GetFlashSaleGamesArguments(CursorArguments):
    """读取指定秒杀场次的安全商品列表。"""

    flash_sale_id: Identifier


class GetCurrentUserArguments(EmptyArguments):
    """读取当前活动run对应的顾客身份摘要。"""


class GetProfileArguments(EmptyArguments):
    """读取当前顾客的最小个性化资料。"""


class ListFavoritesArguments(CursorArguments):
    """读取当前顾客收藏的游戏。"""


class GetCartArguments(EmptyArguments):
    """读取当前顾客购物车。"""


class CheckoutSource(StrEnum):
    """Java现有结算预览支持的三种来源。"""

    DIRECT = "DIRECT"
    CART = "CART"
    FLASH_SALE = "FLASH_SALE"


class CheckoutItem(ToolWireModel):
    """直接结算中的单个报价与数量。"""

    offer_id: Identifier
    quantity: Annotated[int, Field(strict=True, ge=1, le=20)]


class PreviewCheckoutArguments(ToolWireModel):
    """只计算价格和库存的结算预览参数，不创建订单。"""

    source: CheckoutSource
    items: Annotated[tuple[CheckoutItem, ...], Field(max_length=20)] = ()
    cart_item_ids: Annotated[tuple[Identifier, ...], Field(max_length=20)] = ()
    flash_sale_item_id: Identifier | None = None
    quantity: Annotated[int, Field(strict=True, ge=1, le=20)] | None = None

    @model_validator(mode="after")
    def require_source_specific_fields(self) -> "PreviewCheckoutArguments":
        """保证一种结算来源只能携带与它匹配的一组字段。"""
        # 1、DIRECT只允许非空items。
        if self.source is CheckoutSource.DIRECT:
            valid = (
                bool(self.items)
                and not self.cart_item_ids
                and self.flash_sale_item_id is None
                and self.quantity is None
            )
        # 2、CART只允许非空cartItemIds。
        elif self.source is CheckoutSource.CART:
            valid = (
                bool(self.cart_item_ids)
                and not self.items
                and self.flash_sale_item_id is None
                and self.quantity is None
            )
        # 3、FLASH_SALE必须同时提供场次商品ID和数量。
        else:
            valid = (
                not self.items
                and not self.cart_item_ids
                and self.flash_sale_item_id is not None
                and self.quantity is not None
            )
        if not valid:
            raise ValueError("结算参数必须与source匹配")
        return self


class ListOrdersArguments(CursorArguments):
    """按可选状态读取当前顾客订单。"""

    status: ShortText | None = None


class GetOrderDetailArguments(ToolWireModel):
    """读取当前顾客拥有的单个订单详情。"""

    order_id: Identifier


class GetWalletArguments(EmptyArguments):
    """读取当前顾客钱包余额。"""


class ListWalletTransactionsArguments(CursorArguments):
    """读取当前顾客钱包流水。"""


class InspectEntitlementStatusArguments(ToolWireModel):
    """只读取权益状态，绝不读取交付凭据。"""

    entitlement_id: Identifier


ToolArguments: TypeAlias = (
    SearchGamesArguments
    | GetGameFiltersArguments
    | GetGameDetailArguments
    | GetRelatedGamesArguments
    | GetRecommendationsArguments
    | GetFeaturedGamesArguments
    | GetSalesRankingArguments
    | GetDealsArguments
    | GetFlashSaleTimelineArguments
    | GetFlashSaleGamesArguments
    | GetCurrentUserArguments
    | GetProfileArguments
    | ListFavoritesArguments
    | GetCartArguments
    | PreviewCheckoutArguments
    | ListOrdersArguments
    | GetOrderDetailArguments
    | GetWalletArguments
    | ListWalletTransactionsArguments
    | InspectEntitlementStatusArguments
)

TOOL_ARGUMENT_MODELS: dict[ToolName, type[ToolWireModel]] = {
    ToolName.SEARCH_GAMES: SearchGamesArguments,
    ToolName.GET_GAME_FILTERS: GetGameFiltersArguments,
    ToolName.GET_GAME_DETAIL: GetGameDetailArguments,
    ToolName.GET_RELATED_GAMES: GetRelatedGamesArguments,
    ToolName.GET_RECOMMENDATIONS: GetRecommendationsArguments,
    ToolName.GET_FEATURED_GAMES: GetFeaturedGamesArguments,
    ToolName.GET_SALES_RANKING: GetSalesRankingArguments,
    ToolName.GET_DEALS: GetDealsArguments,
    ToolName.GET_FLASH_SALE_TIMELINE: GetFlashSaleTimelineArguments,
    ToolName.GET_FLASH_SALE_GAMES: GetFlashSaleGamesArguments,
    ToolName.GET_CURRENT_USER: GetCurrentUserArguments,
    ToolName.GET_PROFILE: GetProfileArguments,
    ToolName.LIST_FAVORITES: ListFavoritesArguments,
    ToolName.GET_CART: GetCartArguments,
    ToolName.PREVIEW_CHECKOUT: PreviewCheckoutArguments,
    ToolName.LIST_ORDERS: ListOrdersArguments,
    ToolName.GET_ORDER_DETAIL: GetOrderDetailArguments,
    ToolName.GET_WALLET: GetWalletArguments,
    ToolName.LIST_WALLET_TRANSACTIONS: ListWalletTransactionsArguments,
    ToolName.INSPECT_ENTITLEMENT_STATUS: InspectEntitlementStatusArguments,
}


def validate_tool_arguments(tool_name: ToolName, payload: object) -> ToolArguments:
    """按固定工具名选择唯一参数模型并执行严格校验。"""
    # 1、在模型解析前限制原始参数的总大小和嵌套深度。
    validate_tool_payload_limits(payload)
    # 2、工具名只能来自枚举，因此不会被转换为任意类或URL。
    model_type = TOOL_ARGUMENT_MODELS[tool_name]
    # 3、各模型拒绝未知字段，身份和内部连接参数无法夹带。
    return model_type.model_validate(payload)  # type: ignore[return-value]


def validate_tool_payload_limits(payload: object) -> None:
    """拒绝超过跨服务上限的JSON大小和嵌套深度。"""

    # 1、递归计算容器深度，避免畸形响应消耗过多解析资源。
    def depth(value: object, current: int = 0) -> int:
        if current > MAX_TOOL_PAYLOAD_DEPTH:
            return current
        if isinstance(value, dict):
            return max((depth(item, current + 1) for item in value.values()), default=current)
        if isinstance(value, (list, tuple)):
            return max((depth(item, current + 1) for item in value), default=current)
        return current

    if depth(payload) > MAX_TOOL_PAYLOAD_DEPTH:
        raise ValueError("工具载荷嵌套深度超过安全上限")
    # 2、使用稳定紧凑JSON计算UTF-8字节数，超限时不保留原始正文。
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    if len(encoded) > MAX_TOOL_PAYLOAD_BYTES:
        raise ValueError("工具载荷大小超过安全上限")


class AgentToolInvokeRequest(ToolWireModel):
    """Python发给Java内部工具API的统一调用信封。"""

    contract_version: Literal["1.0"]
    tool_call_id: Identifier
    identity: AgentToolIdentity
    arguments: dict[str, object]


class GameSummary(ToolWireModel):
    """允许模型看到的最小游戏摘要。"""

    game_id: Identifier
    name: ShortText
    price: Money
    currency: Literal["CNY"]
    purchase_methods: Annotated[tuple[ShortText, ...], Field(max_length=8)]
    available: bool


class GamePageData(ToolWireModel):
    """游戏列表及Java签发的下一页游标。"""

    kind: Literal["GAME_PAGE"]
    items: Annotated[tuple[GameSummary, ...], Field(max_length=MAX_PAGE_SIZE)]
    next_cursor: Cursor | None


class GameFiltersData(ToolWireModel):
    """Java当前允许使用的商品筛选值。"""

    kind: Literal["GAME_FILTERS"]
    purchase_methods: Annotated[tuple[ShortText, ...], Field(max_length=20)]
    versions: Annotated[tuple[ShortText, ...], Field(max_length=50)]


class GameDetailData(ToolWireModel):
    """游戏详情、报价和退款规则的安全摘要。"""

    kind: Literal["GAME_DETAIL"]
    game: GameSummary
    description: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)
    ]
    delivery_methods: Annotated[tuple[ShortText, ...], Field(max_length=8)]
    refund_rule_summary: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
    ]


class FlashSaleSession(ToolWireModel):
    """单个秒杀场次的时间与状态摘要。"""

    flash_sale_id: Identifier
    status: ShortText
    starts_at: datetime
    ends_at: datetime


class FlashSaleTimelineData(ToolWireModel):
    """秒杀时间轴，缺失的上一场或下一场使用null。"""

    kind: Literal["FLASH_SALE_TIMELINE"]
    previous: FlashSaleSession | None
    current: FlashSaleSession | None
    next: FlashSaleSession | None


class CurrentUserData(ToolWireModel):
    """当前顾客的非敏感身份摘要。"""

    kind: Literal["CURRENT_USER"]
    customer_id: Identifier
    display_name: ShortText


class ProfileData(ToolWireModel):
    """个性化称呼所需的最小资料。"""

    kind: Literal["PROFILE"]
    display_name: ShortText
    registered_at: datetime


class CartItemData(ToolWireModel):
    """购物车中的单个安全商品项。"""

    cart_item_id: Identifier
    offer_id: Identifier
    game_name: ShortText
    quantity: Annotated[int, Field(strict=True, ge=1, le=20)]
    unit_price: Money
    selected: bool
    available: bool


class CartData(ToolWireModel):
    """购物车明细和已选商品合计。"""

    kind: Literal["CART"]
    items: Annotated[tuple[CartItemData, ...], Field(max_length=50)]
    selected_total: Money
    currency: Literal["CNY"]


class CheckoutData(ToolWireModel):
    """无副作用结算预览的价格和库存结论。"""

    kind: Literal["CHECKOUT_PREVIEW"]
    item_count: Annotated[int, Field(strict=True, ge=1, le=50)]
    total_amount: Money
    currency: Literal["CNY"]
    available: bool


class OrderSummary(ToolWireModel):
    """订单列表和详情共享的安全摘要。"""

    order_id: Identifier
    order_no: Identifier
    status: ShortText
    total_amount: Money
    currency: Literal["CNY"]
    created_at: datetime
    expires_at: datetime | None


class OrderPageData(ToolWireModel):
    """当前顾客订单列表及不透明游标。"""

    kind: Literal["ORDER_PAGE"]
    items: Annotated[tuple[OrderSummary, ...], Field(max_length=MAX_PAGE_SIZE)]
    next_cursor: Cursor | None


class OrderDetailData(ToolWireModel):
    """单个订单及其交付类型的安全详情。"""

    kind: Literal["ORDER_DETAIL"]
    order: OrderSummary
    delivery_types: Annotated[tuple[ShortText, ...], Field(max_length=20)]
    entitlement_ids: Annotated[tuple[Identifier, ...], Field(max_length=20)]


class WalletData(ToolWireModel):
    """钱包可用余额摘要。"""

    kind: Literal["WALLET"]
    available_balance: Money
    currency: Literal["CNY"]


class WalletTransaction(ToolWireModel):
    """单笔余额变动的安全摘要。"""

    transaction_id: Identifier
    transaction_type: ShortText
    amount: SignedMoney
    balance_after: Money
    occurred_at: datetime
    order_id: Identifier | None


class WalletTransactionPageData(ToolWireModel):
    """钱包流水列表及不透明游标。"""

    kind: Literal["WALLET_TRANSACTION_PAGE"]
    items: Annotated[tuple[WalletTransaction, ...], Field(max_length=MAX_PAGE_SIZE)]
    next_cursor: Cursor | None


class EntitlementStatusData(ToolWireModel):
    """不包含任何交付内容的权益状态投影。"""

    kind: Literal["ENTITLEMENT_STATUS"]
    entitlement_id: Identifier
    entitlement_type: ShortText
    delivery_status: ShortText
    delivered_at: datetime | None
    requires_manual_handling: bool


ToolInvocationData: TypeAlias = Annotated[
    GamePageData
    | GameFiltersData
    | GameDetailData
    | FlashSaleTimelineData
    | CurrentUserData
    | ProfileData
    | CartData
    | CheckoutData
    | OrderPageData
    | OrderDetailData
    | WalletData
    | WalletTransactionPageData
    | EntitlementStatusData,
    Field(discriminator="kind"),
]

_EXPECTED_DATA_KINDS: dict[ToolName, frozenset[str]] = {
    ToolName.SEARCH_GAMES: frozenset({"GAME_PAGE"}),
    ToolName.GET_GAME_FILTERS: frozenset({"GAME_FILTERS"}),
    ToolName.GET_GAME_DETAIL: frozenset({"GAME_DETAIL"}),
    ToolName.GET_RELATED_GAMES: frozenset({"GAME_PAGE"}),
    ToolName.GET_RECOMMENDATIONS: frozenset({"GAME_PAGE"}),
    ToolName.GET_FEATURED_GAMES: frozenset({"GAME_PAGE"}),
    ToolName.GET_SALES_RANKING: frozenset({"GAME_PAGE"}),
    ToolName.GET_DEALS: frozenset({"GAME_PAGE"}),
    ToolName.GET_FLASH_SALE_TIMELINE: frozenset({"FLASH_SALE_TIMELINE"}),
    ToolName.GET_FLASH_SALE_GAMES: frozenset({"GAME_PAGE"}),
    ToolName.GET_CURRENT_USER: frozenset({"CURRENT_USER"}),
    ToolName.GET_PROFILE: frozenset({"PROFILE"}),
    ToolName.LIST_FAVORITES: frozenset({"GAME_PAGE"}),
    ToolName.GET_CART: frozenset({"CART"}),
    ToolName.PREVIEW_CHECKOUT: frozenset({"CHECKOUT_PREVIEW"}),
    ToolName.LIST_ORDERS: frozenset({"ORDER_PAGE"}),
    ToolName.GET_ORDER_DETAIL: frozenset({"ORDER_DETAIL"}),
    ToolName.GET_WALLET: frozenset({"WALLET"}),
    ToolName.LIST_WALLET_TRANSACTIONS: frozenset({"WALLET_TRANSACTION_PAGE"}),
    ToolName.INSPECT_ENTITLEMENT_STATUS: frozenset({"ENTITLEMENT_STATUS"}),
}


class AgentToolInvokeResponse(ToolWireModel):
    """Java返回的成功信封，业务数据必须与工具名匹配。"""

    contract_version: Literal["1.0"]
    request_id: Identifier
    tool_call_id: Identifier
    tool_name: ToolName
    status: Literal["SUCCESS"]
    observed_at: datetime
    data: ToolInvocationData

    @model_validator(mode="after")
    def require_matching_data_kind(self) -> "AgentToolInvokeResponse":
        """阻止Java把另一工具的响应结构装入当前工具信封。"""
        # 1、根据固定工具枚举查找允许的数据种类。
        allowed_kinds = _EXPECTED_DATA_KINDS[self.tool_name]
        # 2、种类不匹配表示跨语言协议漂移，立即拒绝整份响应。
        if self.data.kind not in allowed_kinds:
            raise ValueError("工具响应data.kind与toolName不匹配")
        # 3、严格模型通过后仍限制完整线响应的深度和UTF-8字节数。
        validate_tool_payload_limits(self.model_dump(mode="json", by_alias=True))
        return self


ToolInvocationResult = AgentToolInvokeResponse
