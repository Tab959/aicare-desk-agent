"""提供20个Java只读工具的最小成功参数与安全响应样例。"""

from typing import Final

from aicare_agent_service.tools.contracts import ToolName

GAME_SUMMARY: Final[dict[str, object]] = {
    "gameId": "game-001",
    "name": "示例游戏",
    "price": 99.0,
    "currency": "CNY",
    "purchaseMethods": ["CDK"],
    "available": True,
}

ORDER_SUMMARY: Final[dict[str, object]] = {
    "orderId": "order-001",
    "orderNo": "AICARE-001",
    "status": "PAID",
    "totalAmount": 99.0,
    "currency": "CNY",
    "createdAt": "2026-08-16T06:00:00Z",
    "expiresAt": None,
}


TOOL_SUCCESS_CASES: Final[dict[ToolName, tuple[str, dict[str, object], dict[str, object]]]] = {
    ToolName.SEARCH_GAMES: (
        "SearchGamesArguments",
        {"query": "动作游戏", "limit": 5},
        {"kind": "GAME_PAGE", "items": [GAME_SUMMARY], "nextCursor": None},
    ),
    ToolName.GET_GAME_FILTERS: (
        "GetGameFiltersArguments",
        {},
        {"kind": "GAME_FILTERS", "purchaseMethods": ["CDK"], "versions": ["标准版"]},
    ),
    ToolName.GET_GAME_DETAIL: (
        "GetGameDetailArguments",
        {"gameId": "game-001"},
        {
            "kind": "GAME_DETAIL",
            "game": GAME_SUMMARY,
            "description": "示例游戏说明",
            "deliveryMethods": ["CDK"],
            "refundRuleSummary": "符合平台退款规则时可申请退款",
        },
    ),
    ToolName.GET_RELATED_GAMES: (
        "GetRelatedGamesArguments",
        {"gameId": "game-001", "limit": 6},
        {"kind": "GAME_PAGE", "items": [GAME_SUMMARY], "nextCursor": None},
    ),
    ToolName.GET_RECOMMENDATIONS: (
        "GetRecommendationsArguments",
        {"limit": 5},
        {"kind": "GAME_PAGE", "items": [GAME_SUMMARY], "nextCursor": None},
    ),
    ToolName.GET_FEATURED_GAMES: (
        "GetFeaturedGamesArguments",
        {"limit": 5},
        {"kind": "GAME_PAGE", "items": [GAME_SUMMARY], "nextCursor": None},
    ),
    ToolName.GET_SALES_RANKING: (
        "GetSalesRankingArguments",
        {"limit": 5},
        {"kind": "GAME_PAGE", "items": [GAME_SUMMARY], "nextCursor": None},
    ),
    ToolName.GET_DEALS: (
        "GetDealsArguments",
        {"limit": 5},
        {"kind": "GAME_PAGE", "items": [GAME_SUMMARY], "nextCursor": None},
    ),
    ToolName.GET_FLASH_SALE_TIMELINE: (
        "GetFlashSaleTimelineArguments",
        {},
        {"kind": "FLASH_SALE_TIMELINE", "previous": None, "current": None, "next": None},
    ),
    ToolName.GET_FLASH_SALE_GAMES: (
        "GetFlashSaleGamesArguments",
        {"flashSaleId": "flash-001", "limit": 5},
        {"kind": "GAME_PAGE", "items": [GAME_SUMMARY], "nextCursor": None},
    ),
    ToolName.GET_CURRENT_USER: (
        "GetCurrentUserArguments",
        {},
        {"kind": "CURRENT_USER", "customerId": "user-customer-001", "displayName": "演示用户"},
    ),
    ToolName.GET_PROFILE: (
        "GetProfileArguments",
        {},
        {"kind": "PROFILE", "displayName": "演示用户", "registeredAt": "2026-08-16T06:00:00Z"},
    ),
    ToolName.LIST_FAVORITES: (
        "ListFavoritesArguments",
        {"limit": 5},
        {"kind": "GAME_PAGE", "items": [GAME_SUMMARY], "nextCursor": None},
    ),
    ToolName.GET_CART: (
        "GetCartArguments",
        {},
        {"kind": "CART", "items": [], "selectedTotal": 0.0, "currency": "CNY"},
    ),
    ToolName.PREVIEW_CHECKOUT: (
        "PreviewCheckoutArguments",
        {"source": "DIRECT", "items": [{"offerId": "offer-001", "quantity": 1}]},
        {
            "kind": "CHECKOUT_PREVIEW",
            "itemCount": 1,
            "totalAmount": 99.0,
            "currency": "CNY",
            "available": True,
        },
    ),
    ToolName.LIST_ORDERS: (
        "ListOrdersArguments",
        {"limit": 5},
        {"kind": "ORDER_PAGE", "items": [ORDER_SUMMARY], "nextCursor": None},
    ),
    ToolName.GET_ORDER_DETAIL: (
        "GetOrderDetailArguments",
        {"orderId": "order-001"},
        {
            "kind": "ORDER_DETAIL",
            "order": ORDER_SUMMARY,
            "deliveryTypes": ["CDK"],
            "entitlementIds": ["entitlement-001"],
        },
    ),
    ToolName.GET_WALLET: (
        "GetWalletArguments",
        {},
        {"kind": "WALLET", "availableBalance": 200.0, "currency": "CNY"},
    ),
    ToolName.LIST_WALLET_TRANSACTIONS: (
        "ListWalletTransactionsArguments",
        {"limit": 5},
        {"kind": "WALLET_TRANSACTION_PAGE", "items": [], "nextCursor": None},
    ),
    ToolName.INSPECT_ENTITLEMENT_STATUS: (
        "InspectEntitlementStatusArguments",
        {"entitlementId": "entitlement-001"},
        {
            "kind": "ENTITLEMENT_STATUS",
            "entitlementId": "entitlement-001",
            "entitlementType": "CDK",
            "deliveryStatus": "DELIVERED",
            "deliveredAt": "2026-08-16T06:00:00Z",
            "requiresManualHandling": False,
        },
    ),
}
