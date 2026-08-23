"""定义 Java 随 Agent 请求传入的最小、非敏感业务上下文。

这里仅携带模型生成回复所需的业务标识和状态摘要，不包含 CDK、密码、下载凭证或完整订单数据。
字段值允许为 ``None``，但在线 JSON 中字段本身仍必须出现，以保持 Java/Python 契约稳定。
"""

# 复用统一的非空文本约束和 camelCase 线模型基类。
from aicare_agent_service.contracts.common import NonEmptyText, WireContractModel


class AgentBusinessContext(WireContractModel):
    """Java 提供给 Python 的受控业务上下文快照。"""

    # 用户当前问题的安全主题摘要；``| None`` 表示可以显式传 null。
    subject: NonEmptyText | None
    # 相关订单的内部 ID；没有关联订单时为 None。
    order_id: NonEmptyText | None
    # 面向用户展示的订单号；没有关联订单时为 None。
    order_no: NonEmptyText | None
    # Java 判断后的订单状态文本；Python 不据此直接修改订单。
    order_status: NonEmptyText | None
    # 相关数字权益 ID；没有权益时为 None。
    entitlement_id: NonEmptyText | None
    # 权益类型摘要，例如 CDK 或下载资源类型；不得放入实际密钥。
    entitlement_type: NonEmptyText | None
    # 权益交付状态摘要；真实交付内容仍由 Java 保管。
    entitlement_status: NonEmptyText | None
