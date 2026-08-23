"""定义 Redis run ledger 的状态、记录、开始判定与稳定异常。

这些Pydantic模型只允许保存身份、摘要、租约和时间等安全元数据，不保存用户正文、
AI回答正文或工具响应。Redis run ledger用于幂等和单飞，不等同于LangGraph checkpoint。
"""

# datetime 表示带时区的运行时间点。
from datetime import datetime

# StrEnum 成员同时具有枚举约束和字符串值，适合写入Redis。
from enum import StrEnum

# Annotated附加字符串约束；ClassVar声明类级常量而非Pydantic字段。
from typing import Annotated, ClassVar

# BaseModel负责运行时校验；SecretStr隐藏租约；StringConstraints定义正则；model_validator做跨字段校验。
from pydantic import BaseModel, ConfigDict, SecretStr, StringConstraints, model_validator

# 复用契约层的非空文本和正整数序号约束。
from aicare_agent_service.contracts.common import NonEmptyText, PositiveSequence

# 64位小写十六进制字符串类型，用于SHA-256请求、租约和终态摘要。
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
# 稳定错误码必须大写字母开头，之后只允许大写字母、数字和下划线。
StableErrorCode = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]*$")]


class RunStatus(StrEnum):
    """Redis run ledger允许持久化的固定状态。"""

    # 当前执行者持有或曾持有租约，图尚未提交终态。
    RUNNING = "RUNNING"
    # 图已产生合法终态且checkpoint摘要已原子写入ledger。
    COMPLETED = "COMPLETED"
    # 模型、图或终态校验失败。
    FAILED = "FAILED"
    # Java已请求取消且当前租约持有者完成取消提交。
    CANCELLED = "CANCELLED"


class RunBeginOutcome(StrEnum):
    """开始run时返回给生命周期编排器的固定判定。"""

    # 首次创建run并成功取得conversation租约。
    STARTED = "STARTED"
    # 同一RUNNING run租约已过期，当前调用重新取得租约并从checkpoint恢复。
    RESUMED = "RESUMED"
    # 同一会话已有执行者或正处于清理保护窗口。
    IN_PROGRESS = "IN_PROGRESS"
    # 相同run已经完成，可从记录的checkpoint安全重放终态。
    REPLAY_COMPLETED = "REPLAY_COMPLETED"
    # runId已存在但请求摘要/终态语义不兼容。
    CONFLICT = "CONFLICT"


class RunBeginResult(BaseModel):
    """RunStore开始判定；原始lease token只短期返回给当前执行者。"""

    # 禁止未知字段且冻结实例，防止生命周期执行中篡改begin判定。
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Redis Lua脚本返回的开始判定。
    outcome: RunBeginOutcome
    # 对应run安全记录；记录不存在或基础设施异常由实现层处理。
    record: "RunRecord | None" = None
    # 只有取得租约的当前执行者可见原始token；SecretStr避免repr和日志泄漏。
    lease_token: SecretStr | None = None

    # mode="after"表示所有字段先完成类型校验，再执行跨字段一致性检查。
    @model_validator(mode="after")
    def validate_lease_ownership(self) -> "RunBeginResult":
        """确保只有STARTED/RESUMED结果携带租约，返回已验证的自身。"""
        # 集合成员判断表示这两种结果才代表当前进程真正拥有执行权。
        owns_lease = self.outcome in {RunBeginOutcome.STARTED, RunBeginOutcome.RESUMED}
        # ``!=``在此表达逻辑异或：拥有租约和token存在必须同时为真或同时为假。
        if owns_lease != (self.lease_token is not None):
            raise ValueError("只有已开始或恢复的run可以返回租约")
        # after验证器必须返回最终模型实例。
        return self


class RunRecord(BaseModel):
    """Redis中只保存安全标识、摘要、租约和生命周期元数据。"""

    # Redis解析出的额外字段一律拒绝；冻结后不可在Python侧伪造状态迁移。
    model_config = ConfigDict(extra="forbid", frozen=True)

    # 以下六项是Java拥有的运行身份，必须与原请求逐项匹配。
    tenant_id: NonEmptyText
    customer_id: NonEmptyText
    conversation_id: NonEmptyText
    run_id: NonEmptyText
    trigger_message_id: NonEmptyText
    trigger_sequence: PositiveSequence
    # 完整规范请求的SHA-256，只保存摘要而非请求正文。
    request_digest: Sha256Digest
    # 当前run生命周期状态。
    status: RunStatus
    # 原始租约token的哈希；终态提交后删除。
    lease_token_digest: Sha256Digest | None = None
    # 租约理论到期时间，用于诊断；真正所有权仍由Redis带TTL的lease key决定。
    lease_expires_at: datetime | None = None
    # Java写入取消意图的时间；不等于已经进入CANCELLED终态。
    cancel_requested_at: datetime | None = None
    # COMPLETED时记录的LangGraph checkpoint ID。
    checkpoint_id: NonEmptyText | None = None
    # COMPLETED终态事件去除eventIndex后的SHA-256语义摘要。
    final_digest: Sha256Digest | None = None
    # FAILED时保存的稳定错误码，不保存异常正文或堆栈。
    error_code: StableErrorCode | None = None
    # run首次开始、最近更新和终态完成时间。
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None


class RunStoreError(RuntimeError):
    """不携带请求正文或基础设施凭据的RunStore稳定错误。"""

    # ClassVar不会成为实例字段，子类只需覆盖常量即可定义稳定错误类型。
    code: ClassVar[str] = "RUN_STORE_ERROR"
    default_message: ClassVar[str] = "Agent运行状态暂时不可用"

    def __init__(self) -> None:
        """使用固定安全消息初始化异常，不接收可能泄露基础设施信息的参数。"""
        # super()调用RuntimeError构造函数，使str(error)只包含稳定中文诊断。
        super().__init__(self.default_message)


class RunStoreUnavailableError(RunStoreError):
    """Redis不可达、脚本返回畸形或记录无法安全解析。"""

    code = "RUN_STORE_UNAVAILABLE"
    default_message = "Agent运行状态服务暂时不可用"


class RunLeaseLostError(RunStoreError):
    """当前执行者已不再持有conversation租约。"""

    code = "RUN_LEASE_LOST"
    default_message = "Agent运行租约已失效"


class InvalidRunTransitionError(RunStoreError):
    """run不存在或当前状态不允许请求的状态迁移。"""

    code = "INVALID_RUN_TRANSITION"
    default_message = "Agent运行状态转换无效"
