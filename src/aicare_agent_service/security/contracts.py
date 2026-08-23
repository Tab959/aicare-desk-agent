"""定义可安全写入LangGraph checkpoint的输入安全与分类失败契约。"""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator


class SecurityContractModel(BaseModel):
    """安全契约公共基类，禁止额外字段并冻结实例。"""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SensitiveDataType(StrEnum):
    """脱敏计数支持的敏感数据类型。"""

    BEARER_TOKEN = "BEARER_TOKEN"
    JWT = "JWT"
    PASSWORD = "PASSWORD"
    LICENSE_KEY = "LICENSE_KEY"
    SIGNED_URL = "SIGNED_URL"


class SecurityLabel(StrEnum):
    """输入安全检测产生的稳定标签。"""

    EMPTY_INPUT = "EMPTY_INPUT"
    INPUT_TOO_LONG = "INPUT_TOO_LONG"
    DISALLOWED_CONTROL_CHARACTERS = "DISALLOWED_CONTROL_CHARACTERS"
    BEARER_TOKEN = "BEARER_TOKEN"
    JWT = "JWT"
    PASSWORD = "PASSWORD"
    LICENSE_KEY = "LICENSE_KEY"
    SIGNED_URL = "SIGNED_URL"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    PRIVILEGE_BYPASS = "PRIVILEGE_BYPASS"
    CREDENTIAL_DISCLOSURE = "CREDENTIAL_DISCLOSURE"
    EXPLICIT_HUMAN_REQUEST = "EXPLICIT_HUMAN_REQUEST"


class SafetyDisposition(StrEnum):
    """根图安全策略允许产生的四种确定性处置。"""

    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    CLARIFY = "CLARIFY"
    HUMAN_HANDOFF = "HUMAN_HANDOFF"


class SafetyReasonCode(StrEnum):
    """安全处置使用的稳定原因码。"""

    SAFE_INPUT = "SAFE_INPUT"
    SENSITIVE_DATA_REDACTED = "SENSITIVE_DATA_REDACTED"
    EMPTY_INPUT = "EMPTY_INPUT"
    INPUT_TOO_LONG = "INPUT_TOO_LONG"
    DISALLOWED_CONTROL_CHARACTERS = "DISALLOWED_CONTROL_CHARACTERS"
    PROMPT_INJECTION_BLOCKED = "PROMPT_INJECTION_BLOCKED"
    PRIVILEGE_BYPASS_BLOCKED = "PRIVILEGE_BYPASS_BLOCKED"
    CREDENTIAL_DISCLOSURE_BLOCKED = "CREDENTIAL_DISCLOSURE_BLOCKED"
    EXPLICIT_HUMAN_REQUEST = "EXPLICIT_HUMAN_REQUEST"


class ClassificationFailureCode(StrEnum):
    """结构化分类阶段允许持久化的失败代码。"""

    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INVALID_STRUCTURED_OUTPUT = "INVALID_STRUCTURED_OUTPUT"


class RedactionCount(SecurityContractModel):
    """记录某类敏感数据被替换的次数，不保存被替换内容。"""

    # 被替换的敏感数据类型。
    data_type: SensitiveDataType
    # 实际替换次数，必须为正整数。
    count: Annotated[int, Field(strict=True, ge=1)]


class RedactionResult(SecurityContractModel):
    """保存可进入后续处理的文本、命中标签与分类聚合计数。"""

    # 已替换敏感值且完成换行规范化的文本。
    sanitized_text: Annotated[str, Field(strict=True, max_length=32768)]
    # 按首次命中顺序排列的敏感类型标签。
    labels: tuple[SecurityLabel, ...] = ()
    # 每种敏感类型仅保留一条聚合计数。
    redactions: tuple[RedactionCount, ...] = ()

    @model_validator(mode="after")
    def validate_unique_entries(self) -> "RedactionResult":
        """保证标签和计数均为无重复的确定性集合。"""
        # 1、拒绝重复标签，避免同一输入产生不稳定的审计语义。
        if len(self.labels) != len(set(self.labels)):
            raise ValueError("脱敏标签不得重复")
        # 2、拒绝同一敏感类型出现多条计数，要求调用方先完成聚合。
        data_types = [item.data_type for item in self.redactions]
        if len(data_types) != len(set(data_types)):
            raise ValueError("同一敏感数据类型不得重复计数")
        # 3、返回已验证实例供Pydantic完成构造。
        return self


_REASON_DISPOSITIONS: dict[SafetyReasonCode, SafetyDisposition] = {
    SafetyReasonCode.SAFE_INPUT: SafetyDisposition.ALLOW,
    SafetyReasonCode.SENSITIVE_DATA_REDACTED: SafetyDisposition.ALLOW,
    SafetyReasonCode.EMPTY_INPUT: SafetyDisposition.CLARIFY,
    SafetyReasonCode.INPUT_TOO_LONG: SafetyDisposition.BLOCK,
    SafetyReasonCode.DISALLOWED_CONTROL_CHARACTERS: SafetyDisposition.BLOCK,
    SafetyReasonCode.PROMPT_INJECTION_BLOCKED: SafetyDisposition.BLOCK,
    SafetyReasonCode.PRIVILEGE_BYPASS_BLOCKED: SafetyDisposition.BLOCK,
    SafetyReasonCode.CREDENTIAL_DISCLOSURE_BLOCKED: SafetyDisposition.BLOCK,
    SafetyReasonCode.EXPLICIT_HUMAN_REQUEST: SafetyDisposition.HUMAN_HANDOFF,
}


class InputSafetyAssessment(SecurityContractModel):
    """保存脱敏文本、安全标签、替换计数和根图处置。"""

    # 允许进入消息、checkpoint和模型上下文的安全文本。
    sanitized_text: Annotated[str, Field(strict=True, max_length=32768)]
    # 本次输入命中的稳定安全标签，禁止重复。
    labels: tuple[SecurityLabel, ...] = ()
    # 各类敏感数据的替换次数，禁止同类型重复。
    redactions: tuple[RedactionCount, ...] = ()
    # 根图下一步使用的确定性处置。
    disposition: SafetyDisposition
    # 与处置一一对应的稳定原因码。
    reason_code: SafetyReasonCode

    @model_validator(mode="after")
    def validate_consistency(self) -> "InputSafetyAssessment":
        """拒绝重复标签、重复计数和原因码与处置不一致。"""
        # 1、标签重复会破坏审计统计的确定性。
        if len(self.labels) != len(set(self.labels)):
            raise ValueError("安全标签不得重复")
        # 2、每类敏感数据必须只出现一条聚合计数。
        data_types = [item.data_type for item in self.redactions]
        if len(data_types) != len(set(data_types)):
            raise ValueError("同一敏感数据类型不得重复计数")
        # 3、原因码只能映射到预先定义的处置，不能由调用方自由组合。
        if _REASON_DISPOSITIONS[self.reason_code] is not self.disposition:
            raise ValueError("安全原因码与处置不一致")
        return self


class RouteClassificationFailure(SecurityContractModel):
    """保存分类失败代码和是否可重试，不保存Provider异常或响应。"""

    # 分类失败的稳定代码。
    code: ClassificationFailureCode
    # 上层是否可以在当前run预算内有限重试。
    retryable: StrictBool
