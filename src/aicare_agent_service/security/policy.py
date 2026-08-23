"""执行无需模型参与的输入安全判定，并生成固定安全处置答复。"""

import re
import unicodedata

from aicare_agent_service.security.contracts import (
    InputSafetyAssessment,
    SafetyDisposition,
    SafetyReasonCode,
    SecurityLabel,
)
from aicare_agent_service.security.redaction import redact_sensitive_input

BLOCKED_INPUT = "[BLOCKED_INPUT]"

_PRIVILEGE_BYPASS_PATTERN = re.compile(
    r"(?:绕过|跳过|规避|关闭|禁用).{0,8}(?:权限|鉴权|授权|校验|风控)|"
    r"(?:直接|强制).{0,8}(?:改成|修改为|设为).{0,8}(?:已退款|已支付|已完成)"
)
_CREDENTIAL_DISCLOSURE_PATTERN = re.compile(
    r"(?=.*(?:显示|输出|告诉|提供|给我|泄露|查看))"
    r"(?=.*(?:密码|口令|访问\s*token|令牌|token|cdk|密钥|凭据))",
    re.IGNORECASE,
)
_PROMPT_INJECTION_PATTERN = re.compile(
    r"(?:忽略|无视|覆盖|忘掉).{0,12}(?:之前|以上|先前).{0,12}(?:规则|指令|提示词)|"
    r"(?:输出|显示|泄露|打印).{0,12}(?:系统提示词|system\s*prompt|隐藏指令)",
    re.IGNORECASE,
)
_HUMAN_REQUEST_PATTERN = re.compile(
    r"(?:转|换|接入|联系).{0,4}(?:人工|真人)(?:客服)?|(?:人工|真人)客服",
    re.IGNORECASE,
)

_FIXED_RESPONSES: dict[SafetyReasonCode, str] = {
    SafetyReasonCode.EMPTY_INPUT: "请描述您需要咨询的问题。",
    SafetyReasonCode.INPUT_TOO_LONG: "消息内容过长，请缩短后重新发送。",
    SafetyReasonCode.DISALLOWED_CONTROL_CHARACTERS: "消息包含不支持的控制字符，请修改后重新发送。",
    SafetyReasonCode.PROMPT_INJECTION_BLOCKED: "该请求涉及不安全或未授权操作，无法处理。",
    SafetyReasonCode.PRIVILEGE_BYPASS_BLOCKED: "该请求涉及不安全或未授权操作，无法处理。",
    SafetyReasonCode.CREDENTIAL_DISCLOSURE_BLOCKED: "该请求涉及不安全或未授权操作，无法处理。",
}


def assess_user_input(raw_text: str, *, max_chars: int) -> InputSafetyAssessment:
    """按固定优先级完成边界校验、脱敏、阻断和显式转人工识别。"""
    # 1、先处理空输入、长度和非法控制字符，阻断内容不进入后续状态。
    if not raw_text.strip():
        return _assessment(
            sanitized_text="",
            label=SecurityLabel.EMPTY_INPUT,
            disposition=SafetyDisposition.CLARIFY,
            reason=SafetyReasonCode.EMPTY_INPUT,
        )
    if len(raw_text) > max_chars:
        return _assessment(
            sanitized_text=BLOCKED_INPUT,
            label=SecurityLabel.INPUT_TOO_LONG,
            disposition=SafetyDisposition.BLOCK,
            reason=SafetyReasonCode.INPUT_TOO_LONG,
        )
    if _contains_disallowed_control_character(raw_text):
        return _assessment(
            sanitized_text=BLOCKED_INPUT,
            label=SecurityLabel.DISALLOWED_CONTROL_CHARACTERS,
            disposition=SafetyDisposition.BLOCK,
            reason=SafetyReasonCode.DISALLOWED_CONTROL_CHARACTERS,
        )

    # 2、对可继续处理的输入脱敏；原文仅在当前函数内用于确定性策略匹配。
    redaction = redact_sensitive_input(raw_text)
    unsafe = _match_unsafe_instruction(raw_text)
    if unsafe is not None:
        label, reason = unsafe
        return InputSafetyAssessment(
            sanitized_text=BLOCKED_INPUT,
            labels=(label,),
            disposition=SafetyDisposition.BLOCK,
            reason_code=reason,
        )

    # 3、只有明确的用户转人工请求进入主动转人工；安全违规永远不会转人工。
    if _HUMAN_REQUEST_PATTERN.search(raw_text):
        return InputSafetyAssessment(
            sanitized_text=redaction.sanitized_text,
            labels=(*redaction.labels, SecurityLabel.EXPLICIT_HUMAN_REQUEST),
            redactions=redaction.redactions,
            disposition=SafetyDisposition.HUMAN_HANDOFF,
            reason_code=SafetyReasonCode.EXPLICIT_HUMAN_REQUEST,
        )

    # 4、普通输入和仅含已脱敏敏感值的输入均可继续进入结构化分类。
    reason = (
        SafetyReasonCode.SENSITIVE_DATA_REDACTED
        if redaction.redactions
        else SafetyReasonCode.SAFE_INPUT
    )
    return InputSafetyAssessment(
        sanitized_text=redaction.sanitized_text,
        labels=redaction.labels,
        redactions=redaction.redactions,
        disposition=SafetyDisposition.ALLOW,
        reason_code=reason,
    )


def build_safety_response(assessment: InputSafetyAssessment) -> str:
    """为无需进入专业子图的安全处置返回固定文案。"""
    # 1、仅允许预定义的阻断或澄清原因生成固定答复。
    try:
        return _FIXED_RESPONSES[assessment.reason_code]
    except KeyError as exc:
        raise ValueError("当前安全处置不生成固定答复") from exc


def _assessment(
    *,
    sanitized_text: str,
    label: SecurityLabel,
    disposition: SafetyDisposition,
    reason: SafetyReasonCode,
) -> InputSafetyAssessment:
    """构造不携带脱敏计数的单原因安全判定。"""
    # 1、所有边界阻断只持久化占位文本、标签和稳定原因码。
    return InputSafetyAssessment(
        sanitized_text=sanitized_text,
        labels=(label,),
        disposition=disposition,
        reason_code=reason,
    )


def _contains_disallowed_control_character(text: str) -> bool:
    """判断文本是否含换行和制表符之外的Unicode控制字符。"""
    # 1、保留客服文本常用的换行和制表符，其余Cc字符全部阻断。
    return any(
        character not in "\n\r\t" and unicodedata.category(character) == "Cc" for character in text
    )


def _match_unsafe_instruction(
    text: str,
) -> tuple[SecurityLabel, SafetyReasonCode] | None:
    """按越权、凭据泄露、提示词注入的固定优先级识别违规指令。"""
    # 1、越权优先级最高，保证混合“越权并转人工”仍固定阻断。
    if _PRIVILEGE_BYPASS_PATTERN.search(text):
        return SecurityLabel.PRIVILEGE_BYPASS, SafetyReasonCode.PRIVILEGE_BYPASS_BLOCKED
    # 2、主动索取敏感凭据按凭据泄露阻断。
    if _CREDENTIAL_DISCLOSURE_PATTERN.search(text):
        return (
            SecurityLabel.CREDENTIAL_DISCLOSURE,
            SafetyReasonCode.CREDENTIAL_DISCLOSURE_BLOCKED,
        )
    # 3、只拦截执行型注入指令，不拦截对“提示词注入”概念的正常提问。
    if _PROMPT_INJECTION_PATTERN.search(text):
        return SecurityLabel.PROMPT_INJECTION, SafetyReasonCode.PROMPT_INJECTION_BLOCKED
    return None
