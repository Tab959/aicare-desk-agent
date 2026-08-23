"""在用户文本进入LangGraph状态、模型和追踪系统前替换常见敏感数据。"""

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from aicare_agent_service.security.contracts import (
    RedactionCount,
    RedactionResult,
    SecurityLabel,
    SensitiveDataType,
)


@dataclass(frozen=True, slots=True)
class _RedactionRule:
    """描述一条有序脱敏规则及其敏感类型、占位符和替换方式。"""

    data_type: SensitiveDataType
    label: SecurityLabel
    pattern: re.Pattern[str]
    replacement: str | Callable[[re.Match[str]], str]


def _replace_bearer(match: re.Match[str]) -> str:
    """保留Authorization/Bearer语义，只替换实际凭据。"""
    # 1、保留可用于识别问题类型的前缀。
    return f"{match.group('prefix')}[REDACTED_BEARER_TOKEN]"


def _replace_password(match: re.Match[str]) -> str:
    """保留密码字段名和分隔符，只替换字段值。"""
    # 1、保留字段上下文，便于售后Agent理解用户遇到的是凭据问题。
    return f"{match.group('prefix')}[REDACTED_PASSWORD]"


_SIGNED_URL_PATTERN = re.compile(
    r"https?://[^\s，。；;]+[?&][^\s，。；;]*(?:"
    r"x-amz-signature|signature|sig|token|expires|x-amz-credential"
    r")=[^\s，。；;]+",
    re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(
    r"(?P<prefix>(?:authorization\s*:\s*)?bearer\s+)[A-Za-z0-9._~+/=-]{8,}",
    re.IGNORECASE,
)
_JWT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
)
_PASSWORD_PATTERN = re.compile(
    r"(?P<prefix>(?:账号密码|登录密码|密码|password|passwd|pwd)\s*[:=：]\s*)"
    r"[^\s,，;；]+",
    re.IGNORECASE,
)
_LICENSE_KEY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z0-9]{4,5}-){2,}[A-Za-z0-9]{4,5}(?![A-Za-z0-9])"
)

_RULES: tuple[_RedactionRule, ...] = (
    _RedactionRule(
        SensitiveDataType.SIGNED_URL,
        SecurityLabel.SIGNED_URL,
        _SIGNED_URL_PATTERN,
        "[REDACTED_SIGNED_URL]",
    ),
    _RedactionRule(
        SensitiveDataType.BEARER_TOKEN,
        SecurityLabel.BEARER_TOKEN,
        _BEARER_PATTERN,
        _replace_bearer,
    ),
    _RedactionRule(
        SensitiveDataType.JWT,
        SecurityLabel.JWT,
        _JWT_PATTERN,
        "[REDACTED_JWT]",
    ),
    _RedactionRule(
        SensitiveDataType.PASSWORD,
        SecurityLabel.PASSWORD,
        _PASSWORD_PATTERN,
        _replace_password,
    ),
    _RedactionRule(
        SensitiveDataType.LICENSE_KEY,
        SecurityLabel.LICENSE_KEY,
        _LICENSE_KEY_PATTERN,
        "[REDACTED_LICENSE_KEY]",
    ),
)


def redact_sensitive_input(raw_text: str) -> RedactionResult:
    """规范化用户文本并按固定顺序替换敏感数据。"""
    # 1、统一跨平台换行并移除首尾空白，保持正文内部格式不变。
    sanitized_text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    # 2、按规则顺序替换，避免签名URL或Bearer中的内容被后续规则拆散。
    counts: Counter[SensitiveDataType] = Counter()
    labels: list[SecurityLabel] = []
    for rule in _RULES:
        sanitized_text, count = rule.pattern.subn(rule.replacement, sanitized_text)
        if count:
            counts[rule.data_type] += count
            labels.append(rule.label)
    # 3、只返回安全文本及聚合元数据，不保留任何原始敏感值。
    return RedactionResult(
        sanitized_text=sanitized_text,
        labels=tuple(labels),
        redactions=tuple(
            RedactionCount(data_type=rule.data_type, count=counts[rule.data_type])
            for rule in _RULES
            if counts[rule.data_type]
        ),
    )
