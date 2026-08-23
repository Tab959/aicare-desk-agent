"""将严格Java工具结果递归转换为模型可见的安全紧凑JSON。"""

import json
from collections.abc import Mapping, Sequence

from pydantic import BaseModel

from aicare_agent_service.security.redaction import redact_sensitive_input

MAX_SAFE_TOOL_DEPTH = 8
MAX_SAFE_TOOL_LIST_ITEMS = 50
MAX_SAFE_TOOL_STRING_CHARS = 2000

_FORBIDDEN_KEY_FRAGMENTS = frozenset(
    {
        "token",
        "password",
        "credential",
        "secret",
        "cdk",
        "licensekey",
        "accountpassword",
        "downloadurl",
        "signedurl",
        "rawresponse",
        "exception",
        "stacktrace",
    }
)


class ToolResultSanitizationError(ValueError):
    """工具结果包含敏感字段、过大容器或不支持的值类型。"""


def sanitize_payload(payload: object) -> str:
    """递归检查并脱敏载荷，再输出按键排序的紧凑JSON。"""
    # 1、Pydantic响应先转换成使用线协议别名的JSON兼容对象。
    source = (
        payload.model_dump(mode="json", by_alias=True)
        if isinstance(payload, BaseModel)
        else payload
    )
    # 2、递归门禁拒绝敏感字段和越界容器，并对全部字符串复用输入脱敏器。
    sanitized = _sanitize_value(source, depth=0)
    # 3、排序键和紧凑分隔符保证同一事实生成确定性ToolMessage正文。
    return json.dumps(sanitized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sanitize_value(value: object, *, depth: int) -> object:
    """递归返回仅由安全JSON标量、列表和字典组成的新对象。"""
    if depth > MAX_SAFE_TOOL_DEPTH:
        raise ToolResultSanitizationError("工具结果嵌套深度超过安全上限")
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, item in value.items():
            if not isinstance(raw_key, str):
                raise ToolResultSanitizationError("工具结果字段名必须是字符串")
            normalized = "".join(character for character in raw_key.lower() if character.isalnum())
            if any(fragment in normalized for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                raise ToolResultSanitizationError(f"工具结果包含禁止字段：{raw_key}")
            result[raw_key] = _sanitize_value(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > MAX_SAFE_TOOL_LIST_ITEMS:
            raise ToolResultSanitizationError("工具结果列表长度超过安全上限")
        return [_sanitize_value(item, depth=depth + 1) for item in value]
    if isinstance(value, str):
        if len(value) > MAX_SAFE_TOOL_STRING_CHARS:
            raise ToolResultSanitizationError("工具结果字符串长度超过安全上限")
        return redact_sensitive_input(value).sanitized_text
    if value is None or isinstance(value, (bool, int, float)):
        return value
    raise ToolResultSanitizationError("工具结果包含不支持的值类型")
