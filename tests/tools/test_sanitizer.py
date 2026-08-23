"""验证Java工具结果在进入ToolMessage、checkpoint和Trace前的安全投影。"""

import pytest

from aicare_agent_service.tools.sanitizer import ToolResultSanitizationError, sanitize_payload


@pytest.mark.parametrize(
    "field_name",
    [
        "token",
        "accountPassword",
        "credential",
        "secretValue",
        "cdkCode",
        "license_key",
        "downloadUrl",
        "signedUrl",
        "rawResponse",
        "internalException",
    ],
)
def test_sensitive_or_internal_fields_fail_closed_recursively(field_name: str) -> None:
    with pytest.raises(ToolResultSanitizationError, match="禁止字段"):
        sanitize_payload({"safe": {field_name: "must-never-appear"}})


def test_strings_reuse_redaction_before_deterministic_compact_json() -> None:
    content = sanitize_payload(
        {"kind": "GAME_DETAIL", "description": "password: hunter2", "available": True}
    )

    assert content == (
        '{"available":true,"description":"password: [REDACTED_PASSWORD]","kind":"GAME_DETAIL"}'
    )
    assert "hunter2" not in content


def test_list_string_and_depth_limits_fail_closed() -> None:
    with pytest.raises(ToolResultSanitizationError, match="列表长度"):
        sanitize_payload({"items": list(range(51))})
    with pytest.raises(ToolResultSanitizationError, match="字符串长度"):
        sanitize_payload({"description": "x" * 2001})
    with pytest.raises(ToolResultSanitizationError, match="嵌套深度"):
        sanitize_payload({"a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": 1}}}}}}}}})
