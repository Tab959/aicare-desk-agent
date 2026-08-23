"""验证敏感值被稳定替换，且脱敏结果不保留原始秘密。"""

import importlib

import pytest


def redaction_module():
    """延迟导入待实现脱敏模块。"""
    return importlib.import_module("aicare_agent_service.security.redaction")


@pytest.mark.parametrize(
    ("raw_text", "secret", "placeholder", "label", "data_type"),
    [
        (
            "Authorization: Bearer bearer-canary-1234567890",
            "bearer-canary-1234567890",
            "[REDACTED_BEARER_TOKEN]",
            "BEARER_TOKEN",
            "BEARER_TOKEN",
        ),
        (
            "token=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjYW5hcnkifQ.signature123",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjYW5hcnkifQ.signature123",
            "[REDACTED_JWT]",
            "JWT",
            "JWT",
        ),
        (
            "账号密码=password-canary-123456",
            "password-canary-123456",
            "账号密码=[REDACTED_PASSWORD]",
            "PASSWORD",
            "PASSWORD",
        ),
        (
            "CDK是AAAAA-BBBBB-CCCCC",
            "AAAAA-BBBBB-CCCCC",
            "[REDACTED_LICENSE_KEY]",
            "LICENSE_KEY",
            "LICENSE_KEY",
        ),
        (
            "下载地址 https://download.example/file?expires=999&signature=url-canary-secret",
            "https://download.example/file?expires=999&signature=url-canary-secret",
            "[REDACTED_SIGNED_URL]",
            "SIGNED_URL",
            "SIGNED_URL",
        ),
    ],
)
def test_sensitive_values_are_replaced_without_retaining_the_original(
    raw_text: str,
    secret: str,
    placeholder: str,
    label: str,
    data_type: str,
) -> None:
    result = redaction_module().redact_sensitive_input(raw_text)

    assert secret not in result.sanitized_text
    assert placeholder in result.sanitized_text
    assert [item.value for item in result.labels] == [label]
    assert [item.model_dump(mode="json") for item in result.redactions] == [
        {"data_type": data_type, "count": 1}
    ]
    assert secret not in repr(result)
    assert secret not in result.model_dump_json()


def test_redaction_normalizes_line_endings_and_aggregates_repeated_types() -> None:
    result = redaction_module().redact_sensitive_input(
        "  密码=first-canary\r\npassword: second-canary  "
    )

    assert result.sanitized_text == ("密码=[REDACTED_PASSWORD]\npassword: [REDACTED_PASSWORD]")
    assert [item.value for item in result.labels] == ["PASSWORD"]
    assert result.redactions[0].count == 2


def test_normal_game_question_is_not_modified() -> None:
    text = "推荐一款100元以内、支持中文的动作游戏。"

    result = redaction_module().redact_sensitive_input(text)

    assert result.sanitized_text == text
    assert result.labels == ()
    assert result.redactions == ()
