"""验证Task 5安全评估与分类失败状态只能保存受控、脱敏的结构。"""

import importlib

import pytest
from pydantic import ValidationError


def security_contracts():
    """在测试执行阶段导入待实现模块，使缺失模块表现为RED失败而非收集错误。"""
    return importlib.import_module("aicare_agent_service.security.contracts")


def test_input_safety_assessment_is_frozen_and_rejects_unknown_fields() -> None:
    contracts = security_contracts()
    assessment = contracts.InputSafetyAssessment(
        sanitized_text="我的密码是[REDACTED_PASSWORD]，CDK无法激活",
        labels=(contracts.SecurityLabel.PASSWORD,),
        redactions=(
            contracts.RedactionCount(
                data_type=contracts.SensitiveDataType.PASSWORD,
                count=1,
            ),
        ),
        disposition=contracts.SafetyDisposition.ALLOW,
        reason_code=contracts.SafetyReasonCode.SENSITIVE_DATA_REDACTED,
    )

    assert assessment.model_dump(mode="json") == {
        "sanitized_text": "我的密码是[REDACTED_PASSWORD]，CDK无法激活",
        "labels": ["PASSWORD"],
        "redactions": [{"data_type": "PASSWORD", "count": 1}],
        "disposition": "ALLOW",
        "reason_code": "SENSITIVE_DATA_REDACTED",
    }

    with pytest.raises(ValidationError):
        assessment.sanitized_text = "尝试修改"

    with pytest.raises(ValidationError):
        contracts.InputSafetyAssessment(
            sanitized_text="安全文本",
            labels=(),
            redactions=(),
            disposition="ALLOW",
            reason_code="SAFE_INPUT",
            raw_input="不允许保存原文",
        )


def test_safety_disposition_has_no_forced_handoff_state() -> None:
    contracts = security_contracts()

    assert {item.value for item in contracts.SafetyDisposition} == {
        "ALLOW",
        "BLOCK",
        "CLARIFY",
        "HUMAN_HANDOFF",
    }
    with pytest.raises(ValueError):
        contracts.SafetyDisposition("FORCED_HANDOFF")


def test_safety_assessment_rejects_duplicate_labels_and_redaction_types() -> None:
    contracts = security_contracts()

    with pytest.raises(ValidationError):
        contracts.InputSafetyAssessment(
            sanitized_text="[REDACTED_PASSWORD]",
            labels=(contracts.SecurityLabel.PASSWORD, contracts.SecurityLabel.PASSWORD),
            redactions=(),
            disposition="ALLOW",
            reason_code="SENSITIVE_DATA_REDACTED",
        )

    with pytest.raises(ValidationError):
        contracts.InputSafetyAssessment(
            sanitized_text="[REDACTED_PASSWORD]",
            labels=(contracts.SecurityLabel.PASSWORD,),
            redactions=(
                {"data_type": "PASSWORD", "count": 1},
                {"data_type": "PASSWORD", "count": 1},
            ),
            disposition="ALLOW",
            reason_code="SENSITIVE_DATA_REDACTED",
        )


def test_classification_failure_is_stable_and_cannot_store_provider_payload() -> None:
    contracts = security_contracts()
    failure = contracts.RouteClassificationFailure(
        code=contracts.ClassificationFailureCode.INVALID_STRUCTURED_OUTPUT,
        retryable=False,
    )

    assert failure.model_dump(mode="json") == {
        "code": "INVALID_STRUCTURED_OUTPUT",
        "retryable": False,
    }

    with pytest.raises(ValidationError):
        contracts.RouteClassificationFailure(
            code="MODEL_UNAVAILABLE",
            retryable=True,
            raw_response={"content": "provider payload"},
        )


def test_customer_service_state_exposes_only_safe_input_security_channels() -> None:
    security_contracts()
    state_module = importlib.import_module("aicare_agent_service.graph.state")
    annotations = state_module.CustomerServiceState.__annotations__

    assert "sanitized_user_message" in annotations
    assert "input_safety_assessment" in annotations
    assert "classification_failure" in annotations
    assert "raw_user_message" not in annotations
    assert "safety_labels" not in annotations
