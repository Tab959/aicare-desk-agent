from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from aicare_agent_service.persistence.models import (
    InvalidRunTransitionError,
    RunBeginOutcome,
    RunBeginResult,
    RunLeaseLostError,
    RunRecord,
    RunStatus,
    RunStoreError,
    RunStoreUnavailableError,
)


def test_persistence_package_exports_run_store_boundaries() -> None:
    from aicare_agent_service.persistence import (
        AgentRunLifecycle,
        RedisRunStore,
        RunStore,
        build_redis_client,
    )

    assert AgentRunLifecycle.__name__ == "AgentRunLifecycle"
    assert RedisRunStore.__name__ == "RedisRunStore"
    assert RunStore.__name__ == "RunStore"
    assert callable(build_redis_client)


def test_run_record_contains_only_safe_ledger_metadata_and_is_frozen() -> None:
    now = datetime.now(UTC)
    record = RunRecord(
        tenant_id="tenant-001",
        customer_id="customer-001",
        conversation_id="conversation-001",
        run_id="run-001",
        trigger_message_id="message-001",
        trigger_sequence=3,
        request_digest="a" * 64,
        status=RunStatus.RUNNING,
        lease_token_digest="b" * 64,
        lease_expires_at=now + timedelta(seconds=30),
        checkpoint_id=None,
        final_digest=None,
        error_code=None,
        started_at=now,
        updated_at=now,
        completed_at=None,
    )

    assert record.status is RunStatus.RUNNING
    assert "user_message" not in RunRecord.model_fields
    assert "final_answer" not in RunRecord.model_fields
    assert "raw_response" not in RunRecord.model_fields
    with pytest.raises(ValidationError):
        record.status = RunStatus.COMPLETED


def test_run_record_rejects_unknown_or_malformed_metadata() -> None:
    now = datetime.now(UTC)
    base = {
        "tenant_id": "tenant-001",
        "customer_id": "customer-001",
        "conversation_id": "conversation-001",
        "run_id": "run-001",
        "trigger_message_id": "message-001",
        "trigger_sequence": 1,
        "request_digest": "a" * 64,
        "status": RunStatus.RUNNING,
        "started_at": now,
        "updated_at": now,
    }

    with pytest.raises(ValidationError):
        RunRecord(**(base | {"request_digest": "not-a-sha256"}))
    with pytest.raises(ValidationError):
        RunRecord(**base, token="must-not-be-accepted")


def test_run_status_and_begin_outcomes_are_fixed() -> None:
    assert {status.value for status in RunStatus} == {
        "RUNNING",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
    }


def test_begin_result_hides_the_lease_token() -> None:
    result = RunBeginResult(
        outcome=RunBeginOutcome.STARTED,
        lease_token="secret-lease-token",
    )

    assert "secret-lease-token" not in repr(result)
    assert result.lease_token is not None
    assert result.lease_token.get_secret_value() == "secret-lease-token"


def test_begin_result_rejects_lease_ownership_mismatch() -> None:
    with pytest.raises(ValidationError):
        RunBeginResult(outcome=RunBeginOutcome.STARTED)
    with pytest.raises(ValidationError):
        RunBeginResult(
            outcome=RunBeginOutcome.CONFLICT,
            lease_token="must-not-be-returned",
        )
    assert {outcome.value for outcome in RunBeginOutcome} == {
        "STARTED",
        "RESUMED",
        "IN_PROGRESS",
        "REPLAY_COMPLETED",
        "CONFLICT",
    }


def test_run_store_errors_have_stable_codes_without_payloads() -> None:
    errors = (
        RunStoreUnavailableError(),
        RunLeaseLostError(),
        InvalidRunTransitionError(),
    )

    assert all(isinstance(error, RunStoreError) for error in errors)
    assert [error.code for error in errors] == [
        "RUN_STORE_UNAVAILABLE",
        "RUN_LEASE_LOST",
        "INVALID_RUN_TRANSITION",
    ]
    assert all("用户消息" not in str(error) for error in errors)
