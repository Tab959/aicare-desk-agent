import asyncio
import os
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from pydantic import SecretStr
from redis.asyncio import Redis

from aicare_agent_service.config import RedisMode, Settings
from aicare_agent_service.contracts.agent_run import AgentRunRequest
from aicare_agent_service.persistence.models import (
    InvalidRunTransitionError,
    RunBeginOutcome,
    RunLeaseLostError,
    RunStatus,
    RunStoreUnavailableError,
)
from aicare_agent_service.persistence.redis_run_store import RedisRunStore, build_redis_client

pytestmark = pytest.mark.redis_integration


def make_request(
    *,
    run_id: str = "run-001",
    conversation_id: str = "conversation-001",
) -> AgentRunRequest:
    return AgentRunRequest.model_validate(
        {
            "tenantId": "tenant-001",
            "customerId": "customer-001",
            "conversationId": conversation_id,
            "runId": run_id,
            "triggerMessageId": f"message-{run_id}",
            "triggerSequence": 1,
            "userMessage": "查询订单交付状态",
            "businessContext": {
                "subject": "订单交付",
                "orderId": "order-001",
                "orderNo": "AD2026-001",
                "orderStatus": "PAID",
                "entitlementId": None,
                "entitlementType": None,
                "entitlementStatus": None,
            },
        }
    )


def reveal(secret: SecretStr | None) -> str:
    assert secret is not None
    return secret.get_secret_value()


@pytest_asyncio.fixture
async def redis_client() -> AsyncIterator[Redis]:
    redis_url = os.getenv("AICARE_AGENT_TEST_REDIS_URL")
    if not redis_url:
        pytest.skip("未配置AICARE_AGENT_TEST_REDIS_URL")

    client = Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    await client.ping()
    try:
        yield client
    finally:
        await client.aclose()


@pytest_asyncio.fixture
async def run_store(redis_client: Redis) -> AsyncIterator[RedisRunStore]:
    key_prefix = f"aicare:agent:{{run-store-test-{uuid4().hex}}}"
    store = RedisRunStore(
        redis_client,
        lease_seconds=1,
        retention_seconds=60,
        key_prefix=key_prefix,
    )
    try:
        yield store
    finally:
        keys = [key async for key in redis_client.scan_iter(match=f"{key_prefix}:*")]
        if keys:
            await redis_client.delete(*keys)


@pytest.mark.asyncio
async def test_redis_client_uses_bounded_async_pool_without_exposing_secret() -> None:
    secret_url = SecretStr("redis://:private-password@127.0.0.1:6379/1")
    client = build_redis_client(
        secret_url,
        max_connections=12,
        socket_timeout_seconds=1.5,
        health_check_interval_seconds=10,
    )
    try:
        kwargs = client.connection_pool.connection_kwargs
        assert client.connection_pool.max_connections == 12
        assert kwargs["db"] == 1
        assert kwargs["socket_timeout"] == 1.5
        assert kwargs["health_check_interval"] == 10
        assert "private-password" not in repr(client)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_sentinel_client_uses_separate_control_plane_and_data_plane_acl() -> None:
    settings = Settings(
        redis_mode="sentinel",
        agent_redis_url="redis://aicare_agent:data-secret@unused:6379/0",
        redis_sentinels="sentinel-a:26379,sentinel-b:26380,sentinel-c:26381",
        redis_master_name="aicare-agent-master",
        redis_sentinel_username="aicare_sentinel",
        redis_sentinel_password="sentinel-secret",
        _env_file=None,
    )

    client = build_redis_client(
        settings.agent_redis_url,
        redis_mode=RedisMode.SENTINEL,
        sentinel_endpoints=settings.redis_sentinel_endpoints,
        sentinel_master_name=settings.redis_master_name,
        sentinel_username=settings.redis_sentinel_username,
        sentinel_password=settings.redis_sentinel_password,
        max_connections=12,
        socket_timeout_seconds=1.5,
        health_check_interval_seconds=10,
    )
    try:
        kwargs = client.connection_pool.connection_kwargs
        sentinel = client.connection_pool.connection_kwargs["connection_pool"].sentinel_manager
        assert kwargs["username"] == "aicare_agent"
        assert kwargs["password"] == "data-secret"
        assert sentinel.sentinel_kwargs["username"] == "aicare_sentinel"
        assert sentinel.sentinel_kwargs["password"] == "sentinel-secret"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_completed_run_replays_without_acquiring_a_new_lease(
    run_store: RedisRunStore,
) -> None:
    request = make_request()
    started = await run_store.begin(request, "a" * 64)
    await run_store.complete(
        request.run_id,
        reveal(started.lease_token),
        checkpoint_id="checkpoint-001",
        final_digest="b" * 64,
    )

    replay = await run_store.begin(request, "a" * 64)

    assert replay.outcome is RunBeginOutcome.REPLAY_COMPLETED
    assert replay.lease_token is None
    assert replay.record is not None
    assert replay.record.status is RunStatus.COMPLETED
    assert replay.record.final_digest == "b" * 64


@pytest.mark.asyncio
async def test_keys_share_one_hash_tag_hide_java_ids_and_use_expected_ttls(
    run_store: RedisRunStore,
    redis_client: Redis,
) -> None:
    request = make_request()
    started = await run_store.begin(request, "a" * 64)
    keys = [key async for key in redis_client.scan_iter(match=f"{run_store.key_prefix}:*")]

    assert len(keys) == 3
    assert all(run_store.key_prefix in key for key in keys)
    assert all(request.run_id not in key and request.conversation_id not in key for key in keys)
    run_key = next(key for key in keys if ":run:" in key)
    lease_key = next(key for key in keys if key.endswith(":lease"))
    active_key = next(key for key in keys if key.endswith(":active-run"))
    assert 0 < await redis_client.pttl(lease_key) <= 1000
    assert await redis_client.ttl(active_key) == -1
    assert 55 <= await redis_client.ttl(run_key) <= 60

    await run_store.complete(
        request.run_id,
        reveal(started.lease_token),
        checkpoint_id="checkpoint-001",
        final_digest="b" * 64,
    )

    assert await redis_client.exists(lease_key) == 0
    assert await redis_client.exists(active_key) == 0
    assert 55 <= await redis_client.ttl(run_key) <= 60


@pytest.mark.asyncio
async def test_run_hash_contains_only_allowlisted_metadata_fields(
    run_store: RedisRunStore,
    redis_client: Redis,
) -> None:
    request = make_request()
    started = await run_store.begin(request, "a" * 64)
    await run_store.complete(
        request.run_id,
        reveal(started.lease_token),
        checkpoint_id="checkpoint-001",
        final_digest="b" * 64,
    )
    run_keys = [key async for key in redis_client.scan_iter(match=f"{run_store.key_prefix}:run:*")]
    assert len(run_keys) == 1
    run_key = run_keys[0]

    fields = set(await redis_client.hgetall(run_key))

    assert fields == {
        "tenant_id",
        "customer_id",
        "conversation_id",
        "run_id",
        "trigger_message_id",
        "trigger_sequence",
        "request_digest",
        "status",
        "checkpoint_id",
        "final_digest",
        "started_at_ms",
        "updated_at_ms",
        "completed_at_ms",
    }
    values = await redis_client.hgetall(run_key)
    assert request.user_message not in values.values()
    assert "订单交付" not in values.values()


@pytest.mark.asyncio
async def test_same_run_with_different_digest_is_a_conflict(
    run_store: RedisRunStore,
) -> None:
    request = make_request()
    await run_store.begin(request, "a" * 64)

    result = await run_store.begin(request, "b" * 64)

    assert result.outcome is RunBeginOutcome.CONFLICT
    assert result.lease_token is None


@pytest.mark.asyncio
async def test_invalid_digest_is_rejected_before_writing_redis(
    run_store: RedisRunStore,
) -> None:
    request = make_request()

    with pytest.raises(ValueError, match="SHA-256"):
        await run_store.begin(request, "invalid")

    assert await run_store.get(request.run_id) is None


@pytest.mark.asyncio
async def test_active_conversation_allows_only_one_run(run_store: RedisRunStore) -> None:
    first = await run_store.begin(make_request(run_id="run-001"), "a" * 64)

    second = await run_store.begin(make_request(run_id="run-002"), "b" * 64)

    assert first.outcome is RunBeginOutcome.STARTED
    assert second.outcome is RunBeginOutcome.IN_PROGRESS
    assert second.record is None


@pytest.mark.asyncio
async def test_different_conversations_can_start_independently(
    run_store: RedisRunStore,
) -> None:
    first = await run_store.begin(
        make_request(run_id="run-001", conversation_id="conversation-001"),
        "a" * 64,
    )
    second = await run_store.begin(
        make_request(run_id="run-002", conversation_id="conversation-002"),
        "b" * 64,
    )

    assert first.outcome is RunBeginOutcome.STARTED
    assert second.outcome is RunBeginOutcome.STARTED


@pytest.mark.asyncio
async def test_expired_lease_is_resumed_and_old_owner_cannot_complete(
    run_store: RedisRunStore,
) -> None:
    request = make_request()
    first = await run_store.begin(request, "a" * 64)
    old_token = reveal(first.lease_token)
    await asyncio.sleep(1.1)

    resumed = await run_store.begin(request, "a" * 64)

    assert resumed.outcome is RunBeginOutcome.RESUMED
    assert reveal(resumed.lease_token) != old_token
    with pytest.raises(RunLeaseLostError):
        await run_store.complete(
            request.run_id,
            old_token,
            checkpoint_id="checkpoint-old",
            final_digest="b" * 64,
        )

    await run_store.complete(
        request.run_id,
        reveal(resumed.lease_token),
        checkpoint_id="checkpoint-new",
        final_digest="c" * 64,
    )
    record = await run_store.get(request.run_id)
    assert record is not None
    assert record.checkpoint_id == "checkpoint-new"


@pytest.mark.asyncio
async def test_running_run_remains_active_after_lease_expiry(
    run_store: RedisRunStore,
) -> None:
    request = make_request()
    await run_store.begin(request, "a" * 64)
    await asyncio.sleep(1.1)

    assert await run_store.is_conversation_active(request.conversation_id) is True

    other = await run_store.begin(make_request(run_id="run-002"), "b" * 64)
    assert other.outcome is RunBeginOutcome.IN_PROGRESS


@pytest.mark.asyncio
async def test_terminal_run_is_not_active_for_checkpoint_cleanup(
    run_store: RedisRunStore,
) -> None:
    request = make_request()
    started = await run_store.begin(request, "a" * 64)
    await run_store.complete(
        request.run_id,
        reveal(started.lease_token),
        checkpoint_id="checkpoint-001",
        final_digest="b" * 64,
    )

    assert await run_store.is_conversation_active(request.conversation_id) is False


@pytest.mark.asyncio
async def test_cleanup_guard_blocks_new_run_and_is_token_safe(
    run_store: RedisRunStore,
) -> None:
    request = make_request()
    token = await run_store.acquire_cleanup_guard(request.conversation_id)
    assert token is not None

    blocked = await run_store.begin(request, "a" * 64)
    assert blocked.outcome is RunBeginOutcome.IN_PROGRESS

    await run_store.release_cleanup_guard(request.conversation_id, "wrong-token")
    still_blocked = await run_store.begin(request, "a" * 64)
    assert still_blocked.outcome is RunBeginOutcome.IN_PROGRESS

    await run_store.release_cleanup_guard(request.conversation_id, token)
    started = await run_store.begin(request, "a" * 64)
    assert started.outcome is RunBeginOutcome.STARTED


@pytest.mark.asyncio
async def test_dangling_active_marker_is_self_healed_before_cleanup(
    run_store: RedisRunStore,
    redis_client: Redis,
) -> None:
    request = make_request()
    await run_store.begin(request, "a" * 64)
    keys = [key async for key in redis_client.scan_iter(match=f"{run_store.key_prefix}:*")]
    run_key = next(key for key in keys if ":run:" in key)
    active_key = next(key for key in keys if key.endswith(":active-run"))
    lease_key = next(key for key in keys if key.endswith(":lease"))
    await redis_client.delete(run_key, lease_key)

    assert await run_store.is_conversation_active(request.conversation_id) is False
    assert await redis_client.exists(active_key) == 0


@pytest.mark.asyncio
async def test_renew_lease_keeps_the_current_owner(run_store: RedisRunStore) -> None:
    request = make_request()
    started = await run_store.begin(request, "a" * 64)
    token = reveal(started.lease_token)

    await asyncio.sleep(0.6)
    await run_store.renew_lease(request.run_id, token)
    await asyncio.sleep(0.6)

    duplicate = await run_store.begin(request, "a" * 64)
    assert duplicate.outcome is RunBeginOutcome.IN_PROGRESS


@pytest.mark.asyncio
async def test_cancel_intent_is_visible_to_current_owner_without_finishing_run(
    run_store: RedisRunStore,
) -> None:
    request = make_request()
    started = await run_store.begin(request, "a" * 64)

    await run_store.request_cancel(request.run_id)

    assert await run_store.is_cancel_requested(request.run_id) is True
    record = await run_store.get(request.run_id)
    assert record is not None
    assert record.status is RunStatus.RUNNING
    assert record.cancel_requested_at is not None
    await run_store.cancel(request.run_id, reveal(started.lease_token))
    assert await run_store.is_cancel_requested(request.run_id) is False


@pytest.mark.asyncio
async def test_complete_is_idempotent_only_for_the_same_terminal_payload(
    run_store: RedisRunStore,
) -> None:
    request = make_request()
    started = await run_store.begin(request, "a" * 64)
    token = reveal(started.lease_token)
    await run_store.complete(request.run_id, token, "checkpoint-001", "b" * 64)

    await run_store.complete(request.run_id, token, "checkpoint-001", "b" * 64)
    with pytest.raises(InvalidRunTransitionError):
        await run_store.complete(request.run_id, token, "checkpoint-002", "c" * 64)


@pytest.mark.asyncio
async def test_invalid_terminal_metadata_does_not_change_running_record(
    run_store: RedisRunStore,
) -> None:
    request = make_request()
    started = await run_store.begin(request, "a" * 64)
    token = reveal(started.lease_token)

    with pytest.raises(ValueError, match="SHA-256"):
        await run_store.complete(request.run_id, token, "checkpoint-001", "invalid")
    with pytest.raises(ValueError, match="错误码"):
        await run_store.fail(request.run_id, token, "model-timeout")

    record = await run_store.get(request.run_id)
    assert record is not None
    assert record.status is RunStatus.RUNNING


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_method", "expected_status"),
    [("fail", RunStatus.FAILED), ("cancel", RunStatus.CANCELLED)],
)
async def test_fail_and_cancel_release_the_conversation_lease(
    run_store: RedisRunStore,
    terminal_method: str,
    expected_status: RunStatus,
) -> None:
    first_request = make_request(run_id="run-001")
    started = await run_store.begin(first_request, "a" * 64)
    token = reveal(started.lease_token)

    if terminal_method == "fail":
        await run_store.fail(first_request.run_id, token, "MODEL_TIMEOUT")
    else:
        await run_store.cancel(first_request.run_id, token)

    record = await run_store.get(first_request.run_id)
    assert record is not None
    assert record.status is expected_status
    second = await run_store.begin(make_request(run_id="run-002"), "b" * 64)
    assert second.outcome is RunBeginOutcome.STARTED


@pytest.mark.asyncio
async def test_invalid_terminal_transition_is_rejected(run_store: RedisRunStore) -> None:
    request = make_request()
    started = await run_store.begin(request, "a" * 64)
    token = reveal(started.lease_token)
    await run_store.fail(request.run_id, token, "MODEL_TIMEOUT")

    with pytest.raises(InvalidRunTransitionError):
        await run_store.cancel(request.run_id, token)


@pytest.mark.asyncio
async def test_unreachable_redis_maps_to_stable_unavailable_error() -> None:
    client = Redis.from_url(
        "redis://127.0.0.1:1/1",
        decode_responses=True,
        socket_connect_timeout=0.1,
        socket_timeout=0.1,
    )
    store = RedisRunStore(client, lease_seconds=1, retention_seconds=60)
    try:
        with pytest.raises(RunStoreUnavailableError):
            await store.begin(make_request(), "a" * 64)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_unknown_lua_result_maps_to_stable_unavailable_error(
    run_store: RedisRunStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unknown_result(*args: object, **kwargs: object) -> list[str]:
        return ["UNKNOWN_RESULT"]

    monkeypatch.setattr(run_store, "_begin_script", unknown_result)

    with pytest.raises(RunStoreUnavailableError):
        await run_store.begin(make_request(), "a" * 64)
