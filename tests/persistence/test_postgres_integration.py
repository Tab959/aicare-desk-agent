import operator
import os
from datetime import UTC, datetime, timedelta
from typing import Annotated, TypedDict
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from langgraph.graph import END, START, StateGraph
from redis.asyncio import Redis

from aicare_agent_service.config import CheckpointBackend, Environment, Settings
from aicare_agent_service.contracts.agent_run import AgentRunRequest
from aicare_agent_service.graph.state import CustomerServiceState
from aicare_agent_service.persistence.checkpoint_cleanup import (
    CheckpointCleanupService,
    PostgresCheckpointCatalog,
)
from aicare_agent_service.persistence.checkpointer import checkpointer_resource
from aicare_agent_service.persistence.lifecycle_runner import AgentRunLifecycle
from aicare_agent_service.persistence.models import RunBeginOutcome, RunStatus
from aicare_agent_service.persistence.redis_run_store import RedisRunStore
from tests.persistence.postgres_test_support import prepare_postgres_test_connection

pytestmark = pytest.mark.postgres_integration

_TEST_AES_KEY = "0123456789abcdef0123456789abcdef"


class RecoveryState(TypedDict):
    steps: Annotated[list[str], operator.add]


def build_recovery_graph(checkpointer: object):
    def append_checkpoint(_: RecoveryState) -> dict[str, list[str]]:
        return {"steps": ["checkpoint"]}

    builder = StateGraph(RecoveryState)
    builder.add_node("append_checkpoint", append_checkpoint)
    builder.add_edge(START, "append_checkpoint")
    builder.add_edge("append_checkpoint", END)
    return builder.compile(checkpointer=checkpointer)


def build_lifecycle_graph(checkpointer: object, calls: list[str]):
    def answer(_: CustomerServiceState) -> dict[str, str]:
        calls.append("answer")
        return {"final_answer": "生命周期恢复成功"}

    builder = StateGraph(CustomerServiceState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=checkpointer)


def lifecycle_request(conversation_id: str, run_id: str) -> AgentRunRequest:
    return AgentRunRequest.model_validate(
        {
            "tenantId": "tenant-integration",
            "customerId": "customer-integration",
            "conversationId": conversation_id,
            "runId": run_id,
            "triggerMessageId": f"message-{run_id}",
            "triggerSequence": 1,
            "userMessage": "验证重启恢复",
            "businessContext": {
                "subject": "持久化集成验证",
                "orderId": None,
                "orderNo": None,
                "orderStatus": None,
                "entitlementId": None,
                "entitlementType": None,
                "entitlementStatus": None,
            },
        }
    )


def test_prepare_postgres_test_connection_removes_password_from_conninfo() -> None:
    sanitized_dsn, _ = prepare_postgres_test_connection(
        "postgresql://agent:fake-password@postgres.example.com:5432/aicare_test?sslmode=require"
    )
    parsed = urlsplit(sanitized_dsn)

    assert parsed.scheme == "postgresql"
    assert parsed.username == "agent"
    assert parsed.password is None
    assert parsed.hostname == "postgres.example.com"
    assert parsed.port == 5432
    assert parsed.path == "/aicare_test"
    assert parsed.query == "sslmode=require"


@pytest.mark.asyncio
async def test_postgres_setup_is_idempotent_and_restores_one_conversation_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dsn = os.getenv("AICARE_AGENT_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("未配置AICARE_AGENT_TEST_POSTGRES_DSN")

    conninfo, password = prepare_postgres_test_connection(dsn)
    monkeypatch.setenv("PGPASSWORD", password)

    settings = Settings(
        environment=Environment.TEST,
        checkpoint_backend=CheckpointBackend.POSTGRES,
        agent_postgres_dsn=conninfo,
        checkpoint_encryption_key=_TEST_AES_KEY,
        _env_file=None,
    )
    conversation_id = f"postgres-integration-conversation-{uuid4().hex}"
    run_id = f"postgres-integration-run-{uuid4().hex}"
    config = {"configurable": {"thread_id": conversation_id}}
    assert config["configurable"]["thread_id"] == conversation_id
    assert config["configurable"]["thread_id"] != run_id

    initialized = False
    primary_error: BaseException | None = None
    try:
        async with checkpointer_resource(settings) as first_saver:
            await first_saver.setup()
            first_graph = build_recovery_graph(first_saver)
            first_result = await first_graph.ainvoke({}, config)
            assert first_result["steps"] == ["checkpoint"]
            initialized = True

        async with checkpointer_resource(settings) as second_saver:
            await second_saver.setup()
            second_graph = build_recovery_graph(second_saver)
            recovered_state = await second_graph.aget_state(config)
            assert recovered_state.values["steps"] == ["checkpoint"]
            second_result = await second_graph.ainvoke({}, config)
            assert second_result["steps"] == ["checkpoint", "checkpoint"]
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if initialized:
            try:
                async with checkpointer_resource(settings) as cleanup_saver:
                    await cleanup_saver.adelete_thread(conversation_id)
            except Exception:
                if primary_error is None:
                    raise


@pytest.mark.redis_integration
@pytest.mark.asyncio
async def test_lifecycle_replays_completed_run_after_postgres_and_redis_reconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    postgres_dsn = os.getenv("AICARE_AGENT_TEST_POSTGRES_DSN")
    redis_url = os.getenv("AICARE_AGENT_TEST_REDIS_URL")
    if not postgres_dsn or not redis_url:
        pytest.skip("未同时配置PostgreSQL与Redis集成测试连接")

    conninfo, password = prepare_postgres_test_connection(postgres_dsn)
    monkeypatch.setenv("PGPASSWORD", password)
    settings = Settings(
        environment=Environment.TEST,
        checkpoint_backend=CheckpointBackend.POSTGRES,
        agent_postgres_dsn=conninfo,
        checkpoint_encryption_key=_TEST_AES_KEY,
        _env_file=None,
    )
    suffix = uuid4().hex
    conversation_id = f"lifecycle-integration-conversation-{suffix}"
    run_id = f"lifecycle-integration-run-{suffix}"
    key_prefix = f"aicare:agent:{{lifecycle-integration-{suffix}}}"
    request = lifecycle_request(conversation_id, run_id)
    calls: list[str] = []
    redis_client = Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    store = RedisRunStore(
        redis_client,
        lease_seconds=5,
        retention_seconds=60,
        key_prefix=key_prefix,
    )
    initialized = False
    primary_error: BaseException | None = None
    try:
        await redis_client.ping()
        async with checkpointer_resource(settings) as first_saver:
            await first_saver.setup()
            first_graph = build_lifecycle_graph(first_saver, calls)
            first_lifecycle = AgentRunLifecycle(
                store,
                heartbeat_seconds=0.1,
                timeout_seconds=5,
            )
            first_result = await first_lifecycle.execute(request, first_graph)
            assert first_result.status is RunStatus.COMPLETED
            initialized = True

        async with checkpointer_resource(settings) as second_saver:
            await second_saver.setup()
            second_graph = build_lifecycle_graph(second_saver, calls)
            second_lifecycle = AgentRunLifecycle(
                store,
                heartbeat_seconds=0.1,
                timeout_seconds=5,
            )
            replayed = await second_lifecycle.execute(request, second_graph)
            assert replayed.status is RunStatus.COMPLETED
            assert calls == ["answer"]
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if initialized:
            try:
                async with checkpointer_resource(settings) as cleanup_saver:
                    await cleanup_saver.adelete_thread(conversation_id)
            except Exception:
                if primary_error is None:
                    raise
        keys = [key async for key in redis_client.scan_iter(match=f"{key_prefix}:*")]
        if keys:
            await redis_client.delete(*keys)
        await redis_client.aclose()


@pytest.mark.redis_integration
@pytest.mark.asyncio
async def test_checkpoint_cleanup_dry_run_apply_active_guard_and_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    postgres_dsn = os.getenv("AICARE_AGENT_TEST_POSTGRES_DSN")
    redis_url = os.getenv("AICARE_AGENT_TEST_REDIS_URL")
    if not postgres_dsn or not redis_url:
        pytest.skip("未同时配置PostgreSQL与Redis集成测试连接")

    conninfo, password = prepare_postgres_test_connection(postgres_dsn)
    monkeypatch.setenv("PGPASSWORD", password)
    settings = Settings(
        environment=Environment.TEST,
        checkpoint_backend=CheckpointBackend.POSTGRES,
        agent_postgres_dsn=conninfo,
        checkpoint_encryption_key=_TEST_AES_KEY,
        _env_file=None,
    )
    suffix = uuid4().hex
    conversation_id = f"cleanup-integration-conversation-{suffix}"
    run_id = f"cleanup-integration-run-{suffix}"
    key_prefix = f"aicare:agent:{{cleanup-integration-{suffix}}}"
    request = lifecycle_request(conversation_id, run_id)
    redis_client = Redis.from_url(
        redis_url,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    store = RedisRunStore(
        redis_client,
        lease_seconds=5,
        retention_seconds=60,
        key_prefix=key_prefix,
    )
    try:
        await redis_client.ping()
        async with checkpointer_resource(settings) as saver:
            await saver.setup()
            graph = build_recovery_graph(saver)
            config = {"configurable": {"thread_id": conversation_id}}
            await graph.ainvoke({}, config)
            cleanup = CheckpointCleanupService(PostgresCheckpointCatalog(saver), store)
            future_cutoff = datetime.now(UTC) + timedelta(days=1)

            dry_run = await cleanup.run(cutoff=future_cutoff, limit=100, dry_run=True)
            assert dry_run.eligible == 1
            assert (await graph.aget_state(config)).values["steps"] == ["checkpoint"]

            started = await store.begin(request, "a" * 64)
            assert started.outcome is RunBeginOutcome.STARTED
            guarded = await cleanup.run(cutoff=future_cutoff, limit=100, dry_run=False)
            assert guarded.active == 1
            await store.fail(
                request.run_id,
                started.lease_token.get_secret_value(),
                "TEST_FINISHED",
            )

            applied = await cleanup.run(cutoff=future_cutoff, limit=100, dry_run=False)
            repeated = await cleanup.run(cutoff=future_cutoff, limit=100, dry_run=False)
            assert applied.deleted == 1
            assert repeated.scanned == 0
            assert not (await graph.aget_state(config)).values
    finally:
        try:
            async with checkpointer_resource(settings) as cleanup_saver:
                await cleanup_saver.adelete_thread(conversation_id)
        finally:
            keys = [key async for key in redis_client.scan_iter(match=f"{key_prefix}:*")]
            if keys:
                await redis_client.delete(*keys)
            await redis_client.aclose()
