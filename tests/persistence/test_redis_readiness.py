import logging

import pytest
from redis.exceptions import RedisError

from aicare_agent_service.config import Environment
from aicare_agent_service.persistence.redis_readiness import (
    RedisReadinessError,
    audit_redis_readiness,
    enforce_redis_readiness,
)


class FakeRedis:
    def __init__(
        self,
        *,
        server: dict[str, object] | None = None,
        persistence: dict[str, object] | None = None,
        memory: dict[str, object] | None = None,
        replication: dict[str, object] | None = None,
        cluster: dict[str, object] | None = None,
        role: object | None = None,
        acl_user: object = "aicare-agent",
        appendfsync: object = "everysec",
        error: Exception | None = None,
    ) -> None:
        self.sections = {
            "server": server or {"redis_version": "7.2.0"},
            "persistence": persistence or {"aof_enabled": 1},
            "memory": memory or {"maxmemory_policy": "noeviction"},
            "replication": replication or {"connected_slaves": 1},
            "cluster": cluster or {"cluster_enabled": 0},
        }
        self.role = role or ["master", 0, []]
        self.acl_user = acl_user
        self.appendfsync = appendfsync
        self.error = error
        self.config_calls: list[str] = []

    async def info(self, section: str) -> dict[str, object]:
        if self.error:
            raise self.error
        return self.sections[section]

    async def execute_command(self, *args: str) -> object:
        if self.error:
            raise self.error
        if args == ("ROLE",):
            return self.role
        if args == ("ACL", "WHOAMI"):
            return self.acl_user
        raise AssertionError(args)

    async def config_get(self, parameter: str) -> dict[str, object]:
        self.config_calls.append(parameter)
        if self.error:
            raise self.error
        assert parameter == "appendfsync"
        return {"appendfsync": self.appendfsync}


@pytest.mark.asyncio
async def test_audit_accepts_minimum_production_safe_redis_without_secret_output() -> None:
    client = FakeRedis()

    report = await audit_redis_readiness(client)

    assert report.findings == ()
    assert client.config_calls == ["appendfsync"]
    assert "aicare-agent" not in repr(report)


@pytest.mark.asyncio
async def test_audit_returns_stable_codes_for_unsafe_settings() -> None:
    client = FakeRedis(
        server={"redis_version": "6.2.0"},
        persistence={"aof_enabled": 0},
        memory={"maxmemory_policy": "allkeys-lru"},
        replication={"connected_slaves": 0},
        acl_user="default",
    )

    report = await audit_redis_readiness(client)

    assert {finding.code for finding in report.findings} == {
        "REDIS_VERSION_UNSUPPORTED",
        "REDIS_AOF_DISABLED",
        "REDIS_MAXMEMORY_POLICY_UNSAFE",
        "REDIS_DEFAULT_USER_FORBIDDEN",
        "REDIS_HA_UNAVAILABLE",
    }
    assert "default" not in repr(report)


@pytest.mark.asyncio
async def test_cluster_is_blocked_until_a_cluster_aware_client_is_adapted() -> None:
    report = await audit_redis_readiness(FakeRedis(cluster={"cluster_enabled": 1}))

    assert [finding.code for finding in report.findings] == ["REDIS_CLUSTER_CLIENT_UNSUPPORTED"]


@pytest.mark.asyncio
async def test_replica_endpoint_is_blocked_because_runstore_requires_a_writable_master() -> None:
    report = await audit_redis_readiness(
        FakeRedis(role=["replica", 0, "192.168.1.10", 6379, "connected"])
    )

    assert [finding.code for finding in report.findings] == ["REDIS_PRIMARY_REQUIRED"]


@pytest.mark.asyncio
async def test_master_without_connected_replica_is_not_highly_available() -> None:
    report = await audit_redis_readiness(FakeRedis(replication={"connected_slaves": 0}))

    assert [finding.code for finding in report.findings] == ["REDIS_HA_UNAVAILABLE"]


@pytest.mark.asyncio
async def test_unsafe_appendfsync_is_reported_with_a_stable_code() -> None:
    report = await audit_redis_readiness(FakeRedis(appendfsync="no"))

    assert [finding.code for finding in report.findings] == ["REDIS_APPEND_FSYNC_UNSAFE"]


@pytest.mark.asyncio
async def test_malformed_acl_response_is_reported_with_a_stable_code() -> None:
    report = await audit_redis_readiness(FakeRedis(acl_user={"unexpected": "value"}))

    assert [finding.code for finding in report.findings] == ["REDIS_ACL_UNAVAILABLE"]


@pytest.mark.asyncio
@pytest.mark.parametrize("acl_user", ["", b"", b"\xff"])
async def test_empty_or_invalid_acl_identity_is_reported_with_a_stable_code(
    acl_user: object,
) -> None:
    report = await audit_redis_readiness(FakeRedis(acl_user=acl_user))

    assert [finding.code for finding in report.findings] == ["REDIS_ACL_UNAVAILABLE"]


@pytest.mark.asyncio
async def test_malformed_role_response_is_reported_with_a_stable_code() -> None:
    report = await audit_redis_readiness(FakeRedis(role={"unexpected": "value"}))

    assert [finding.code for finding in report.findings] == ["REDIS_ROLE_UNAVAILABLE"]


@pytest.mark.asyncio
async def test_audit_maps_unavailable_or_acl_permission_failure_to_stable_code() -> None:
    report = await audit_redis_readiness(FakeRedis(error=RedisError("redis://secret@host")))

    assert [finding.code for finding in report.findings] == ["REDIS_AUDIT_UNAVAILABLE"]
    assert "secret" not in repr(report)


@pytest.mark.asyncio
async def test_production_rejects_blocking_findings_and_development_only_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FakeRedis(memory={"maxmemory_policy": "allkeys-lru"})

    with pytest.raises(RedisReadinessError, match="Redis生产环境门禁未通过") as exc_info:
        await enforce_redis_readiness(client, Environment.PRODUCTION)
    assert "allkeys" not in str(exc_info.value)
    assert "REDIS_MAXMEMORY_POLICY_UNSAFE" in str(exc_info.value)

    caplog.set_level(logging.WARNING)
    report = await enforce_redis_readiness(client, Environment.DEVELOPMENT)
    assert report.findings
    assert "REDIS_MAXMEMORY_POLICY_UNSAFE" in caplog.text
    assert "allkeys" not in caplog.text
