import pytest
from pydantic import SecretStr

from aicare_agent_service.config import CheckpointBackend, RedisMode, Settings
from aicare_agent_service.persistence import cleanup_checkpoints
from aicare_agent_service.persistence.checkpoint_cleanup import CheckpointCleanupResult


def test_cli_defaults_to_dry_run() -> None:
    args = cleanup_checkpoints.build_parser().parse_args([])

    assert args.apply is False


def test_cli_requires_explicit_apply_flag_for_deletion() -> None:
    args = cleanup_checkpoints.build_parser().parse_args(["--apply"])

    assert args.apply is True


def test_cli_returns_stable_error_without_exception_detail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fail(_: bool) -> None:
        raise RuntimeError("postgresql://user:password@database/private")

    monkeypatch.setattr(cleanup_checkpoints, "run_cleanup", fail)

    exit_code = cleanup_checkpoints.main([])
    output = capsys.readouterr()

    assert exit_code == 1
    assert output.out == ""
    assert output.err == "Checkpoint清理失败：请检查Agent持久化配置和基础设施状态。\n"
    assert "password" not in output.err


@pytest.mark.asyncio
async def test_cleanup_uses_sentinel_discovery_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class ProbeRedis:
        async def aclose(self) -> None:
            return None

    class ProbeRunStore:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

    class ProbeService:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        async def run(self, **kwargs: object) -> CheckpointCleanupResult:
            del kwargs
            return CheckpointCleanupResult(scanned=0, eligible=0, active=0, deleted=0, failed=0)

    def fake_build_client(*args: object, **kwargs: object) -> ProbeRedis:
        del args
        captured.update(kwargs)
        return ProbeRedis()

    class ProbeCheckpointerResource:
        async def __aenter__(self) -> object:
            return object()

        async def __aexit__(self, *args: object) -> None:
            del args

    settings = Settings(
        checkpoint_backend=CheckpointBackend.POSTGRES,
        agent_postgres_dsn="postgresql://agent:password@postgres/aicare_agent",
        agent_redis_url="redis://aicare_agent:data-secret@unused:6379/0",
        redis_mode=RedisMode.SENTINEL,
        redis_sentinels="sentinel-a:26379,sentinel-b:26380,sentinel-c:26381",
        redis_master_name="aicare-agent-master",
        redis_sentinel_username="aicare_sentinel",
        redis_sentinel_password=SecretStr("sentinel-secret"),
        _env_file=None,
    )
    monkeypatch.setattr(cleanup_checkpoints, "get_settings", lambda: settings)
    monkeypatch.setattr(cleanup_checkpoints, "build_redis_client", fake_build_client)
    monkeypatch.setattr(
        cleanup_checkpoints,
        "checkpointer_resource",
        lambda _: ProbeCheckpointerResource(),
    )
    monkeypatch.setattr(cleanup_checkpoints, "RedisRunStore", ProbeRunStore)
    monkeypatch.setattr(cleanup_checkpoints, "PostgresCheckpointCatalog", lambda _: object())
    monkeypatch.setattr(cleanup_checkpoints, "CheckpointCleanupService", ProbeService)

    await cleanup_checkpoints.run_cleanup(False)

    assert captured["redis_mode"] is RedisMode.SENTINEL
    assert captured["sentinel_endpoints"] == (
        ("sentinel-a", 26379),
        ("sentinel-b", 26380),
        ("sentinel-c", 26381),
    )
