from contextlib import asynccontextmanager
from selectors import SelectSelector
from typing import Self

import pytest

from aicare_agent_service.config import CheckpointBackend, Environment, Settings
from aicare_agent_service.persistence import init_db


def postgres_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": Environment.TEST,
        "checkpoint_backend": CheckpointBackend.POSTGRES,
        "agent_postgres_dsn": "postgresql://agent:password@db/agent",
        "checkpoint_encryption_key": "0123456789abcdef0123456789abcdef",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.mark.asyncio
async def test_initialize_database_uses_resource_and_runs_setup_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class ProbeSaver:
        async def setup(self) -> None:
            events.append("setup")

    @asynccontextmanager
    async def fake_resource(settings: Settings):
        assert settings.checkpoint_backend is CheckpointBackend.POSTGRES
        events.append("enter")
        yield ProbeSaver()
        events.append("exit")

    monkeypatch.setattr(init_db, "checkpointer_resource", fake_resource)

    await init_db.initialize_database(postgres_settings())

    assert events == ["enter", "setup", "exit"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "settings",
    [
        postgres_settings(checkpoint_backend=CheckpointBackend.MEMORY),
        postgres_settings(agent_postgres_dsn=None),
        postgres_settings(checkpoint_encryption_key=None),
    ],
)
async def test_initialize_database_rejects_incomplete_or_memory_configuration(
    settings: Settings,
) -> None:
    with pytest.raises(ValueError):
        await init_db.initialize_database(settings)


def test_init_command_hides_dsn_and_raw_failure_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret_dsn = "postgresql://agent:private-password@private-host/aicare_agent"

    async def failing_initialize(settings: Settings) -> None:
        del settings
        raise RuntimeError(secret_dsn)

    monkeypatch.setattr(
        init_db, "get_settings", lambda: postgres_settings(agent_postgres_dsn=secret_dsn)
    )
    monkeypatch.setattr(init_db, "initialize_database", failing_initialize)

    assert init_db.main() == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "PostgreSQL Checkpointer初始化失败：无法完成初始化\n"
    assert "private" not in captured.err


def test_init_command_uses_selector_event_loop_runner_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    async def successful_initialize(settings: Settings) -> None:
        del settings
        events.append("initialize")

    class ProbeRunner:
        def __init__(self, *, loop_factory: object) -> None:
            self._loop_factory = loop_factory

        def __enter__(self) -> Self:
            events.append("enter")
            return self

        def __exit__(self, *args: object) -> None:
            events.append("exit")

        def run(self, coroutine: object) -> None:
            loop = self._loop_factory()  # type: ignore[operator]
            try:
                assert isinstance(loop, init_db.asyncio.SelectorEventLoop)
                assert isinstance(loop._selector, SelectSelector)
                events.append("run")
            finally:
                loop.close()
                coroutine.close()  # type: ignore[union-attr]

    monkeypatch.setattr(init_db.sys, "platform", "win32")
    monkeypatch.setattr(init_db, "get_settings", postgres_settings)
    monkeypatch.setattr(init_db, "initialize_database", successful_initialize)
    monkeypatch.setattr(init_db.asyncio, "Runner", ProbeRunner)

    assert init_db.main() == 0
    assert events == ["enter", "run", "exit"]


def test_init_command_keeps_asyncio_run_outside_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def fake_run(coroutine: object) -> None:
        calls.append(coroutine)
        coroutine.close()  # type: ignore[union-attr]

    monkeypatch.setattr(init_db.sys, "platform", "linux")
    monkeypatch.setattr(init_db, "get_settings", postgres_settings)
    monkeypatch.setattr(init_db.asyncio, "run", fake_run)

    assert init_db.main() == 0
    assert len(calls) == 1
