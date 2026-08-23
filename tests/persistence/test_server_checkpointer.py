"""验证Agent Server只能装配生产PostgreSQL Checkpointer资源。"""

from contextlib import asynccontextmanager

import pytest

from aicare_agent_service.config import CheckpointBackend, Environment, Settings
from aicare_agent_service.persistence import server_checkpointer


@pytest.mark.asyncio
async def test_server_checkpointer_rejects_memory_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server_checkpointer,
        "get_settings",
        lambda: Settings(
            environment=Environment.TEST,
            checkpoint_backend=CheckpointBackend.MEMORY,
            _env_file=None,
        ),
    )

    with pytest.raises(ValueError, match="PostgreSQL"):
        async with server_checkpointer.generate_server_checkpointer():
            pass


@pytest.mark.asyncio
async def test_server_checkpointer_yields_application_postgres_resource(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        environment=Environment.TEST,
        checkpoint_backend=CheckpointBackend.POSTGRES,
        agent_postgres_dsn="postgresql://agent:password@localhost:5432/agent",
        checkpoint_encryption_key="0123456789abcdef0123456789abcdef",
        _env_file=None,
    )
    sentinel = object()

    @asynccontextmanager
    async def fake_resource(received: Settings):
        assert received is settings
        yield sentinel

    monkeypatch.setattr(server_checkpointer, "get_settings", lambda: settings)
    monkeypatch.setattr(server_checkpointer, "checkpointer_resource", fake_resource)

    async with server_checkpointer.generate_server_checkpointer() as saver:
        assert saver is sentinel
