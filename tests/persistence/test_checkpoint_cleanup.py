import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import pytest

from aicare_agent_service.persistence.checkpoint_cleanup import (
    CheckpointCandidate,
    CheckpointCleanupService,
    PostgresCheckpointCatalog,
    checkpoint_timestamp,
)


class FakeCatalog:
    def __init__(self, candidates: list[CheckpointCandidate]) -> None:
        self.candidates = candidates
        self.deleted: list[str] = []
        self.failures: set[str] = set()
        self.delays: dict[str, float] = {}

    async def list_candidates(
        self,
        cutoff: datetime,
        limit: int,
    ) -> list[CheckpointCandidate]:
        return [candidate for candidate in self.candidates if candidate.updated_at < cutoff][:limit]

    async def delete_thread(self, thread_id: str) -> None:
        await asyncio.sleep(self.delays.get(thread_id, 0))
        if thread_id in self.failures:
            raise RuntimeError("database detail must not escape")
        self.deleted.append(thread_id)
        self.candidates = [item for item in self.candidates if item.thread_id != thread_id]


class FakeActivityStore:
    def __init__(self, active: set[str] | None = None) -> None:
        self.active = active or set()
        self.guards: set[str] = set()

    async def is_conversation_active(self, conversation_id: str) -> bool:
        return conversation_id in self.active

    async def acquire_cleanup_guard(self, conversation_id: str) -> str | None:
        if conversation_id in self.active or conversation_id in self.guards:
            return None
        self.guards.add(conversation_id)
        return f"guard-{conversation_id}"

    async def release_cleanup_guard(self, conversation_id: str, token: str) -> None:
        assert token == f"guard-{conversation_id}"
        self.guards.discard(conversation_id)


class FakeCursor:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows
        self.executed: list[tuple[str, tuple[int]]] = []

    async def execute(self, query: str, params: tuple[int]) -> None:
        self.executed.append((query, params))

    async def fetchall(self) -> list[dict[str, str]]:
        return self.rows


class FakeSaver:
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.cursor = FakeCursor(rows)
        self.deleted: list[str] = []

    @asynccontextmanager
    async def _cursor(self):
        yield self.cursor

    async def adelete_thread(self, thread_id: str) -> None:
        self.deleted.append(thread_id)


def candidate(thread_id: str, *, age_days: int) -> CheckpointCandidate:
    return CheckpointCandidate(
        thread_id=thread_id,
        checkpoint_id=f"checkpoint-{thread_id}",
        updated_at=datetime.now(UTC) - timedelta(days=age_days),
    )


def test_checkpoint_timestamp_accepts_langgraph_uuid6_and_uuid7() -> None:
    uuid6_id = "1f196d3a-dc99-623d-bffe-bdb9f84b014d"
    uuid7_id = "019ff97c-fb5b-75b0-8d69-327d424acbc3"

    assert checkpoint_timestamp(uuid6_id).year == 2026
    assert checkpoint_timestamp(uuid7_id).year == 2026


@pytest.mark.parametrize("checkpoint_id", ["not-a-uuid", "550e8400-e29b-41d4-a716-446655440000"])
def test_checkpoint_timestamp_rejects_unknown_or_non_time_uuid(checkpoint_id: str) -> None:
    with pytest.raises(ValueError, match="checkpoint ID"):
        checkpoint_timestamp(checkpoint_id)


@pytest.mark.asyncio
async def test_postgres_catalog_reads_metadata_only_and_skips_unparseable_ids() -> None:
    saver = FakeSaver(
        [
            {
                "thread_id": "valid",
                "checkpoint_id": "1f196d3a-dc99-623d-bffe-bdb9f84b014d",
            },
            {"thread_id": "unknown", "checkpoint_id": "legacy-id"},
        ]
    )
    catalog = PostgresCheckpointCatalog(saver, scan_limit=1000)

    candidates = await catalog.list_candidates(datetime(2027, 1, 1, tzinfo=UTC), limit=10)

    assert [item.thread_id for item in candidates] == ["valid"]
    query, params = saver.cursor.executed[0]
    selected = query.lower().split("from", maxsplit=1)[0]
    assert "checkpoint," not in selected
    assert "metadata" not in query.lower()
    assert params == (1000,)


@pytest.mark.asyncio
async def test_postgres_catalog_delegates_thread_delete_to_official_saver() -> None:
    saver = FakeSaver([])
    catalog = PostgresCheckpointCatalog(saver)

    await catalog.delete_thread("conversation-001")

    assert saver.deleted == ["conversation-001"]


@pytest.mark.asyncio
async def test_dry_run_reports_eligible_candidate_without_deleting() -> None:
    catalog = FakeCatalog([candidate("old", age_days=10), candidate("new", age_days=1)])
    service = CheckpointCleanupService(catalog, FakeActivityStore())

    result = await service.run(
        cutoff=datetime.now(UTC) - timedelta(days=7),
        limit=100,
        dry_run=True,
    )

    assert result.scanned == 1
    assert result.eligible == 1
    assert result.deleted == 0
    assert result.failed == 0
    assert catalog.deleted == []
    assert service._activity_store.guards == set()


@pytest.mark.asyncio
async def test_active_conversation_is_never_deleted() -> None:
    catalog = FakeCatalog([candidate("active", age_days=10)])
    service = CheckpointCleanupService(catalog, FakeActivityStore({"active"}))

    result = await service.run(
        cutoff=datetime.now(UTC) - timedelta(days=7),
        limit=100,
        dry_run=False,
    )

    assert result.active == 1
    assert result.deleted == 0
    assert catalog.deleted == []


@pytest.mark.asyncio
async def test_apply_is_idempotent_after_successful_delete() -> None:
    catalog = FakeCatalog([candidate("old", age_days=10)])
    service = CheckpointCleanupService(catalog, FakeActivityStore())
    cutoff = datetime.now(UTC) - timedelta(days=7)

    first = await service.run(cutoff=cutoff, limit=100, dry_run=False)
    second = await service.run(cutoff=cutoff, limit=100, dry_run=False)

    assert first.deleted == 1
    assert second.scanned == 0
    assert catalog.deleted == ["old"]
    assert service._activity_store.guards == set()


@pytest.mark.asyncio
async def test_partial_delete_failure_continues_and_can_retry() -> None:
    catalog = FakeCatalog([candidate("failed", age_days=10), candidate("ok", age_days=10)])
    catalog.failures.add("failed")
    service = CheckpointCleanupService(catalog, FakeActivityStore())
    cutoff = datetime.now(UTC) - timedelta(days=7)

    first = await service.run(cutoff=cutoff, limit=100, dry_run=False)
    catalog.failures.clear()
    second = await service.run(cutoff=cutoff, limit=100, dry_run=False)

    assert first.deleted == 1
    assert first.failed == 1
    assert second.deleted == 1
    assert catalog.deleted == ["ok", "failed"]
    assert service._activity_store.guards == set()


@pytest.mark.asyncio
async def test_delete_timeout_is_failed_and_retriable_before_guard_expires() -> None:
    catalog = FakeCatalog([candidate("slow", age_days=10)])
    catalog.delays["slow"] = 0.05
    activity = FakeActivityStore()
    service = CheckpointCleanupService(catalog, activity, delete_timeout_seconds=0.01)
    cutoff = datetime.now(UTC) - timedelta(days=7)

    first = await service.run(cutoff=cutoff, limit=100, dry_run=False)
    catalog.delays.clear()
    second = await service.run(cutoff=cutoff, limit=100, dry_run=False)

    assert first.failed == 1
    assert second.deleted == 1
    assert activity.guards == set()


@pytest.mark.asyncio
async def test_cleanup_guard_blocks_candidate_when_another_cleaner_owns_it() -> None:
    catalog = FakeCatalog([candidate("guarded", age_days=10)])
    activity = FakeActivityStore()
    activity.guards.add("guarded")
    service = CheckpointCleanupService(catalog, activity)

    result = await service.run(
        cutoff=datetime.now(UTC) - timedelta(days=7),
        limit=100,
        dry_run=False,
    )

    assert result.active == 1
    assert catalog.deleted == []
