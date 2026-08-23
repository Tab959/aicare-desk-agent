import asyncio
from dataclasses import dataclass

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import SecretStr

from aicare_agent_service.contracts.adapters import adapt_run_request
from aicare_agent_service.contracts.agent_run import AgentRunRequest
from aicare_agent_service.contracts.common import CONTRACT_HEADER_VERSION
from aicare_agent_service.contracts.decisions import MessageRole, SafeConversationMessage
from aicare_agent_service.contracts.events import AgentEvent, AgentEventType
from aicare_agent_service.graph.state import CustomerServiceState
from aicare_agent_service.persistence.identity import build_thread_config
from aicare_agent_service.persistence.lifecycle_runner import (
    AgentRunLifecycle,
    RunAlreadyInProgressError,
    RunFinalStateError,
    RunReplayUnavailableError,
    RunRequestConflictError,
    terminal_event_digest,
)
from aicare_agent_service.persistence.models import (
    RunBeginOutcome,
    RunBeginResult,
    RunLeaseLostError,
    RunRecord,
    RunStatus,
    RunStoreUnavailableError,
)


def run_request() -> AgentRunRequest:
    return AgentRunRequest.model_validate(
        {
            "tenantId": "tenant-001",
            "customerId": "customer-001",
            "conversationId": "conversation-001",
            "runId": "run-001",
            "triggerMessageId": "message-001",
            "triggerSequence": 12,
            "userMessage": "CDK 无法激活",
            "businessContext": {
                "subject": "CDK激活问题",
                "orderId": "order-001",
                "orderNo": "AD20260001",
                "orderStatus": "PAID",
                "entitlementId": "entitlement-001",
                "entitlementType": "CDK",
                "entitlementStatus": "DELIVERED",
            },
        }
    )


@dataclass
class Snapshot:
    values: CustomerServiceState
    config: RunnableConfig


class FakeGraph:
    def __init__(self, *, delay: float = 0, final_answer: str = "请检查激活区域。") -> None:
        self.delay = delay
        self.final_answer = final_answer
        self.invoke_inputs: list[CustomerServiceState | None] = []
        self.invoke_count = 0
        self.cancelled = False
        self.snapshot: Snapshot | None = None
        self.state_error: Exception | None = None

    async def ainvoke(
        self,
        input_state: CustomerServiceState | None,
        config: RunnableConfig,
        *,
        context: object | None = None,
    ) -> CustomerServiceState:
        del context
        self.invoke_inputs.append(input_state)
        self.invoke_count += 1
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if input_state is None:
            assert self.snapshot is not None
            state = dict(self.snapshot.values)
        else:
            state = dict(input_state)
        state["final_answer"] = self.final_answer
        self.snapshot = Snapshot(
            values=state,
            config={
                "configurable": {
                    "thread_id": config["configurable"]["thread_id"],
                    "checkpoint_id": "checkpoint-001",
                }
            },
        )
        return state

    async def aget_state(self, config: RunnableConfig) -> Snapshot:
        del config
        if self.state_error is not None:
            raise self.state_error
        if self.snapshot is None:
            return Snapshot(values={}, config={"configurable": {}})
        return self.snapshot


class FakeRunStore:
    def __init__(self, outcome: RunBeginOutcome = RunBeginOutcome.STARTED) -> None:
        self.outcome = outcome
        self.record: RunRecord | None = None
        self.completed: list[tuple[str, str]] = []
        self.failed: list[str] = []
        self.cancelled = 0
        self.renewed = 0
        self.cancel_requested = False
        self.complete_error: Exception | None = None
        self.renew_error: Exception | None = None

    async def begin(self, request: AgentRunRequest, request_digest: str) -> RunBeginResult:
        del request_digest
        return RunBeginResult(
            outcome=self.outcome,
            record=self.record,
            lease_token=(
                SecretStr("lease-token")
                if self.outcome in {RunBeginOutcome.STARTED, RunBeginOutcome.RESUMED}
                else None
            ),
        )

    async def get(self, run_id: str) -> RunRecord | None:
        del run_id
        return self.record

    async def renew_lease(self, run_id: str, lease_token: str) -> None:
        del run_id, lease_token
        if self.renew_error is not None:
            raise self.renew_error
        self.renewed += 1

    async def complete(
        self,
        run_id: str,
        lease_token: str,
        checkpoint_id: str,
        final_digest: str,
    ) -> None:
        del run_id, lease_token
        if self.complete_error is not None:
            raise self.complete_error
        self.completed.append((checkpoint_id, final_digest))

    async def fail(self, run_id: str, lease_token: str, error_code: str) -> None:
        del run_id, lease_token
        self.failed.append(error_code)

    async def cancel(self, run_id: str, lease_token: str) -> None:
        del run_id, lease_token
        self.cancelled += 1

    async def request_cancel(self, run_id: str) -> None:
        del run_id
        self.cancel_requested = True

    async def is_cancel_requested(self, run_id: str) -> bool:
        del run_id
        return self.cancel_requested


async def collect_event(events: list[AgentEvent], event: AgentEvent) -> None:
    events.append(event)


@pytest.mark.asyncio
async def test_new_run_executes_with_conversation_thread_and_completes_after_checkpoint() -> None:
    store = FakeRunStore()
    graph = FakeGraph()
    events: list[AgentEvent] = []
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    result = await lifecycle.execute(
        run_request(),
        graph,
        emit=lambda event: collect_event(events, event),
    )

    assert result.status is RunStatus.COMPLETED
    assert graph.invoke_inputs[0] is not None
    assert graph.snapshot is not None
    assert graph.snapshot.config["configurable"]["thread_id"] == "conversation-001"
    assert [event.type for event in events] == [
        AgentEventType.RUN_ACCEPTED,
        AgentEventType.FINAL_MESSAGE,
    ]
    assert store.completed == [("checkpoint-001", terminal_event_digest(events[-1]))]


@pytest.mark.asyncio
async def test_new_run_accepts_java_supplied_checkpoint_safe_recovery_state() -> None:
    request = run_request()
    recovery_state = adapt_run_request(
        CONTRACT_HEADER_VERSION,
        request.model_dump(by_alias=True, mode="json"),
    )
    recovery_state["safe_history"] = [
        SafeConversationMessage(
            message_id="message-before",
            sequence=11,
            role=MessageRole.AI,
            content="此前已完成基础排查。",
        ),
        *recovery_state["safe_history"],
    ]
    store = FakeRunStore()
    graph = FakeGraph()
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    result = await lifecycle.execute(request, graph, initial_state=recovery_state)

    assert result.status is RunStatus.COMPLETED
    assert graph.invoke_inputs[0] is recovery_state


@pytest.mark.asyncio
async def test_recovery_state_with_different_java_identity_is_rejected() -> None:
    request = run_request()
    recovery_state = adapt_run_request(
        CONTRACT_HEADER_VERSION,
        request.model_dump(by_alias=True, mode="json"),
    )
    recovery_state["identity"] = recovery_state["identity"].model_copy(
        update={"run_id": "run-other"}
    )
    store = FakeRunStore()
    graph = FakeGraph()
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    with pytest.raises(RunRequestConflictError):
        await lifecycle.execute(request, graph, initial_state=recovery_state)

    assert graph.invoke_count == 0


@pytest.mark.asyncio
async def test_resumed_run_invokes_graph_with_none_to_continue_latest_checkpoint() -> None:
    store = FakeRunStore(RunBeginOutcome.RESUMED)
    graph = FakeGraph()
    graph.snapshot = Snapshot(
        values={"final_answer": "旧快照"},
        config={"configurable": {"thread_id": "conversation-001", "checkpoint_id": "old"}},
    )
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    result = await lifecycle.execute(run_request(), graph)

    assert result.status is RunStatus.COMPLETED
    assert graph.invoke_inputs == [None]


@pytest.mark.asyncio
async def test_completed_run_rebuilds_terminal_from_recorded_checkpoint_without_graph_invoke() -> (
    None
):
    request = run_request()
    store = FakeRunStore(RunBeginOutcome.REPLAY_COMPLETED)
    graph = FakeGraph()
    graph.snapshot = Snapshot(
        values={
            "identity": adapt_run_request(
                CONTRACT_HEADER_VERSION,
                request.model_dump(by_alias=True, mode="json"),
            )["identity"],
            "final_answer": "已完成回答",
        },
        config={
            "configurable": {"thread_id": request.conversation_id, "checkpoint_id": "cp-final"}
        },
    )
    expected_event = AgentRunLifecycle.build_terminal_event(request, graph.snapshot.values, 2)
    store.record = RunRecord.model_validate(
        {
            "tenant_id": request.tenant_id,
            "customer_id": request.customer_id,
            "conversation_id": request.conversation_id,
            "run_id": request.run_id,
            "trigger_message_id": request.trigger_message_id,
            "trigger_sequence": request.trigger_sequence,
            "request_digest": "a" * 64,
            "status": "COMPLETED",
            "checkpoint_id": "cp-final",
            "final_digest": terminal_event_digest(expected_event),
            "started_at": "2026-08-13T00:00:00Z",
            "updated_at": "2026-08-13T00:00:01Z",
            "completed_at": "2026-08-13T00:00:01Z",
        }
    )
    events: list[AgentEvent] = []
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    result = await lifecycle.execute(
        request,
        graph,
        emit=lambda event: collect_event(events, event),
    )

    assert result.status is RunStatus.COMPLETED
    assert graph.invoke_count == 0
    assert [event.type for event in events] == [
        AgentEventType.RUN_ACCEPTED,
        AgentEventType.FINAL_MESSAGE,
    ]


@pytest.mark.asyncio
async def test_completed_run_with_missing_checkpoint_fails_explicitly_without_rerun() -> None:
    store = FakeRunStore(RunBeginOutcome.REPLAY_COMPLETED)
    store.record = RunRecord.model_validate(
        {
            "tenant_id": "tenant-001",
            "customer_id": "customer-001",
            "conversation_id": "conversation-001",
            "run_id": "run-001",
            "trigger_message_id": "message-001",
            "trigger_sequence": 12,
            "request_digest": "a" * 64,
            "status": "COMPLETED",
            "checkpoint_id": "missing",
            "final_digest": "b" * 64,
            "started_at": "2026-08-13T00:00:00Z",
            "updated_at": "2026-08-13T00:00:01Z",
            "completed_at": "2026-08-13T00:00:01Z",
        }
    )
    graph = FakeGraph()
    events: list[AgentEvent] = []
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    with pytest.raises(RunReplayUnavailableError):
        await lifecycle.execute(
            run_request(),
            graph,
            initial_state=adapt_run_request(
                CONTRACT_HEADER_VERSION,
                run_request().model_dump(by_alias=True, mode="json"),
            ),
            emit=lambda event: collect_event(events, event),
        )

    assert graph.invoke_count == 0
    assert events == []


@pytest.mark.asyncio
async def test_completed_replay_with_malformed_checkpoint_is_unavailable_without_events() -> None:
    request = run_request()
    store = FakeRunStore(RunBeginOutcome.REPLAY_COMPLETED)
    graph = FakeGraph()
    graph.snapshot = Snapshot(
        values={
            "identity": adapt_run_request(
                CONTRACT_HEADER_VERSION,
                request.model_dump(by_alias=True, mode="json"),
            )["identity"],
            "final_answer": "已完成回答",
        },
        config={"configurable": {}},
    )
    expected = AgentRunLifecycle.build_terminal_event(request, graph.snapshot.values, 2)
    store.record = RunRecord.model_validate(
        {
            "tenant_id": request.tenant_id,
            "customer_id": request.customer_id,
            "conversation_id": request.conversation_id,
            "run_id": request.run_id,
            "trigger_message_id": request.trigger_message_id,
            "trigger_sequence": request.trigger_sequence,
            "request_digest": "a" * 64,
            "status": "COMPLETED",
            "checkpoint_id": "cp-final",
            "final_digest": terminal_event_digest(expected),
            "started_at": "2026-08-13T00:00:00Z",
            "updated_at": "2026-08-13T00:00:01Z",
            "completed_at": "2026-08-13T00:00:01Z",
        }
    )
    events: list[AgentEvent] = []
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    with pytest.raises(RunReplayUnavailableError):
        await lifecycle.execute(
            request,
            graph,
            emit=lambda event: collect_event(events, event),
        )

    assert events == []


@pytest.mark.asyncio
async def test_completed_replay_propagates_checkpoint_store_outage_without_events() -> None:
    request = run_request()
    store = FakeRunStore(RunBeginOutcome.REPLAY_COMPLETED)
    graph = FakeGraph()
    graph.state_error = ConnectionError("temporary checkpoint outage")
    store.record = RunRecord.model_validate(
        {
            "tenant_id": request.tenant_id,
            "customer_id": request.customer_id,
            "conversation_id": request.conversation_id,
            "run_id": request.run_id,
            "trigger_message_id": request.trigger_message_id,
            "trigger_sequence": request.trigger_sequence,
            "request_digest": "a" * 64,
            "status": "COMPLETED",
            "checkpoint_id": "cp-final",
            "final_digest": "b" * 64,
            "started_at": "2026-08-13T00:00:00Z",
            "updated_at": "2026-08-13T00:00:01Z",
            "completed_at": "2026-08-13T00:00:01Z",
        }
    )
    events: list[AgentEvent] = []
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    with pytest.raises(ConnectionError, match="temporary checkpoint outage"):
        await lifecycle.execute(
            request,
            graph,
            emit=lambda event: collect_event(events, event),
        )

    assert events == []


@pytest.mark.asyncio
async def test_completed_replay_rejects_checkpoint_identity_drift() -> None:
    request = run_request()
    store = FakeRunStore(RunBeginOutcome.REPLAY_COMPLETED)
    graph = FakeGraph()
    identity = adapt_run_request(
        CONTRACT_HEADER_VERSION,
        request.model_dump(by_alias=True, mode="json"),
    )["identity"]
    graph.snapshot = Snapshot(
        values={
            "identity": identity.model_copy(update={"run_id": "run-other"}),
            "final_answer": "已完成回答",
        },
        config={"configurable": {"thread_id": request.conversation_id, "checkpoint_id": "cp"}},
    )
    expected = AgentRunLifecycle.build_terminal_event(request, graph.snapshot.values, 2)
    store.record = RunRecord.model_validate(
        {
            "tenant_id": request.tenant_id,
            "customer_id": request.customer_id,
            "conversation_id": request.conversation_id,
            "run_id": request.run_id,
            "trigger_message_id": request.trigger_message_id,
            "trigger_sequence": request.trigger_sequence,
            "request_digest": "a" * 64,
            "status": "COMPLETED",
            "checkpoint_id": "cp",
            "final_digest": terminal_event_digest(expected),
            "started_at": "2026-08-13T00:00:00Z",
            "updated_at": "2026-08-13T00:00:01Z",
            "completed_at": "2026-08-13T00:00:01Z",
        }
    )
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    with pytest.raises(RunReplayUnavailableError):
        await lifecycle.execute(request, graph)


@pytest.mark.asyncio
async def test_cooperative_cancel_stops_graph_and_never_completes_final() -> None:
    store = FakeRunStore()
    graph = FakeGraph(delay=1)
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)
    execution = asyncio.create_task(lifecycle.execute(run_request(), graph))
    await asyncio.sleep(0.02)

    await lifecycle.request_cancel("run-001")
    result = await execution

    assert result.status is RunStatus.CANCELLED
    assert graph.cancelled is True
    assert store.cancelled == 1
    assert store.completed == []


@pytest.mark.asyncio
async def test_timeout_cancels_graph_emits_failure_and_never_completes_final() -> None:
    store = FakeRunStore()
    graph = FakeGraph(delay=1)
    events: list[AgentEvent] = []
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=0.03)

    result = await lifecycle.execute(
        run_request(),
        graph,
        emit=lambda event: collect_event(events, event),
    )

    assert result.status is RunStatus.FAILED
    assert graph.cancelled is True
    assert store.failed == ["MODEL_TIMEOUT"]
    assert store.completed == []
    assert events[-1].type is AgentEventType.RUN_FAILED


@pytest.mark.asyncio
async def test_in_progress_run_is_rejected_without_invoking_graph() -> None:
    store = FakeRunStore(RunBeginOutcome.IN_PROGRESS)
    graph = FakeGraph()
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    with pytest.raises(RunAlreadyInProgressError):
        await lifecycle.execute(run_request(), graph)

    assert graph.invoke_count == 0


@pytest.mark.asyncio
async def test_slow_graph_renews_lease_and_emits_heartbeat_before_final() -> None:
    store = FakeRunStore()
    graph = FakeGraph(delay=0.03)
    events: list[AgentEvent] = []
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    result = await lifecycle.execute(
        run_request(),
        graph,
        emit=lambda event: collect_event(events, event),
    )

    assert result.status is RunStatus.COMPLETED
    assert store.renewed >= 1
    assert AgentEventType.RUN_HEARTBEAT in [event.type for event in events]
    assert events[-1].type is AgentEventType.FINAL_MESSAGE


@pytest.mark.asyncio
async def test_run_store_failure_during_renew_is_not_misreported_as_model_failure() -> None:
    store = FakeRunStore()
    store.renew_error = RunStoreUnavailableError()
    graph = FakeGraph(delay=1)
    events: list[AgentEvent] = []
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    with pytest.raises(RunStoreUnavailableError):
        await lifecycle.execute(
            run_request(),
            graph,
            emit=lambda event: collect_event(events, event),
        )

    assert graph.cancelled is True
    assert store.failed == []
    assert [event.type for event in events] == [AgentEventType.RUN_ACCEPTED]


@pytest.mark.asyncio
async def test_lost_lease_during_complete_never_emits_final() -> None:
    store = FakeRunStore()
    store.complete_error = RunLeaseLostError()
    graph = FakeGraph()
    events: list[AgentEvent] = []
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    with pytest.raises(RunLeaseLostError):
        await lifecycle.execute(
            run_request(),
            graph,
            emit=lambda event: collect_event(events, event),
        )

    assert [event.type for event in events] == [AgentEventType.RUN_ACCEPTED]


@pytest.mark.asyncio
async def test_invalid_final_state_is_failed_without_sending_final() -> None:
    store = FakeRunStore()
    graph = FakeGraph(final_answer="")
    events: list[AgentEvent] = []
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    result = await lifecycle.execute(
        run_request(),
        graph,
        emit=lambda event: collect_event(events, event),
    )

    assert result.status is RunStatus.FAILED
    assert store.failed == ["AGENT_FINAL_STATE_INVALID"]


@pytest.mark.asyncio
async def test_checkpoint_store_outage_after_graph_does_not_mark_run_failed() -> None:
    store = FakeRunStore()
    graph = FakeGraph()
    graph.state_error = ConnectionError("temporary checkpoint outage")
    events: list[AgentEvent] = []
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    with pytest.raises(ConnectionError, match="temporary checkpoint outage"):
        await lifecycle.execute(
            run_request(),
            graph,
            emit=lambda event: collect_event(events, event),
        )

    assert store.failed == []
    assert store.completed == []
    assert [event.type for event in events] == [AgentEventType.RUN_ACCEPTED]


def test_terminal_digest_ignores_event_index_but_detects_content_change() -> None:
    request = run_request()
    first = AgentRunLifecycle.build_terminal_event(
        request,
        {"final_answer": "相同回答"},
        2,
    )
    replay = AgentRunLifecycle.build_terminal_event(
        request,
        {"final_answer": "相同回答"},
        9,
    )
    changed = AgentRunLifecycle.build_terminal_event(
        request,
        {"final_answer": "不同回答"},
        2,
    )

    assert terminal_event_digest(first) == terminal_event_digest(replay)
    assert terminal_event_digest(first) != terminal_event_digest(changed)


def test_terminal_event_uses_same_strict_terminal_contract_as_finalize() -> None:
    request = run_request()

    with pytest.raises(RunFinalStateError):
        AgentRunLifecycle.build_terminal_event(
            request,
            {
                "handoff_suggestion": {
                    "reason": "未经契约校验",
                    "priority": "MEDIUM",
                    "summary": "非法字典",
                }
            },
            2,
        )


@pytest.mark.asyncio
async def test_real_langgraph_resumes_from_latest_interrupted_checkpoint() -> None:
    request = run_request()
    checkpointer = InMemorySaver()
    calls: list[str] = []

    def prepare(_: CustomerServiceState) -> dict[str, str]:
        calls.append("prepare")
        return {"conversation_summary": "已准备"}

    def answer(_: CustomerServiceState) -> dict[str, str]:
        calls.append("answer")
        return {"final_answer": "从checkpoint继续完成"}

    builder = StateGraph(CustomerServiceState)
    builder.add_node("prepare", prepare)
    builder.add_node("answer", answer)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "answer")
    builder.add_edge("answer", END)
    interrupted_graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_after=["prepare"],
    )
    config = build_thread_config(request)
    initial_state = adapt_run_request(
        CONTRACT_HEADER_VERSION,
        request.model_dump(by_alias=True, mode="json"),
    )
    await interrupted_graph.ainvoke(initial_state, config)
    assert calls == ["prepare"]

    resumed_graph = builder.compile(checkpointer=checkpointer)
    store = FakeRunStore(RunBeginOutcome.RESUMED)
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    result = await lifecycle.execute(request, resumed_graph)

    assert result.status is RunStatus.COMPLETED
    assert calls == ["prepare", "answer"]
    assert result.events[-1].type is AgentEventType.FINAL_MESSAGE


@pytest.mark.asyncio
async def test_real_langgraph_completed_replay_does_not_call_node_again() -> None:
    request = run_request()
    checkpointer = InMemorySaver()
    calls = 0

    def answer(_: CustomerServiceState) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"final_answer": "只生成一次"}

    builder = StateGraph(CustomerServiceState)
    builder.add_node("answer", answer)
    builder.add_edge(START, "answer")
    builder.add_edge("answer", END)
    graph = builder.compile(checkpointer=checkpointer)
    store = FakeRunStore()
    lifecycle = AgentRunLifecycle(store, heartbeat_seconds=0.01, timeout_seconds=1)

    first = await lifecycle.execute(request, graph)
    checkpoint_id, final_digest = store.completed[0]
    store.outcome = RunBeginOutcome.REPLAY_COMPLETED
    store.record = RunRecord.model_validate(
        {
            "tenant_id": request.tenant_id,
            "customer_id": request.customer_id,
            "conversation_id": request.conversation_id,
            "run_id": request.run_id,
            "trigger_message_id": request.trigger_message_id,
            "trigger_sequence": request.trigger_sequence,
            "request_digest": "a" * 64,
            "status": "COMPLETED",
            "checkpoint_id": checkpoint_id,
            "final_digest": final_digest,
            "started_at": "2026-08-13T00:00:00Z",
            "updated_at": "2026-08-13T00:00:01Z",
            "completed_at": "2026-08-13T00:00:01Z",
        }
    )

    replay = await lifecycle.execute(request, graph)

    assert calls == 1
    assert first.events[-1].type is AgentEventType.FINAL_MESSAGE
    assert replay.events[-1].type is AgentEventType.FINAL_MESSAGE
    assert terminal_event_digest(first.events[-1]) == terminal_event_digest(replay.events[-1])
