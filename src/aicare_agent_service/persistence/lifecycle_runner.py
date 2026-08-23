"""确定性编排 Agent run 的 Redis 幂等/租约与 LangGraph checkpoint。

执行顺序为：校验Java身份→Redis begin→发送RUN_ACCEPTED→首次调用或checkpoint恢复→
心跳续租/取消/超时→读取最终checkpoint→原子提交Redis终态→发送唯一终态事件。
Python只生成事件和建议，不修改Java会话状态；Redis ledger与PostgreSQL checkpoint职责分离。
"""

# asyncio负责并发运行图任务、周期等待、超时和安全取消。
import asyncio

# hashlib/json生成不受临时eventIndex影响的终态语义摘要。
import hashlib
import json

# Awaitable/Callable定义异步事件sink；Mapping描述只读状态映射。
from collections.abc import Awaitable, Callable, Mapping

# dataclass定义不可变生命周期结果。
from dataclasses import dataclass

# Any兼容LangGraph StateSnapshot等第三方动态类型；Protocol声明最小图接口。
from typing import Any, Protocol

from langchain_core.runnables import RunnableConfig

from aicare_agent_service.contracts.adapters import adapt_run_request
from aicare_agent_service.contracts.agent_run import AgentRunRequest
from aicare_agent_service.contracts.common import CONTRACT_HEADER_VERSION
from aicare_agent_service.contracts.decisions import EscalationSuggestion, HandoffSuggestion
from aicare_agent_service.contracts.events import (
    AgentEvent,
    AgentEventType,
    EscalationRequestedEvent,
    FinalMessageEvent,
    HandoffRequestedEvent,
    RunAcceptedEvent,
    RunFailedEvent,
    RunHeartbeatEvent,
)
from aicare_agent_service.graph.state import AgentIdentity, CustomerServiceState
from aicare_agent_service.nodes.finalize import (
    FinalStateValidationError,
    TerminalKind,
    validate_terminal_state,
)
from aicare_agent_service.persistence.identity import (
    build_thread_config,
    canonical_request_digest,
)
from aicare_agent_service.persistence.models import (
    RunBeginOutcome,
    RunLeaseLostError,
    RunRecord,
    RunStatus,
    RunStoreError,
)
from aicare_agent_service.persistence.run_store import RunStore

# 事件sink类型别名：接收一个AgentEvent，调用后返回可await且最终值为None的对象。
EventSink = Callable[[AgentEvent], Awaitable[None]]


class LifecycleGraph(Protocol):
    """生命周期执行所需的最小异步LangGraph接口。"""

    async def ainvoke(
        # self由具体编译图实例自动传入。
        self,
        # 首次执行传安全初始状态，恢复执行传None。
        input_state: CustomerServiceState | None,
        # config必须包含由Java conversationId建立的thread_id。
        config: RunnableConfig,
        # ``*``让context必须按名称传入。
        *,
        # context承载模型Provider和Java客户端等不可checkpoint依赖。
        context: object | None = None,
    ) -> CustomerServiceState:
        """异步执行或恢复图并返回最终状态。"""
        ...

    async def aget_state(self, config: RunnableConfig) -> Any:
        """读取指定thread/checkpoint的StateSnapshot。"""
        ...


class AgentRunLifecycleError(RuntimeError):
    """不携带请求正文的生命周期稳定错误。"""

    # 子类覆盖code，异常本身不携带请求正文。
    code = "AGENT_RUN_LIFECYCLE_ERROR"


class RunAlreadyInProgressError(AgentRunLifecycleError):
    """同一conversation已有run或恢复者执行。"""

    code = "RUN_ALREADY_IN_PROGRESS"


class RunRequestConflictError(AgentRunLifecycleError):
    """相同runId对应不同请求，或初始状态身份不匹配。"""

    code = "RUN_REQUEST_CONFLICT"


class RunReplayUnavailableError(AgentRunLifecycleError):
    """已完成run的记录/checkpoint/摘要无法安全验证。"""

    code = "RUN_REPLAY_UNAVAILABLE"


class RunFinalStateError(AgentRunLifecycleError):
    """最终checkpoint没有且仅有一个合法终态。"""

    code = "RUN_FINAL_STATE_INVALID"


@dataclass(frozen=True, slots=True)
class AgentRunLifecycleResult:
    """一次执行对调用者返回的不可变状态和有序事件集合。"""

    # Redis最终提交状态。
    status: RunStatus
    # 本次调用生成的事件；完成重放也会重建RUN_ACCEPTED和终态。
    events: tuple[AgentEvent, ...]


class _EventEmitter:
    """为单次run分配严格递增eventIndex，并可选实时转发事件。"""

    def __init__(self, request: AgentRunRequest, sink: EventSink | None) -> None:
        """保存Java请求、可选sink，并初始化空事件列表。"""
        # request提供每个事件必须携带的Java身份。
        self.request = request
        # sink为None时只在内存收集，便于测试或非流式调用。
        self.sink = sink
        # list保持事件生成顺序，最终会冻结为tuple。
        self.events: list[AgentEvent] = []

    @property
    def next_index(self) -> int:
        """返回下一事件从1开始的序号，不修改列表。"""
        # 已发送数量加1即下一eventIndex。
        return len(self.events) + 1

    async def send(self, event: AgentEvent) -> None:
        """先记录事件，再按存在性异步交给外部sink。"""
        # 先append保证sink观察事件时内部顺序已确定。
        self.events.append(event)
        # sink可选；非None时await其发送完成或传播基础设施异常。
        if self.sink is not None:
            await self.sink(event)

    def base(self, event_type: AgentEventType) -> dict[str, object]:
        """用Java身份和下一索引构造所有线事件共享字段。"""
        # 返回camelCase键，随后交给具体Pydantic事件模型严格校验。
        return {
            "type": event_type,
            "runId": self.request.run_id,
            "conversationId": self.request.conversation_id,
            "triggerMessageId": self.request.trigger_message_id,
            "triggerSequence": self.request.trigger_sequence,
            "eventIndex": self.next_index,
        }


class AgentRunLifecycle:
    """统一执行 begin、图恢复、心跳、取消/超时和 Redis 终态提交。

    类职责：统一编排 Redis 幂等检查 → 图执行 → 心跳续租 → 取消/超时处理
    → 终态提交 → 事件发送。
    """

    def __init__(
        # run_store是Redis实现或测试Fake，提供幂等、租约和终态接口。
        self,
        run_store: RunStore,
        # 以下配置必须按名称传入。
        *,
        heartbeat_seconds: float,
        timeout_seconds: float,
        input_max_chars: int = 8000,
        contract_version: str = CONTRACT_HEADER_VERSION,
    ) -> None:
        """校验时限并保存RunStore、心跳、总超时和共享契约版本。"""
        # 心跳与整轮总超时都必须为正；更复杂关系已由Settings验证。
        if heartbeat_seconds <= 0 or timeout_seconds <= 0 or input_max_chars <= 0:
            raise ValueError("Agent run心跳、超时和输入上限必须为正数")
        # 内部字段不向模型或API直接暴露。
        self._run_store = run_store
        self._heartbeat_seconds = heartbeat_seconds
        self._timeout_seconds = timeout_seconds
        self._input_max_chars = input_max_chars
        self._contract_version = contract_version

    async def request_cancel(self, run_id: str) -> None:
        """把Java取消请求记录为Redis意图，实际终态由执行者提交。"""
        await self._run_store.request_cancel(run_id)

    async def execute(
        # request包含Java生成的conversationId、runId和触发消息身份。
        self,
        request: AgentRunRequest,
        # graph必须满足LifecycleGraph协议，可为真实编译图或测试Fake。
        graph: LifecycleGraph,
        # 以下可选依赖必须按名称传入。
        *,
        # Java可为全新run提供安全初始状态；恢复run会忽略并传None给图。
        initial_state: CustomerServiceState | None = None,
        # context包含不可checkpoint运行时依赖。
        context: object | None = None,
        # emit用于实时转发线事件；None时仍收集结果事件。
        emit: EventSink | None = None,
    ) -> AgentRunLifecycleResult:
        """执行、恢复或重放一个run，并只在Redis终态提交后发送终态事件。"""
        # 若调用者提供初始状态，其identity必须与Java请求派生身份完全一致。
        if initial_state is not None and initial_state.get(
            "identity"
        ) != AgentIdentity.from_request(request):
            raise RunRequestConflictError()
        # 摘要包含契约版本和完整规范请求，但Redis只保存哈希值。
        request_digest = canonical_request_digest(self._contract_version, request)
        # begin通过Lua原子取得开始/恢复/重放/冲突判定。
        begin = await self._run_store.begin(request, request_digest)
        # 同会话已有执行者时不启动第二个图任务。
        if begin.outcome is RunBeginOutcome.IN_PROGRESS:
            raise RunAlreadyInProgressError()
        # 相同runId请求语义不同，拒绝覆盖旧执行。
        if begin.outcome is RunBeginOutcome.CONFLICT:
            raise RunRequestConflictError()

        # emitter从eventIndex=1开始收集，并可选实时转发。
        emitter = _EventEmitter(request, emit)
        # 已完成run不再次执行模型，改为严格验证记录的checkpoint并重建事件。
        if begin.outcome is RunBeginOutcome.REPLAY_COMPLETED:
            return await self._replay_completed(request, begin.record, graph, emitter)

        # 只有取得新/恢复租约后才能首先发送RUN_ACCEPTED。
        await emitter.send(
            RunAcceptedEvent.model_validate(emitter.base(AgentEventType.RUN_ACCEPTED))
        )

        # Pydantic模型已验证STARTED/RESUMED必有token；assert也帮助类型检查器收窄None。
        assert begin.lease_token is not None
        # SecretStr显式取值只用于Redis续租和终态Lua，不写日志。
        lease_token = begin.lease_token.get_secret_value()
        # conversationId由Java生成并映射为唯一LangGraph thread_id。
        config = build_thread_config(request)
        # 默认None表示从现有checkpoint继续，而不是写入新输入。
        input_state = None
        # 只有首次STARTED才建立安全初始状态。
        if begin.outcome is RunBeginOutcome.STARTED:
            # 优先使用调用者提供且已验身份的状态，否则用契约适配器裁剪请求。
            input_state = initial_state or adapt_run_request(
                self._contract_version,
                request.model_dump(by_alias=True, mode="json"),
                input_max_chars=self._input_max_chars,
            )

        # create_task让图与当前协程的心跳/取消监控并发运行。
        graph_task = asyncio.create_task(
            graph.ainvoke(input_state, config, context=context),
        )
        # 任何退出路径都必须正确处理后台graph_task。
        try:
            graph_status = await self._drive_graph(
                request,
                graph_task,
                lease_token,
                emitter,
            )
        except RunLeaseLostError:
            # 租约丢失时先停止本进程图任务，再把错误交给上层；绝不写终态。
            await _cancel_task(graph_task)
            raise
        except RunStoreError:
            # Redis基础设施错误不能伪装为模型失败，同样取消图并原样传播稳定异常。
            await _cancel_task(graph_task)
            raise
        except Exception:  # noqa: BLE001 - 图边界统一映射为稳定错误且不泄漏异常正文。
            # 未知图异常不泄漏正文；先确保任务停止。
            await _cancel_task(graph_task)
            # 当前租约持有者原子提交固定FAILED错误码。
            await self._run_store.fail(request.run_id, lease_token, "AGENT_EXECUTION_FAILED")
            # 构造面向Java/用户安全的失败事件。
            failure = self._failure_event(request, emitter.next_index, "AGENT_EXECUTION_FAILED")
            await emitter.send(failure)
            # 返回已经冻结的事件元组。
            return AgentRunLifecycleResult(RunStatus.FAILED, tuple(emitter.events))

        # _drive_graph已原子提交取消/超时失败时，直接返回对应结果。
        if graph_status is RunStatus.CANCELLED:
            return AgentRunLifecycleResult(RunStatus.CANCELLED, tuple(emitter.events))
        if graph_status is RunStatus.FAILED:
            return AgentRunLifecycleResult(RunStatus.FAILED, tuple(emitter.events))

        # 图刚完成与读取snapshot之间仍可能收到取消；终态提交前再检查一次。
        if await self._run_store.is_cancel_requested(request.run_id):
            await self._run_store.cancel(request.run_id, lease_token)
            return AgentRunLifecycleResult(RunStatus.CANCELLED, tuple(emitter.events))

        # 从相同thread读取LangGraph最终StateSnapshot及checkpoint配置。
        snapshot = await graph.aget_state(config)
        # 终态构建和checkpoint ID解析统一映射为稳定失败。
        try:
            terminal_event = self.build_terminal_event(
                request,
                snapshot.values,
                emitter.next_index,
            )
            checkpoint_id = _checkpoint_id(snapshot.config)
        except Exception:  # noqa: BLE001 - checkpoint终态边界只返回稳定错误。
            # 无/多终态、类型错误或缺checkpoint ID都提交固定错误码。
            await self._run_store.fail(
                request.run_id,
                lease_token,
                "AGENT_FINAL_STATE_INVALID",
            )
            await emitter.send(
                self._failure_event(
                    request,
                    emitter.next_index,
                    "AGENT_FINAL_STATE_INVALID",
                )
            )
            return AgentRunLifecycleResult(RunStatus.FAILED, tuple(emitter.events))
        # 摘要忽略临时eventIndex，只验证终态业务语义。
        digest = terminal_event_digest(terminal_event)
        # 先在Redis原子提交checkpoint ID和摘要，只有成功后才允许向Java发送终态。
        await self._run_store.complete(request.run_id, lease_token, checkpoint_id, digest)
        # Redis确认完成后发送唯一终态事件。
        await emitter.send(terminal_event)
        return AgentRunLifecycleResult(RunStatus.COMPLETED, tuple(emitter.events))

    async def _drive_graph(
        self,
        request: AgentRunRequest,
        graph_task: asyncio.Task[CustomerServiceState],
        lease_token: str,
        emitter: _EventEmitter,
    ) -> RunStatus:
        """在图运行期间周期处理超时、取消、租约续期和心跳。"""
        # 获取当前事件循环的单调时钟，避免系统时间回拨影响超时。
        loop = asyncio.get_running_loop()
        # deadline只在内存中使用，不进入checkpoint或Redis。
        deadline = loop.time() + self._timeout_seconds
        # 循环直到图完成、取消或超时。
        while True:
            # 动态计算剩余总预算。
            remaining = deadline - loop.time()
            # 预算耗尽时取消图、提交失败并发送安全事件。
            if remaining <= 0:
                await _cancel_task(graph_task)
                await self._run_store.fail(request.run_id, lease_token, "MODEL_TIMEOUT")
                await emitter.send(
                    self._failure_event(request, emitter.next_index, "MODEL_TIMEOUT")
                )
                return RunStatus.FAILED

            # 最多等待一个心跳周期，但不会超过总剩余预算。
            done, _ = await asyncio.wait(
                {graph_task},
                timeout=min(self._heartbeat_seconds, remaining),
            )
            # 图任务进入done集合时，result()会重新抛出图内部异常供execute处理。
            if graph_task in done:
                graph_task.result()
                return RunStatus.COMPLETED

            # 每个等待周期先检查Java取消意图。
            if await self._run_store.is_cancel_requested(request.run_id):
                await _cancel_task(graph_task)
                await self._run_store.cancel(request.run_id, lease_token)
                return RunStatus.CANCELLED

            # wait返回后若刚好达到deadline，回到循环顶部走统一超时分支，不再续租。
            if loop.time() >= deadline:
                continue

            # 仍有时间且未取消，验证token并延长Redis lease。
            await self._run_store.renew_lease(request.run_id, lease_token)
            # 续租成功后才发送心跳，避免向Java虚报已失去所有权的run仍活跃。
            await emitter.send(
                RunHeartbeatEvent.model_validate(emitter.base(AgentEventType.RUN_HEARTBEAT))
            )

    async def _replay_completed(
        self,
        request: AgentRunRequest,
        record: RunRecord | None,
        graph: LifecycleGraph,
        emitter: _EventEmitter,
    ) -> AgentRunLifecycleResult:
        """验证已完成记录和指定checkpoint，重建事件而不再次执行图/模型。"""
        # ledger必须完整、为COMPLETED、含checkpoint/摘要且Java身份逐项匹配。
        if (
            record is None
            or record.status is not RunStatus.COMPLETED
            or record.checkpoint_id is None
            or record.final_digest is None
            or not _record_matches_request(record, request)
        ):
            raise RunReplayUnavailableError()
        # 使用相同conversation thread配置，并显式定位Redis记录的checkpoint ID。
        config = build_thread_config(request)
        config["configurable"]["checkpoint_id"] = record.checkpoint_id
        # 只读StateSnapshot，不调用ainvoke，因此不会重复模型或工具副作用。
        snapshot = await graph.aget_state(config)
        # 第一次以固定index=2构造终态，只用于摘要验证。
        try:
            checkpoint_id = _checkpoint_id(snapshot.config)
            terminal = self.build_terminal_event(request, snapshot.values, 2)
        except (RunFinalStateError, KeyError, TypeError, ValueError):
            # 任何checkpoint结构问题统一隐藏细节并拒绝重放。
            raise RunReplayUnavailableError() from None
        # 进一步核对状态非空、不可变身份和实际snapshot checkpoint ID。
        if (
            not snapshot.values
            or snapshot.values.get("identity") != AgentIdentity.from_request(request)
            or checkpoint_id != record.checkpoint_id
        ):
            raise RunReplayUnavailableError()
        # Redis保存摘要必须与重建终态语义一致，防止损坏或状态漂移。
        if terminal_event_digest(terminal) != record.final_digest:
            raise RunReplayUnavailableError()
        # 全部验证成功后才发RUN_ACCEPTED，避免无效重放先对外宣称受理。
        await emitter.send(
            RunAcceptedEvent.model_validate(emitter.base(AgentEventType.RUN_ACCEPTED))
        )
        # 使用当前emitter下一索引重新构建终态，确保线事件序号连续。
        terminal = self.build_terminal_event(request, snapshot.values, emitter.next_index)
        await emitter.send(terminal)
        return AgentRunLifecycleResult(RunStatus.COMPLETED, tuple(emitter.events))

    @staticmethod
    def build_terminal_event(
        request: AgentRunRequest,
        state: Mapping[str, Any],
        event_index: int,
    ) -> FinalMessageEvent | HandoffRequestedEvent | EscalationRequestedEvent:
        """要求状态中恰有最终回答、转人工或升级工单建议之一，并构造线事件。"""
        # 1、复用根图finalize的严格选择器，非法类型和终态冲突统一映射稳定错误。
        try:
            terminal_kind = validate_terminal_state(state)
        except FinalStateValidationError as exc:
            raise RunFinalStateError() from exc
        # 2、三种终态共享Java身份字段和本次临时eventIndex。
        base = {
            "runId": request.run_id,
            "conversationId": request.conversation_id,
            "triggerMessageId": request.trigger_message_id,
            "triggerSequence": request.trigger_sequence,
            "eventIndex": event_index,
        }
        # 3、根据唯一终态枚举构造对应线事件，不执行任何Java业务状态变化。
        if terminal_kind is TerminalKind.FINAL_ANSWER:
            return FinalMessageEvent.model_validate(
                base | {"type": AgentEventType.FINAL_MESSAGE, "content": state["final_answer"]}
            )
        if terminal_kind is TerminalKind.HANDOFF:
            handoff = state["handoff_suggestion"]
            assert isinstance(handoff, HandoffSuggestion)
            return HandoffRequestedEvent.model_validate(
                base
                | {
                    "type": AgentEventType.HANDOFF_REQUESTED,
                    "reason": handoff.reason,
                    "priority": handoff.priority,
                    "summary": handoff.summary,
                }
            )
        escalation = state["escalation_suggestion"]
        assert isinstance(escalation, EscalationSuggestion)
        return EscalationRequestedEvent.model_validate(
            base
            | {
                "type": AgentEventType.ESCALATION_REQUESTED,
                "issueType": escalation.issue_type,
                "reason": escalation.reason,
                "summary": escalation.summary,
            }
        )

    @staticmethod
    def _failure_event(
        request: AgentRunRequest,
        event_index: int,
        error_code: str,
    ) -> RunFailedEvent:
        """把内部稳定错误码映射为固定、可重试的用户安全失败事件。"""
        # 条件表达式只为超时提供更具体提示，其他内部失败统一使用通用文案。
        message = (
            "AI客服响应超时，请稍后重试或转人工客服。"
            if error_code == "MODEL_TIMEOUT"
            else "AI客服暂时无法完成处理，请稍后重试或转人工客服。"
        )
        # Pydantic校验完整camelCase线事件，内部异常正文不会进入响应。
        return RunFailedEvent.model_validate(
            {
                "type": AgentEventType.RUN_FAILED,
                "runId": request.run_id,
                "conversationId": request.conversation_id,
                "triggerMessageId": request.trigger_message_id,
                "triggerSequence": request.trigger_sequence,
                "eventIndex": event_index,
                "errorCode": error_code,
                "retryable": True,
                "userSafeMessage": message,
            }
        )


def terminal_event_digest(event: AgentEvent) -> str:
    """计算与临时eventIndex无关的终态语义摘要。"""
    # by_alias输出Java字段名，mode=json把枚举等转换为稳定JSON值。
    payload = event.model_dump(by_alias=True, mode="json")
    # eventIndex每次重放可不同，不属于终态业务语义，计算前删除。
    payload.pop("eventIndex", None)
    # 固定键排序和紧凑分隔符后UTF-8编码，保证摘要可重复。
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    # 返回64位小写摘要供Redis终态记录。
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_id(config: RunnableConfig) -> str:
    """从StateSnapshot配置读取非空checkpoint ID，否则抛终态错误。"""
    # 连续get使用空字典默认值，避免configurable缺失时立即KeyError。
    checkpoint_id = config.get("configurable", {}).get("checkpoint_id")
    # 必须为非空字符串；其他动态类型不接受隐式str转换。
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise RunFinalStateError()
    return checkpoint_id


def _record_matches_request(record: RunRecord, request: AgentRunRequest) -> bool:
    """逐项比较Redis记录与Java请求的全部不可变运行身份。"""
    # ``and``短路求值，任一字段不同立即返回False。
    return (
        record.tenant_id == request.tenant_id
        and record.customer_id == request.customer_id
        and record.conversation_id == request.conversation_id
        and record.run_id == request.run_id
        and record.trigger_message_id == request.trigger_message_id
        and record.trigger_sequence == request.trigger_sequence
    )


async def _cancel_task(task: asyncio.Task[Any]) -> None:
    """幂等取消未完成任务，并吞掉预期的CancelledError。"""
    # 已完成任务无需cancel，也避免覆盖其结果状态。
    if task.done():
        return
    # cancel安排在下一个可取消点注入CancelledError。
    task.cancel()
    # 必须await任务以完成其finally清理，避免后台任务泄漏警告。
    try:
        await task
    except asyncio.CancelledError:
        # 这是主动取消的预期控制流，不需要转为生命周期失败。
        pass
