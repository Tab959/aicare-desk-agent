"""编译无独立Checkpointer的Knowledge RAG子图，并适配为根图生产分支。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, TypedDict, cast, runtime_checkable

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel, ConfigDict, Field

from aicare_agent_service.contracts import Citation
from aicare_agent_service.graph.branches import RootBranchDeployment
from aicare_agent_service.graph.context import AgentRuntimeContext
from aicare_agent_service.graph.state import CustomerServiceState
from aicare_agent_service.rag.answering import RagAnswerGenerator
from aicare_agent_service.rag.contracts import RagAnswer, RetrievalQuery, RetrievalResult
from aicare_agent_service.rag.faithfulness import (
    FaithfulnessDecision,
    RagFaithfulnessVerifier,
)
from aicare_agent_service.security.redaction import redact_sensitive_input

_INSUFFICIENT_ANSWER = "当前知识库中没有足够依据回答该问题，请补充具体场景或联系人工客服。"
_MAX_REPAIR_COUNT = 1
_RECURSION_LIMIT = 12


class KnowledgeRagState(TypedDict, total=False):
    """仅在局部子图存在的有限安全状态，不直接合并到父图checkpoint。"""

    sanitized_query: str
    retrieval_result: RetrievalResult
    answer: RagAnswer
    citations: tuple[Citation, ...]
    verification: FaithfulnessDecision
    repair_count: int


class KnowledgeRagRunResult(BaseModel):
    """受控RAG调用端口返回给根图或专业Agent的最小结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    answer: RagAnswer
    citations: tuple[Citation, ...] = Field(max_length=6)
    sufficient_evidence: bool
    repair_count: int = Field(strict=True, ge=0, le=1)


@runtime_checkable
class KnowledgeRagPort(Protocol):
    """Task 7专业Agent可复用的受控知识问答端口。"""

    async def ainvoke(
        self,
        *,
        query: str,
        context: AgentRuntimeContext,
        config: RunnableConfig | None = None,
    ) -> KnowledgeRagRunResult:
        """执行一次有界知识问答，不修改Java业务状态。"""
        ...


class _Route(StrEnum):
    """局部条件边使用的固定路由常量。"""

    GENERATE = "generate"
    INSUFFICIENT = "insufficient"
    FINISH = "finish"
    REPAIR = "repair"
    SAFE = "safe"


class KnowledgeRagSubgraph:
    """固定执行检索、生成、审核和最多一次修正的生产知识子图。"""

    def __init__(self) -> None:
        """编译不绑定Saver和thread_id的局部图。"""
        # 1、局部图只使用父运行时上下文，不创建独立持久化所有者。
        builder = StateGraph(KnowledgeRagState, context_schema=AgentRuntimeContext)
        # 2、节点按固定业务顺序注册并使用低基数追踪元数据。
        self._add_node(builder, "prepare_query", self._prepare_query)
        self._add_node(builder, "hybrid_retrieve", self._hybrid_retrieve)
        self._add_node(builder, "evidence_gate", self._evidence_gate)
        self._add_node(builder, "generate", self._generate)
        self._add_node(builder, "verify", self._verify)
        self._add_node(builder, "repair", self._repair)
        self._add_node(builder, "safe_insufficient", self._safe_insufficient)
        # 3、唯一循环边verify→repair→verify由repair_count限制为一次。
        builder.add_edge(START, "prepare_query")
        builder.add_edge("prepare_query", "hybrid_retrieve")
        builder.add_edge("hybrid_retrieve", "evidence_gate")
        builder.add_conditional_edges(
            "evidence_gate",
            self._route_evidence,
            {_Route.GENERATE: "generate", _Route.INSUFFICIENT: END},
        )
        builder.add_edge("generate", "verify")
        builder.add_conditional_edges(
            "verify",
            self._route_verification,
            {
                _Route.FINISH: END,
                _Route.REPAIR: "repair",
                _Route.SAFE: "safe_insufficient",
            },
        )
        builder.add_edge("repair", "verify")
        builder.add_edge("safe_insufficient", END)
        # 4、不传checkpointer，父图只接收最终白名单更新。
        self.graph: CompiledStateGraph[Any, AgentRuntimeContext, Any, Any] = cast(
            CompiledStateGraph[Any, AgentRuntimeContext, Any, Any],
            builder.compile(),
        )

    async def ainvoke(
        self,
        *,
        query: str,
        context: AgentRuntimeContext,
        config: RunnableConfig | None = None,
    ) -> KnowledgeRagRunResult:
        """在父请求deadline和固定递归上限内执行局部图。"""
        # 1、生产检索器缺失直接阻断，不注册内存或Mock回退。
        if context.knowledge_retriever is None:
            raise RuntimeError("RAG_RETRIEVER_NOT_CONFIGURED")
        # 2、整个局部图共享Java请求绝对deadline，覆盖所有模型和检索调用。
        remaining = (context.request_deadline - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise TimeoutError("RAG_DEADLINE_EXCEEDED")
        sanitized_query = redact_sensitive_input(query).sanitized_text.strip()
        if not sanitized_query:
            raise ValueError("RAG_QUERY_EMPTY")
        child_config = cast(RunnableConfig, dict(config or {}))
        child_config["recursion_limit"] = _RECURSION_LIMIT
        child_config["run_name"] = "knowledge_rag"
        child_config["tags"] = ["rag", "knowledge-subgraph"]
        child_config["metadata"] = {
            "subgraph": "knowledge_rag",
            "prompt_version": "rag-answer-v1+rag-faithfulness-v1",
            "max_model_calls": 4,
            "data_classification": "redacted",
        }
        async with asyncio.timeout(remaining):
            state = await self.graph.ainvoke(
                {"sanitized_query": sanitized_query[:4000], "repair_count": 0},
                child_config,
                context=context,
            )
        # 3、局部图必须形成一个完整RagAnswer，不能把中间候选交给父图。
        answer = state.get("answer")
        if not isinstance(answer, RagAnswer):
            raise TypeError("RAG_TERMINAL_STATE_INVALID")
        citations = tuple(state.get("citations", answer.citations))
        return KnowledgeRagRunResult(
            answer=answer,
            citations=citations,
            sufficient_evidence=answer.sufficient_evidence,
            repair_count=int(state.get("repair_count", 0)),
        )

    @staticmethod
    def _add_node(
        builder: StateGraph[KnowledgeRagState, AgentRuntimeContext, Any, Any],
        name: str,
        action: Callable[..., object],
    ) -> None:
        """使用稳定元数据注册局部节点。"""
        # 1、节点元数据不记录问题正文、身份、候选或模型输出。
        builder.add_node(
            name,
            action,
            metadata={"layer": "knowledge_rag", "node": name, "data_classification": "redacted"},
        )

    @staticmethod
    def _prepare_query(state: KnowledgeRagState) -> dict[str, object]:
        """再次脱敏并限制知识查询长度。"""
        # 1、子图入口不盲信父节点已完成脱敏，执行纵深防御。
        sanitized = redact_sensitive_input(state.get("sanitized_query", "")).sanitized_text.strip()
        if not sanitized:
            raise ValueError("RAG_QUERY_EMPTY")
        # 2、检索契约上限为4000字符，超长输入确定性截断。
        return {"sanitized_query": sanitized[:4000], "repair_count": 0}

    @staticmethod
    async def _hybrid_retrieve(
        state: KnowledgeRagState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, object]:
        """使用Java身份租户调用同一个生产Hybrid Retriever。"""
        # 1、tenant_id只从不可持久化Java身份注入，不接受模型或用户参数。
        retriever = runtime.context.knowledge_retriever
        if retriever is None:
            raise RuntimeError("RAG_RETRIEVER_NOT_CONFIGURED")
        query = RetrievalQuery(
            tenant_id=runtime.context.expected_identity.tenant_id,
            text=state["sanitized_query"],
            candidate_limit=30,
            result_limit=6,
        )
        # 2、返回的RetrievalResult已经限制候选数量并验证单租户。
        result = await retriever.retrieve(query)
        return {"retrieval_result": result}

    @staticmethod
    def _evidence_gate(state: KnowledgeRagState) -> dict[str, object]:
        """证据不足时构造固定答复，充分时不改变状态。"""
        # 1、门禁只读取检索器的确定性结论，不调用模型复判分数。
        retrieval = state["retrieval_result"]
        if retrieval.sufficient_evidence:
            return {}
        # 2、证据不足不引用弱候选，也不生成可能幻觉的自然语言答案。
        answer = RagAnswer(
            content=_INSUFFICIENT_ANSWER,
            sufficient_evidence=False,
            citations=(),
        )
        return {"answer": answer, "citations": ()}

    @staticmethod
    def _route_evidence(state: KnowledgeRagState) -> str:
        """根据确定性证据门禁选择生成或直接结束。"""
        # 1、只有显式充分且存在候选才允许模型生成。
        retrieval = state["retrieval_result"]
        return (
            _Route.GENERATE
            if retrieval.sufficient_evidence and retrieval.chunks
            else _Route.INSUFFICIENT
        )

    @staticmethod
    async def _generate(
        state: KnowledgeRagState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, object]:
        """执行首次知识回答生成。"""
        # 1、生成器只能访问脱敏问题和有限候选。
        answer = await RagAnswerGenerator(runtime.context.model_provider).generate(
            query=state["sanitized_query"],
            candidates=state["retrieval_result"].chunks,
            deadline=_loop_deadline(runtime.context),
        )
        # 2、引用同时保存于局部状态，最终只映射安全Citation到父图。
        return {"answer": answer, "citations": answer.citations}

    @staticmethod
    async def _verify(
        state: KnowledgeRagState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, object]:
        """结构化审核当前回答。"""
        # 1、每次生成后恰好执行一次REVIEW用途模型调用。
        decision = await RagFaithfulnessVerifier(runtime.context.model_provider).verify(
            query=state["sanitized_query"],
            candidates=state["retrieval_result"].chunks,
            answer=state["answer"],
            deadline=_loop_deadline(runtime.context),
            repair_count=state.get("repair_count", 0),
        )
        # 2、只保存有限布尔和原因码，不保存审核自由文本。
        return {"verification": decision}

    @staticmethod
    def _route_verification(state: KnowledgeRagState) -> str:
        """审核成功结束，首次失败修正，第二次失败安全降级。"""
        # 1、四项门禁全部通过才接受当前回答。
        if state["verification"].passed:
            return _Route.FINISH
        # 2、修正计数小于固定上限时只允许再生成一次。
        if state.get("repair_count", 0) < _MAX_REPAIR_COUNT:
            return _Route.REPAIR
        return _Route.SAFE

    @staticmethod
    async def _repair(
        state: KnowledgeRagState,
        runtime: Runtime[AgentRuntimeContext],
    ) -> dict[str, object]:
        """使用稳定审核原因码执行唯一一次受限修正。"""
        # 1、修正仍只能访问同一批候选，不重新检索或扩大工具预算。
        answer = await RagAnswerGenerator(runtime.context.model_provider).generate(
            query=state["sanitized_query"],
            candidates=state["retrieval_result"].chunks,
            deadline=_loop_deadline(runtime.context),
            previous_answer=state["answer"].content,
            repair_reason=state["verification"].reason_code.value,
        )
        # 2、计数固定增加一次，条件边不会允许第二次修正。
        return {
            "answer": answer,
            "citations": answer.citations,
            "repair_count": state.get("repair_count", 0) + 1,
        }

    @staticmethod
    def _safe_insufficient(_: KnowledgeRagState) -> dict[str, object]:
        """二次审核仍失败时清空引用并返回确定性安全答复。"""
        # 1、不保留失败回答及其引用，避免父图误把未通过内容交给Java。
        answer = RagAnswer(
            content=_INSUFFICIENT_ANSWER,
            sufficient_evidence=False,
            citations=(),
        )
        return {"answer": answer, "citations": ()}


class KnowledgeRagBranch:
    """把可复用RAG端口适配为根图的生产知识分支。"""

    deployment_kind = RootBranchDeployment.PRODUCTION

    def __init__(self, service: KnowledgeRagPort | None = None) -> None:
        """绑定唯一受控知识问答端口。"""
        # 1、默认构造正式子图；不提供演示或内存替代实现。
        self._service = service or KnowledgeRagSubgraph()

    async def ainvoke(
        self,
        input: CustomerServiceState,
        config: RunnableConfig,
        *,
        context: AgentRuntimeContext,
    ) -> Mapping[str, object]:
        """执行知识子图并映射为根图允许的三个局部字段。"""
        # 1、只读取根图安全预处理后的当前问题。
        result = await self._service.ainvoke(
            query=input["sanitized_user_message"],
            context=context,
            config=config,
        )
        # 2、不返回identity、route、候选、验证详情或Java状态。
        return {
            "final_answer": result.answer.content,
            "citations": list(result.citations),
            "handoff_suggestion": None,
        }


def _loop_deadline(context: AgentRuntimeContext) -> float:
    """把UTC绝对截止时间转换为当前事件循环绝对时间。"""
    # 1、以当前UTC计算剩余秒数，跨平台不依赖系统单调时钟起点。
    remaining = (context.request_deadline - datetime.now(UTC)).total_seconds()
    if remaining <= 0:
        raise TimeoutError("RAG_DEADLINE_EXCEEDED")
    # 2、asyncio.timeout_at使用当前loop的单调时间坐标。
    return asyncio.get_running_loop().time() + remaining
