"""验证Knowledge RAG子图的证据门禁、引用绑定、单次修正和根图端口边界。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from langchain_core.messages import AIMessage

from aicare_agent_service.contracts import Citation
from aicare_agent_service.graph.branches import RootBranchDeployment
from aicare_agent_service.graph.context import AgentRuntimeContext
from aicare_agent_service.models.contracts import ModelPurpose
from aicare_agent_service.models.fake import ScriptedFakeChatModel
from aicare_agent_service.rag.contracts import (
    KnowledgeMetadata,
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
)
from aicare_agent_service.subgraphs.knowledge_rag import (
    KnowledgeRagBranch,
    KnowledgeRagSubgraph,
)


def _candidate() -> RetrievedChunk:
    """构造一条可稳定引用的同租户候选证据。"""
    metadata = KnowledgeMetadata(
        tenant_id="tenant-rag",
        knowledge_base_id="kb-policy",
        document_id="doc-gift",
        version=3,
        language="zh-CN",
        category="DELIVERY_POLICY",
    )
    return RetrievedChunk(
        chunk_id="chunk-gift-1",
        metadata=metadata,
        content="Steam礼物需要由人工客服核对收货地区后发送到用户自己的Steam账户。",
        citation=Citation(
            document_id="doc-gift",
            version=3,
            title_path=("交付政策", "Steam礼物"),
            source_uri="aicare://knowledge/doc-gift",
        ),
        fused_score=0.03,
        rerank_score=0.91,
    )


class StaticRetriever:
    """返回固定有限候选并记录身份与查询，作为检索端口替身。"""

    def __init__(self, result: RetrievalResult) -> None:
        self.result = result
        self.queries: list[RetrievalQuery] = []

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """记录调用后返回预制结果。"""
        # 1、保存强类型查询，测试据此验证租户由运行上下文注入。
        self.queries.append(query)
        # 2、返回不含Embedding和ES原始响应的安全结果。
        return self.result


class CountingProvider:
    """为ANSWER和REVIEW用途复用各自脚本游标并统计模型调用次数。"""

    def __init__(self, *, answers: list[AIMessage], reviews: list[AIMessage]) -> None:
        self.models = {
            ModelPurpose.ANSWER: ScriptedFakeChatModel(answers),
            ModelPurpose.REVIEW: ScriptedFakeChatModel(reviews),
        }
        self.creations: list[ModelPurpose] = []

    def create(self, purpose: ModelPurpose) -> ScriptedFakeChatModel:
        """返回对应用途的共享测试模型。"""
        # 1、记录每次固定用途创建请求。
        self.creations.append(purpose)
        # 2、复用模型实例使修正调用消费下一条脚本响应。
        return self.models[purpose]


def _answer(content: str, marker: str = "K1") -> AIMessage:
    """构造严格回答Schema对应的function call。"""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "GeneratedRagAnswer",
                "args": {"content": content, "citation_markers": [marker]},
                "id": "answer-call",
                "type": "tool_call",
            }
        ],
    )


def _review(*, passed: bool) -> AIMessage:
    """构造严格忠实度Schema对应的function call。"""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "FaithfulnessDecision",
                "args": {
                    "citation_coverage": passed,
                    "factually_supported": passed,
                    "realtime_boundary_respected": passed,
                    "document_instructions_ignored": passed,
                    "reason_code": "PASS" if passed else "UNSUPPORTED_CLAIM",
                },
                "id": "review-call",
                "type": "tool_call",
            }
        ],
    )


def _context(retriever: StaticRetriever, provider: CountingProvider) -> AgentRuntimeContext:
    """构造包含检索器但不把依赖写入图状态的运行上下文。"""
    from types import SimpleNamespace

    return AgentRuntimeContext(
        expected_identity=SimpleNamespace(tenant_id="tenant-rag"),
        java_client=SimpleNamespace(),
        model_provider=provider,
        request_deadline=datetime.now(UTC) + timedelta(seconds=30),
        knowledge_retriever=retriever,
    )


@pytest.mark.asyncio
async def test_insufficient_evidence_returns_deterministic_answer_without_model() -> None:
    """证据不足不得调用回答或审核模型。"""
    retriever = StaticRetriever(
        RetrievalResult(
            tenant_id="tenant-rag",
            chunks=(),
            sufficient_evidence=False,
            elapsed_ms=2,
        )
    )
    provider = CountingProvider(answers=[], reviews=[])

    result = await KnowledgeRagSubgraph().ainvoke(
        query="Steam礼物怎么交付？",
        context=_context(retriever, provider),
    )

    assert result.sufficient_evidence is False
    assert result.citations == ()
    assert result.repair_count == 0
    assert provider.creations == []
    assert retriever.queries[0].tenant_id == "tenant-rag"


@pytest.mark.asyncio
async def test_supported_answer_binds_stable_marker_and_root_branch_update() -> None:
    """模型只能引用候选标记，根图分支只返回允许字段。"""
    candidate = _candidate()
    retriever = StaticRetriever(
        RetrievalResult(
            tenant_id="tenant-rag",
            chunks=(candidate,),
            sufficient_evidence=True,
            elapsed_ms=4,
        )
    )
    provider = CountingProvider(
        answers=[_answer("Steam礼物由人工客服核对地区后发送。[K1]")],
        reviews=[_review(passed=True)],
    )
    context = _context(retriever, provider)
    service = KnowledgeRagSubgraph()

    result = await service.ainvoke(query="Steam礼物怎么交付？", context=context)
    branch = KnowledgeRagBranch(service)
    branch_context = _context(
        retriever,
        CountingProvider(
            answers=[_answer("Steam礼物由人工客服核对地区后发送。[K1]")],
            reviews=[_review(passed=True)],
        ),
    )
    update = await branch.ainvoke(
        {"sanitized_user_message": "Steam礼物怎么交付？"},
        {},
        context=branch_context,
    )

    assert service.graph.checkpointer is None
    assert result.answer.content.endswith("[K1]")
    assert result.citations == (candidate.citation,)
    assert result.repair_count == 0
    assert update == {
        "final_answer": "Steam礼物由人工客服核对地区后发送。[K1]",
        "citations": [candidate.citation],
        "handoff_suggestion": None,
    }
    assert branch.deployment_kind is RootBranchDeployment.PRODUCTION


@pytest.mark.asyncio
async def test_failed_review_repairs_once_then_returns_safe_insufficient_answer() -> None:
    """连续审核失败只能修正一次，之后确定性降级且图不会无限循环。"""
    candidate = _candidate()
    retriever = StaticRetriever(
        RetrievalResult(
            tenant_id="tenant-rag",
            chunks=(candidate,),
            sufficient_evidence=True,
            elapsed_ms=3,
        )
    )
    provider = CountingProvider(
        answers=[
            _answer("未经证据支持的第一次回答。[K1]"),
            _answer("仍然未经证据支持的修正回答。[K1]"),
        ],
        reviews=[_review(passed=False), _review(passed=False)],
    )

    result = await KnowledgeRagSubgraph().ainvoke(
        query="Steam礼物怎么交付？",
        context=_context(retriever, provider),
    )

    assert result.sufficient_evidence is False
    assert result.repair_count == 1
    assert result.citations == ()
    assert provider.creations == [
        ModelPurpose.ANSWER,
        ModelPurpose.REVIEW,
        ModelPurpose.ANSWER,
        ModelPurpose.REVIEW,
    ]


@pytest.mark.asyncio
async def test_missing_production_retriever_fails_closed() -> None:
    """知识路由没有真实检索器时必须阻断，不能返回演示答案。"""
    from types import SimpleNamespace

    provider = CountingProvider(answers=[], reviews=[])
    context = AgentRuntimeContext(
        expected_identity=SimpleNamespace(tenant_id="tenant-rag"),
        java_client=SimpleNamespace(),
        model_provider=provider,
        request_deadline=datetime.now(UTC) + timedelta(seconds=30),
    )

    with pytest.raises(RuntimeError, match="RAG_RETRIEVER_NOT_CONFIGURED"):
        await KnowledgeRagSubgraph().ainvoke(query="交付规则", context=context)
