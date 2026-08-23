import pytest
from pydantic import ValidationError

from aicare_agent_service.contracts.decisions import (
    AgentCode,
    Citation,
    EscalationSuggestion,
    HandoffPriority,
    HandoffSuggestion,
    Intent,
    MessageRole,
    RouteCode,
    RouteDecision,
    SafeConversationMessage,
    SafeToolResult,
    ToolResultStatus,
)


def test_safe_conversation_message_keeps_only_checkpoint_safe_fields() -> None:
    message = SafeConversationMessage(
        message_id=" message-001 ",
        sequence=12,
        role=MessageRole.CUSTOMER,
        content=" CDK 无法激活 ",
    )

    assert message.model_dump(mode="json") == {
        "message_id": "message-001",
        "sequence": 12,
        "role": "CUSTOMER",
        "content": "CDK 无法激活",
    }


@pytest.mark.parametrize("sequence", [0, -1, True, "12"])
def test_safe_conversation_message_rejects_invalid_java_sequence(sequence: object) -> None:
    with pytest.raises(ValidationError):
        SafeConversationMessage(
            message_id="message-001",
            sequence=sequence,
            role="CUSTOMER",
            content="订单问题",
        )


@pytest.mark.parametrize("role", ["TOOL", "DEVELOPER", "customer"])
def test_safe_conversation_message_rejects_non_conversation_roles(role: str) -> None:
    with pytest.raises(ValidationError):
        SafeConversationMessage(
            message_id="message-001",
            sequence=1,
            role=role,
            content="订单问题",
        )


@pytest.mark.parametrize(
    ("intent", "route_code", "agent_code"),
    [
        (Intent.HUMAN_REQUEST, RouteCode.HUMAN_HANDOFF, AgentCode.HUMAN_HANDOFF),
        (Intent.AFTER_SALES, RouteCode.AFTER_SALES, AgentCode.AFTER_SALES_AGENT),
        (Intent.ORDER_SUPPORT, RouteCode.ORDER_SUPPORT, AgentCode.ORDER_SUPPORT_AGENT),
        (Intent.PRE_SALES, RouteCode.PRE_SALES, AgentCode.PRE_SALES_AGENT),
        (Intent.KNOWLEDGE, RouteCode.KNOWLEDGE, AgentCode.KNOWLEDGE_RAG),
        (Intent.UNSUPPORTED, RouteCode.UNSUPPORTED, AgentCode.SAFE_FALLBACK),
    ],
)
def test_route_decision_accepts_only_the_fixed_route_for_each_intent(
    intent: Intent, route_code: RouteCode, agent_code: AgentCode
) -> None:
    decision = RouteDecision(
        intent=intent,
        route_code=route_code,
        agent_code=agent_code,
        confidence=0.95,
        reason="用户明确表达对应意图",
    )

    assert decision.intent is intent
    assert decision.route_code is route_code
    assert decision.agent_code is agent_code


def test_route_decision_rejects_a_contradictory_agent_target() -> None:
    with pytest.raises(ValidationError):
        RouteDecision(
            intent="AFTER_SALES",
            route_code="PRE_SALES",
            agent_code="PRE_SALES_AGENT",
            confidence=0.95,
            reason="错误组合",
        )


def test_knowledge_intent_replaces_the_narrow_faq_route() -> None:
    decision = RouteDecision(
        intent="KNOWLEDGE",
        route_code="KNOWLEDGE",
        agent_code="KNOWLEDGE_RAG",
        confidence=0.95,
        reason="需要查询平台知识库",
    )

    assert decision.intent is Intent.KNOWLEDGE
    assert decision.route_code is RouteCode.KNOWLEDGE
    assert decision.agent_code is AgentCode.KNOWLEDGE_RAG

    with pytest.raises(ValidationError):
        RouteDecision(
            intent="FAQ",
            route_code="FAQ",
            agent_code="FAQ_RAG",
            confidence=0.95,
            reason="旧路由不再接受",
        )


@pytest.mark.parametrize("confidence", [-0.01, 1.01, True, "0.95"])
def test_route_decision_rejects_invalid_confidence(confidence: object) -> None:
    with pytest.raises(ValidationError):
        RouteDecision(
            intent="KNOWLEDGE",
            route_code="KNOWLEDGE",
            agent_code="KNOWLEDGE_RAG",
            confidence=confidence,
            reason="常见问题",
        )


def test_citation_keeps_source_identity_without_document_body() -> None:
    citation = Citation(
        document_id="doc-001",
        version=3,
        title_path=("售后政策", "CDK激活"),
        source_uri="kb://doc-001/v3",
    )

    assert citation.model_dump(mode="json") == {
        "document_id": "doc-001",
        "version": 3,
        "title_path": ["售后政策", "CDK激活"],
        "source_uri": "kb://doc-001/v3",
    }

    with pytest.raises(ValidationError):
        Citation(
            document_id="doc-001",
            version=3,
            title_path=("售后政策",),
            source_uri="kb://doc-001/v3",
            content="不应把文档正文复制到引用模型",
        )


def test_safe_tool_result_accepts_only_flat_redacted_facts() -> None:
    result = SafeToolResult(
        tool_name="get_order_detail",
        status=ToolResultStatus.SUCCESS,
        summary="订单已支付，权益已交付。",
        facts={"orderStatus": "PAID", "itemCount": 1, "manualDelivery": False},
    )

    assert result.facts == {
        "orderStatus": "PAID",
        "itemCount": 1,
        "manualDelivery": False,
    }


@pytest.mark.parametrize(
    "facts",
    [
        {"rawResponse": "upstream body"},
        {"privateDownloadUrl": "https://private.example/signed"},
        {"accessToken": "secret-token"},
        {"accountPassword": "plain-password"},
        {"password": "plain-password"},
        {"cdk": "AAAA-BBBB-CCCC"},
        {"entitlementCdk": "AAAA-BBBB-CCCC"},
        {"downloadUrl": "https://private.example/signed"},
        {"stackTrace": "Traceback ..."},
        {"order": {"status": "PAID"}},
        {"items": ["one", "two"]},
    ],
)
def test_safe_tool_result_rejects_raw_sensitive_or_nested_evidence(
    facts: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        SafeToolResult(
            tool_name="get_order_detail",
            status="SUCCESS",
            summary="已获取订单状态。",
            facts=facts,
        )


def test_handoff_is_a_suggestion_without_execution_state() -> None:
    suggestion = HandoffSuggestion(
        reason="用户明确要求人工客服",
        priority=HandoffPriority.HIGH,
        summary="已核验订单，尚未执行任何业务变更。",
    )

    assert suggestion.priority is HandoffPriority.HIGH
    with pytest.raises(ValidationError):
        HandoffSuggestion(
            reason="用户明确要求人工客服",
            priority="HIGH",
            summary="等待Java校验",
            executed=True,
        )


def test_escalation_is_a_suggestion_without_work_order_state() -> None:
    suggestion = EscalationSuggestion(
        issue_type="CDK_INVALID",
        reason="CDK无法激活且基础排查无效",
        summary="建议Java校验后创建售后工单。",
    )

    assert suggestion.issue_type == "CDK_INVALID"
    with pytest.raises(ValidationError):
        EscalationSuggestion(
            issue_type="CDK_INVALID",
            reason="需要创建工单",
            summary="等待Java校验",
            work_order_id="work-order-001",
        )


def test_internal_contract_models_are_frozen_and_reject_unknown_fields() -> None:
    decision = RouteDecision(
        intent="KNOWLEDGE",
        route_code="KNOWLEDGE",
        agent_code="KNOWLEDGE_RAG",
        confidence=0.9,
        reason="平台规则问题",
    )

    with pytest.raises(ValidationError):
        decision.reason = "修改后的理由"

    with pytest.raises(ValidationError):
        RouteDecision(
            intent="KNOWLEDGE",
            route_code="KNOWLEDGE",
            agent_code="KNOWLEDGE_RAG",
            confidence=0.9,
            reason="平台规则问题",
            raw_response={"provider": "deepseek"},
        )
