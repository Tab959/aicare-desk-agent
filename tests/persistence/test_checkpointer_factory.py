import logging
from contextlib import asynccontextmanager

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from aicare_agent_service.config import CheckpointBackend, Environment, Settings
from aicare_agent_service.contracts import AgentBusinessContext
from aicare_agent_service.contracts.decisions import (
    AgentCode,
    Citation,
    EscalationSuggestion,
    HandoffSuggestion,
    Intent,
    MessageRole,
    RouteCode,
    RouteDecision,
    SafeConversationMessage,
    SafeToolResult,
    ToolResultStatus,
)
from aicare_agent_service.contracts.events import HandoffPriority
from aicare_agent_service.graph.state import AgentIdentity
from aicare_agent_service.persistence import checkpointer as checkpointer_module
from aicare_agent_service.persistence.checkpointer import (
    build_checkpoint_serializer,
    checkpointer_resource,
)


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": Environment.TEST,
        "checkpoint_backend": CheckpointBackend.MEMORY,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.mark.asyncio
async def test_development_memory_checkpointer_is_explicitly_non_persistent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING)

    async with checkpointer_resource(settings(environment=Environment.DEVELOPMENT)) as checkpointer:
        assert isinstance(checkpointer, InMemorySaver)
        assert isinstance(checkpointer.serde, JsonPlusSerializer)
        assert checkpointer.serde.pickle_fallback is False
        assert checkpointer.serde._allowed_msgpack_modules is not True

    assert "非持久化，仅开发" in caplog.text


@pytest.mark.asyncio
async def test_production_rejects_memory_checkpointer() -> None:
    with pytest.raises(ValueError, match="生产环境必须使用PostgreSQL Checkpointer"):
        async with checkpointer_resource(
            settings(
                environment=Environment.PRODUCTION,
                checkpoint_backend=CheckpointBackend.MEMORY,
            )
        ):
            pytest.fail("生产环境不应创建内存Checkpointer")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"checkpoint_backend": CheckpointBackend.POSTGRES},
            "缺少AICARE_AGENT_POSTGRES_DSN",
        ),
        (
            {
                "checkpoint_backend": CheckpointBackend.POSTGRES,
                "agent_postgres_dsn": "postgresql://agent:password@db/agent",
            },
            "缺少LANGGRAPH_AES_KEY",
        ),
    ],
)
async def test_postgres_checkpointer_requires_dsn_and_encryption_key(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        async with checkpointer_resource(settings(**overrides)):
            pytest.fail("配置不完整时不应创建PostgreSQL Checkpointer")


@pytest.mark.asyncio
async def test_postgres_factory_owns_context_without_running_schema_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class ProbeSaver:
        async def setup(self) -> None:
            pytest.fail("运行时工厂不得调用setup()")

    probe = ProbeSaver()

    @asynccontextmanager
    async def fake_from_conn_string(
        dsn: str,
        *,
        serde: object,
        pipeline: bool = False,
    ):
        del pipeline
        assert dsn == "postgresql://agent:password@db/agent"
        assert isinstance(serde, EncryptedSerializer)
        events.append("enter")
        yield probe
        events.append("exit")

    monkeypatch.setattr(
        checkpointer_module.AsyncPostgresSaver,
        "from_conn_string",
        staticmethod(fake_from_conn_string),
    )

    async with checkpointer_resource(
        settings(
            checkpoint_backend=CheckpointBackend.POSTGRES,
            agent_postgres_dsn="postgresql://agent:password@db/agent",
            checkpoint_encryption_key="0123456789abcdef0123456789abcdef",
        )
    ) as checkpointer:
        assert checkpointer is probe
        assert events == ["enter"]

    assert events == ["enter", "exit"]


def test_encrypted_serializer_round_trips_task3_state_without_pickle() -> None:
    serializer = build_checkpoint_serializer(
        settings(checkpoint_encryption_key="0123456789abcdef0123456789abcdef")
    )
    state = {
        "identity": AgentIdentity(
            tenant_id="tenant-001",
            customer_id="customer-001",
            conversation_id="conversation-001",
            run_id="run-001",
            trigger_message_id="message-001",
            trigger_sequence=1,
        ),
        "messages": [HumanMessage(content="查询订单", id="message-001")],
        "business_context": AgentBusinessContext(
            subject="订单查询",
            orderId="order-001",
            orderNo="AD20260001",
            orderStatus="PAID",
            entitlementId=None,
            entitlementType=None,
            entitlementStatus=None,
        ),
        "route_decision": RouteDecision(
            intent=Intent.ORDER_SUPPORT,
            route_code=RouteCode.ORDER_SUPPORT,
            agent_code=AgentCode.ORDER_SUPPORT_AGENT,
            confidence=0.98,
            reason="用户明确查询订单",
        ),
        "safe_history": [
            SafeConversationMessage(
                message_id="message-001",
                sequence=1,
                role=MessageRole.CUSTOMER,
                content="查询订单",
            ),
        ],
        "citations": [
            Citation(
                document_id="document-001",
                version=1,
                title_path=("订单帮助",),
                source_uri="kb://document-001/1",
            ),
        ],
        "tool_results": [
            SafeToolResult(
                tool_name="get_order_detail",
                status=ToolResultStatus.SUCCESS,
                summary="订单已支付",
                facts={"orderStatus": "PAID"},
            ),
        ],
        "handoff_suggestion": HandoffSuggestion(
            reason="用户要求人工客服",
            priority=HandoffPriority.MEDIUM,
            summary="订单查询需人工处理",
        ),
        "escalation_suggestion": EscalationSuggestion(
            issue_type="ORDER_SUPPORT_REQUIRED",
            reason="自动查询未解决问题",
            summary="建议升级订单售后工单",
        ),
    }

    data_type, ciphertext = serializer.dumps_typed(state)
    restored = serializer.loads_typed((data_type, ciphertext))

    assert isinstance(serializer, EncryptedSerializer)
    assert isinstance(serializer.serde, JsonPlusSerializer)
    assert serializer.serde.pickle_fallback is False
    assert serializer.serde._allowed_msgpack_modules is not True
    assert data_type == "msgpack+aes"
    assert "查询订单".encode() not in ciphertext
    assert restored == state
    assert isinstance(restored["identity"], AgentIdentity)
    assert isinstance(restored["business_context"], AgentBusinessContext)
    assert isinstance(restored["route_decision"], RouteDecision)


@pytest.mark.parametrize("key", ["short", "长" * 16])
def test_encryption_key_must_have_an_aes_byte_length(key: str) -> None:
    with pytest.raises(ValueError, match="16、24或32字节"):
        build_checkpoint_serializer(settings(checkpoint_encryption_key=key))
