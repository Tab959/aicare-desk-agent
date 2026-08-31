"""验证查询改写只接收脱敏问题，并在低置信度或故障时安全回退。"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage

from aicare_agent_service.models.contracts import ModelPurpose
from aicare_agent_service.models.fake import FakeModelProvider
from aicare_agent_service.rag.query_rewrite import QueryRewriter, build_rewrite_messages


def _rewrite_message(*, query: str, confidence: float = 0.9) -> AIMessage:
    """构造LangChain function-calling兼容的结构化改写响应。"""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "QueryRewriteDecision",
                "args": {
                    "normalized_query": query,
                    "language_hint": "zh-CN",
                    "intent_hint": "POLICY",
                    "confidence": confidence,
                },
                "id": "rewrite-1",
                "type": "tool_call",
            }
        ],
    )


@pytest.mark.asyncio
async def test_rewrite_uses_structured_output_after_redaction() -> None:
    secret = "rewrite-password-canary"
    provider = FakeModelProvider(
        scripts={ModelPurpose.ROUTING: [_rewrite_message(query="Steam礼物 折扣 政策")]}
    )
    rewriter = QueryRewriter(model_provider=provider)

    result = await rewriter.rewrite(
        f"密码={secret}，Steam礼物有没有折扣？",
        deadline=asyncio.get_running_loop().time() + 2,
    )
    messages = build_rewrite_messages(result.redacted_original)
    rendered = repr(messages)

    assert result.query == "Steam礼物 折扣 政策"
    assert result.used_fallback is False
    assert secret not in result.redacted_original
    assert secret not in rendered
    assert "[REDACTED_PASSWORD]" in rendered


@pytest.mark.asyncio
async def test_low_confidence_and_timeout_fall_back_to_redacted_original() -> None:
    low_provider = FakeModelProvider(
        scripts={ModelPurpose.ROUTING: [_rewrite_message(query="错误改写", confidence=0.2)]}
    )
    timeout_provider = FakeModelProvider(
        scripts={ModelPurpose.ROUTING: [TimeoutError("provider-secret-canary")]}
    )

    low = await QueryRewriter(model_provider=low_provider).rewrite(
        "Steam CDK怎么激活？", deadline=asyncio.get_running_loop().time() + 2
    )
    timed_out = await QueryRewriter(model_provider=timeout_provider).rewrite(
        "Steam礼物怎么交付？", deadline=asyncio.get_running_loop().time() + 2
    )

    assert low.query == "Steam CDK怎么激活？"
    assert low.used_fallback is True
    assert timed_out.query == "Steam礼物怎么交付？"
    assert timed_out.used_fallback is True
    assert "provider-secret-canary" not in repr(timed_out)


def test_model_cannot_supply_tenant_or_business_filters() -> None:
    schema = QueryRewriter.output_schema().model_json_schema()
    properties = schema["properties"]

    assert "tenant_id" not in properties
    assert "filters" not in properties
    assert set(properties) == {
        "normalized_query",
        "language_hint",
        "intent_hint",
        "confidence",
    }
