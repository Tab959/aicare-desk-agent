from concurrent.futures import ThreadPoolExecutor
from typing import Literal, cast

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field, ValidationError

from aicare_agent_service.models.contracts import ModelConfigurationError, ModelPurpose
from aicare_agent_service.models.fake import (
    FakeModelProvider,
    FakeModelScriptExhaustedError,
    ScriptedFakeChatModel,
)


class RouteProbe(BaseModel):
    """Task 2专用结构化输出探针，不作为正式路由契约。"""

    intent: Literal["pre_sales", "after_sales"]
    confidence: float = Field(ge=0, le=1)


def test_fake_model_returns_scripted_reply() -> None:
    model = ScriptedFakeChatModel([AIMessage(content="已找到游戏")])

    result = model.invoke("查询游戏")

    assert result.content == "已找到游戏"


def test_fake_model_consumes_multi_turn_script_in_order() -> None:
    model = ScriptedFakeChatModel([AIMessage(content="第一轮"), AIMessage(content="第二轮")])

    assert model.invoke("一").content == "第一轮"
    assert model.invoke("二").content == "第二轮"


def test_fake_model_can_return_standard_tool_call() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_games",
                "args": {"query": "RPG"},
                "id": "call-1",
                "type": "tool_call",
            }
        ],
    )

    result = ScriptedFakeChatModel([message]).invoke("找RPG")

    assert result.tool_calls[0]["name"] == "search_games"
    assert result.tool_calls[0]["args"] == {"query": "RPG"}


@pytest.mark.asyncio
async def test_sync_and_async_calls_share_one_consumption_rule() -> None:
    model = ScriptedFakeChatModel([AIMessage(content="同步"), AIMessage(content="异步")])

    assert model.invoke("同步调用").content == "同步"
    assert (await model.ainvoke("异步调用")).content == "异步"


def test_fake_model_streams_scripted_content() -> None:
    model = ScriptedFakeChatModel([AIMessage(content="已找到游戏")])

    content = "".join(cast(str, chunk.content) for chunk in model.stream("查询游戏"))

    assert content == "已找到游戏"


@pytest.mark.asyncio
async def test_fake_model_streams_scripted_content_asynchronously() -> None:
    model = ScriptedFakeChatModel([AIMessage(content="异步流式")])

    chunks = [cast(str, chunk.content) async for chunk in model.astream("查询游戏")]

    assert "".join(chunks) == "异步流式"


def test_fake_model_parses_pydantic_structured_output() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "RouteProbe",
                "args": {"intent": "pre_sales", "confidence": 0.9},
                "id": "route-1",
                "type": "tool_call",
            }
        ],
    )
    model = ScriptedFakeChatModel([message]).with_structured_output(RouteProbe)

    result = model.invoke("判断意图")

    assert result == RouteProbe(intent="pre_sales", confidence=0.9)


def test_fake_model_rejects_invalid_pydantic_structured_output() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "RouteProbe",
                "args": {"intent": "unsupported", "confidence": 2},
                "id": "route-invalid",
                "type": "tool_call",
            }
        ],
    )
    model = ScriptedFakeChatModel([message]).with_structured_output(RouteProbe)

    with pytest.raises(ValidationError):
        model.invoke("判断意图")


@pytest.mark.parametrize(
    "scripted_error",
    [RuntimeError("模拟限流"), TimeoutError("模拟超时")],
)
def test_fake_model_raises_scripted_failures(scripted_error: Exception) -> None:
    model = ScriptedFakeChatModel([scripted_error])

    with pytest.raises(type(scripted_error)) as exc_info:
        model.invoke("触发故障")

    assert exc_info.value is scripted_error


def test_fake_model_reports_script_exhaustion_in_chinese() -> None:
    model = ScriptedFakeChatModel([AIMessage(content="唯一回复")])
    model.invoke("第一次")

    with pytest.raises(FakeModelScriptExhaustedError, match="Fake模型脚本已耗尽"):
        model.invoke("第二次")


def test_fake_model_rejects_unsupported_script_item() -> None:
    with pytest.raises(ModelConfigurationError, match="Fake模型脚本项仅支持"):
        ScriptedFakeChatModel([cast(AIMessage, "错误脚本")])


def test_fake_model_consumption_is_thread_safe() -> None:
    response_count = 12
    model = ScriptedFakeChatModel(
        [AIMessage(content=f"回复-{index}") for index in range(response_count)]
    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        replies = list(executor.map(lambda index: model.invoke(str(index)).content, range(12)))

    assert set(replies) == {f"回复-{index}" for index in range(response_count)}


def test_fake_provider_isolates_scripts_by_model_purpose() -> None:
    provider = FakeModelProvider(
        {
            ModelPurpose.ROUTING: [AIMessage(content="路由回复")],
            ModelPurpose.ANSWER: [AIMessage(content="答案回复")],
        }
    )

    routing_model = provider.create(ModelPurpose.ROUTING)
    answer_model = provider.create(ModelPurpose.ANSWER)

    assert answer_model.invoke("回答").content == "答案回复"
    assert routing_model.invoke("路由").content == "路由回复"


def test_fake_provider_creates_fresh_script_for_each_model() -> None:
    provider = FakeModelProvider({ModelPurpose.ROUTING: [AIMessage(content="独立回复")]})

    first_model = provider.create(ModelPurpose.ROUTING)
    second_model = provider.create(ModelPurpose.ROUTING)

    assert first_model.invoke("第一次").content == "独立回复"
    assert second_model.invoke("第二次").content == "独立回复"


def test_fake_provider_rejects_missing_or_unknown_purpose() -> None:
    provider = FakeModelProvider()

    with pytest.raises(ModelConfigurationError, match="未配置Fake模型脚本"):
        provider.create(ModelPurpose.ROUTING)

    with pytest.raises(ModelConfigurationError, match="不支持的模型用途"):
        provider.create(cast(ModelPurpose, "unknown"))
