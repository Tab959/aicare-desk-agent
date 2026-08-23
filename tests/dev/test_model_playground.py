import pytest
from langchain.messages import AIMessage, AIMessageChunk, HumanMessage

from aicare_agent_service.dev.model_playground import build_model_playground_graph
from aicare_agent_service.models.fake import ScriptedFakeChatModel


@pytest.mark.asyncio
async def test_model_playground_preserves_input_and_appends_model_reply() -> None:
    model = ScriptedFakeChatModel([AIMessage(content="模型调试回复")])
    graph = build_model_playground_graph(lambda: model)
    input_messages = [HumanMessage(content="连接模型")]

    result = await graph.ainvoke({"messages": input_messages})

    assert len(input_messages) == 1
    assert input_messages[0].content == "连接模型"
    assert [message.content for message in result["messages"]] == [
        "连接模型",
        "模型调试回复",
    ]
    assert isinstance(result["messages"][-1], AIMessage)


@pytest.mark.asyncio
async def test_model_playground_streams_ai_chunks_from_answer_node() -> None:
    model = ScriptedFakeChatModel([AIMessage(content="流式回复")])
    graph = build_model_playground_graph(lambda: model)
    streamed_text: list[str] = []

    async for message, metadata in graph.astream(
        {"messages": [HumanMessage(content="测试流式输出")]},
        stream_mode="messages",
    ):
        if (
            isinstance(message, AIMessageChunk)
            and metadata.get("langgraph_node") == "model_playground_answer"
        ):
            streamed_text.append(message.text)

    assert "".join(streamed_text) == "流式回复"


@pytest.mark.asyncio
async def test_model_playground_defers_model_creation_until_invocation() -> None:
    creation_count = 0

    def create_model() -> ScriptedFakeChatModel:
        nonlocal creation_count
        creation_count += 1
        return ScriptedFakeChatModel([AIMessage(content="按需创建")])

    graph = build_model_playground_graph(create_model)

    assert creation_count == 0

    result = await graph.ainvoke({"messages": [HumanMessage(content="开始调用")]})

    assert creation_count == 1
    assert result["messages"][-1].content == "按需创建"
