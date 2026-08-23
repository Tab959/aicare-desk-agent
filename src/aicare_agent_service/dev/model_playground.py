"""构建用于本地验证模型调用与流式输出的最小消息图。

本文件只定义图的组装方式。调用者通过 ``model_factory`` 注入模型创建逻辑，
因此图不绑定某个模型厂商，也便于测试时换成 Fake 模型。它不承担正式客服路由。
"""

# Callable 用于描述“可以被调用的对象”，这里表示一个无参数模型工厂函数。
from collections.abc import Callable

# BaseChatModel 是所有 LangChain 聊天模型实现共同继承的抽象基类。
from langchain_core.language_models import BaseChatModel

# BaseMessage 是 HumanMessage、AIMessage 等消息类型的共同基类。
from langchain_core.messages import BaseMessage

# 导入图的入口、出口、内置消息状态以及图构建器。
from langgraph.graph import END, START, MessagesState, StateGraph

# CompiledStateGraph 是 StateGraph.compile() 返回的可执行图类型。
from langgraph.graph.state import CompiledStateGraph


def build_model_playground_graph(
    # ``Callable[[], BaseChatModel]`` 表示：不接收参数，调用后返回聊天模型的函数。
    model_factory: Callable[[], BaseChatModel],
) -> CompiledStateGraph[MessagesState, None, MessagesState, MessagesState]:
    """构建只用于本地模型与流式输出调试的最小消息图。

    返回值泛型依次描述图状态、上下文、输入和输出结构；此图不需要额外运行时上下文，
    所以第二项是 ``None``。
    """

    # 内部函数只服务于本次建图；``async`` 表示它需要异步等待模型。
    async def model_playground_answer(
        # state 是节点执行时收到的 MessagesState，至少包含 ``messages`` 字段。
        state: MessagesState,
    ) -> dict[str, list[BaseMessage]]:
        """调用一次模型，并把回复作为消息状态的局部更新返回。"""
        # 到节点真正运行时才调用工厂，避免模块加载阶段要求模型密钥存在。
        model = model_factory()
        # ``await`` 异步等待模型处理完整消息历史，不阻塞事件循环线程。
        reply = await model.ainvoke(state["messages"])
        # 返回新消息列表；LangGraph 的消息 reducer 会负责与已有消息正确合并。
        return {"messages": [reply]}

    # 创建以 LangGraph 内置 MessagesState 为状态定义的图构建器。
    builder = StateGraph(MessagesState)
    # 将内部异步函数注册为名为 model_playground_answer 的节点。
    builder.add_node("model_playground_answer", model_playground_answer)
    # 固定执行路径的第一条边：START → 模型节点。
    builder.add_edge(START, "model_playground_answer")
    # 固定执行路径的第二条边：模型节点 → END。
    builder.add_edge("model_playground_answer", END)
    # 编译并返回可执行图；调用者可对它使用 ainvoke、astream 等方法。
    return builder.compile()
