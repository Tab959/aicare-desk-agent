"""校验Java请求身份、LangGraph thread、触发消息和运行截止时间的一致性。"""

from datetime import UTC, datetime

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from aicare_agent_service.contracts.decisions import MessageRole
from aicare_agent_service.graph.context import AgentRuntimeContext
from aicare_agent_service.graph.state import CustomerServiceState


class ContextConsistencyError(ValueError):
    """Java身份、thread或触发消息上下文不一致。"""


class RequestDeadlineExceededError(TimeoutError):
    """Agent run在进入分类前已经超过Java给定截止时间。"""


def context_sync_node(
    state: CustomerServiceState,
    config: RunnableConfig,
    runtime: Runtime[AgentRuntimeContext],
) -> dict[str, object]:
    """验证根图运行上下文，并清除上一个run遗留的临时决策和终态。"""
    # 1、状态身份必须与Java为本次run注入的期望身份完全一致。
    identity = state.get("identity")
    context = runtime.context
    if identity is None or identity != context.expected_identity:
        raise ContextConsistencyError("Agent状态身份与Java请求不一致")

    # 2、LangGraph thread_id只能使用Java生成的conversationId。
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id != identity.conversation_id:
        raise ContextConsistencyError("LangGraph thread与Java会话不一致")

    # 3、当前安全历史和HumanMessage必须对应本次Java触发消息。
    customer_messages = [
        message for message in state.get("safe_history", []) if message.role is MessageRole.CUSTOMER
    ]
    human_messages = [
        message for message in state.get("messages", []) if isinstance(message, HumanMessage)
    ]
    if not customer_messages or not human_messages:
        raise ContextConsistencyError("缺少当前触发消息")
    current_customer = customer_messages[-1]
    current_human = human_messages[-1]
    if (
        current_customer.message_id != identity.trigger_message_id
        or current_customer.sequence != identity.trigger_sequence
        or current_human.id != identity.trigger_message_id
        or current_customer.content != state.get("sanitized_user_message")
        or current_human.content != state.get("sanitized_user_message")
    ):
        raise ContextConsistencyError("触发消息与Java身份不一致")

    # 4、截止时间必须带时区且仍有剩余预算。
    deadline = context.request_deadline
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ContextConsistencyError("请求截止时间必须包含时区")
    if datetime.now(UTC) >= deadline.astimezone(UTC):
        raise RequestDeadlineExceededError("Agent run已超过请求截止时间")

    # 5、新run清除旧的分类、证据和终态，避免同一thread恢复时混入上一轮结果。
    return {
        "route_decision": None,
        "classification_failure": None,
        "citations": [],
        "tool_results": [],
        "handoff_suggestion": None,
        "escalation_suggestion": None,
        "final_answer": None,
    }
