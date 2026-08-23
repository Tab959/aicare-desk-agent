"""调用路由用途模型获得最小结构化分类，并转换为代码固定的路由决定。"""

import json
from functools import lru_cache
from importlib.resources import files
from typing import cast

import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from pydantic import ValidationError

from aicare_agent_service.contracts.decisions import (
    ROUTE_TARGETS,
    RouteClassification,
    RouteDecision,
)
from aicare_agent_service.graph.context import AgentRuntimeContext
from aicare_agent_service.graph.routes import RootRoute
from aicare_agent_service.graph.state import CustomerServiceState
from aicare_agent_service.models.contracts import ModelConfigurationError, ModelPurpose
from aicare_agent_service.security.contracts import (
    ClassificationFailureCode,
    RouteClassificationFailure,
)
from aicare_agent_service.security.redaction import redact_sensitive_input


@lru_cache(maxsize=1)
def _router_prompt() -> str:
    """读取随包发布的版本化路由提示词。"""
    # 1、使用包资源读取，兼容源码运行和wheel安装。
    return files("aicare_agent_service.prompts").joinpath("router.md").read_text(encoding="utf-8")


def build_router_messages(state: CustomerServiceState) -> list[BaseMessage]:
    """从已脱敏状态构造路由模型消息，不读取原始Java请求。"""
    # 1、只选择分类所需的安全文本、摘要和最小业务上下文。
    payload = {
        "currentQuestion": state["sanitized_user_message"],
        "conversationSummary": state.get("conversation_summary"),
        "businessContext": state["business_context"].model_dump(by_alias=True, mode="json"),
    }
    # 2、系统规则和用户数据使用不同消息角色，避免把正文拼进系统指令。
    return [
        SystemMessage(content=_router_prompt()),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
    ]


async def classify_node(
    state: CustomerServiceState,
    runtime: Runtime[AgentRuntimeContext],
) -> dict[str, object]:
    """执行一次结构化分类，预期故障只保存稳定失败码。"""
    # 1、创建固定ROUTING用途模型，并强制function-calling结构化输出。
    try:
        model = runtime.context.model_provider.create(ModelPurpose.ROUTING)
        structured_model = model.with_structured_output(
            RouteClassification,
            method="function_calling",
        )
        raw_result = await structured_model.ainvoke(
            build_router_messages(state),
            config={
                "run_name": "root.route.classify",
                "tags": ["root", "routing"],
                "metadata": {"node": "classify", "data_classification": "redacted"},
            },
        )
        classification = RouteClassification.model_validate(raw_result)
    except (TimeoutError, openai.APITimeoutError):
        return _classification_failure(ClassificationFailureCode.MODEL_TIMEOUT, retryable=True)
    except (ValidationError, OutputParserException):
        return _classification_failure(
            ClassificationFailureCode.INVALID_STRUCTURED_OUTPUT,
            retryable=False,
        )
    except ModelConfigurationError:
        return _classification_failure(
            ClassificationFailureCode.MODEL_UNAVAILABLE,
            retryable=False,
        )
    except openai.APIError as exc:
        return _classification_failure(
            ClassificationFailureCode.MODEL_UNAVAILABLE,
            retryable=_is_retryable_provider_error(exc),
        )

    # 2、节点目标只从代码映射补全，模型无法提供任意路由或Agent名称。
    route_code, agent_code = ROUTE_TARGETS[classification.intent]
    safe_reason = redact_sensitive_input(classification.reason).sanitized_text
    decision = RouteDecision(
        intent=classification.intent,
        route_code=route_code,
        agent_code=agent_code,
        confidence=classification.confidence,
        reason=safe_reason,
    )
    # 3、状态只保存已验证决定，不保存Provider原始消息或异常正文。
    return {"route_decision": decision, "classification_failure": None}


def build_classification_terminal(route: RootRoute) -> dict[str, object]:
    """为澄清、分类失败和不支持请求生成固定互斥终态。"""
    # 1、只接受无需调用专业子图的三个确定性分类终点。
    answers = {
        RootRoute.CLARIFY: "请补充您想咨询的游戏、订单或售后问题。",
        RootRoute.CLASSIFICATION_FALLBACK: "暂时无法准确识别您的问题，请换一种方式描述。",
        RootRoute.UNSUPPORTED: "当前AI客服暂不支持处理该请求。",
    }
    try:
        answer = answers[route]
    except KeyError as exc:
        raise ValueError("该路由不能生成分类固定终态") from exc
    # 2、显式清空其他终态，保持同一thread跨run时终态互斥。
    return {
        "final_answer": answer,
        "handoff_suggestion": None,
        "escalation_suggestion": None,
    }


def _classification_failure(
    code: ClassificationFailureCode,
    *,
    retryable: bool,
) -> dict[str, object]:
    """构造不携带Provider正文的稳定分类失败更新。"""
    # 1、失败时清空路由决定，只保存代码和重试语义。
    return {
        "route_decision": None,
        "classification_failure": RouteClassificationFailure(
            code=code,
            retryable=retryable,
        ),
    }


def _is_retryable_provider_error(error: openai.APIError) -> bool:
    """根据OpenAI兼容SDK异常类型判断Provider错误是否可有限重试。"""
    # 1、认证、权限、请求格式和资源错误不会通过原样重试恢复。
    non_retryable = (
        openai.AuthenticationError,
        openai.BadRequestError,
        openai.PermissionDeniedError,
        openai.NotFoundError,
        openai.UnprocessableEntityError,
    )
    return not isinstance(error, cast(tuple[type[BaseException], ...], non_retryable))
