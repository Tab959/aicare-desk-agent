"""用DeepSeek结构化改写脱敏查询，并在可预期故障时回退到安全原问题。"""

from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from functools import lru_cache
from importlib.resources import files
from typing import Annotated, Literal

import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from aicare_agent_service.models.contracts import (
    ChatModelProvider,
    ModelConfigurationError,
    ModelPurpose,
)
from aicare_agent_service.security.redaction import redact_sensitive_input


class RewriteIntent(StrEnum):
    """模型可以建议的低风险检索意图提示，不包含路由或工具动作。"""

    PRODUCT = "PRODUCT"
    POLICY = "POLICY"
    DELIVERY = "DELIVERY"
    TROUBLESHOOTING = "TROUBLESHOOTING"
    GENERAL = "GENERAL"


class QueryRewriteDecision(BaseModel):
    """DeepSeek只能返回规范化文本、有限提示和置信度。"""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    normalized_query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
    ]
    language_hint: Literal["zh-CN", "en", "mixed"]
    intent_hint: RewriteIntent
    confidence: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


class QueryRewriteResult(BaseModel):
    """查询改写后的安全运行结果，保留是否降级但不保留模型原始响应。"""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
    redacted_original: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
    ]
    language_hint: Literal["zh-CN", "en", "mixed"] | None = None
    intent_hint: RewriteIntent | None = None
    confidence: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)] = 0.0
    used_fallback: bool


@lru_cache(maxsize=1)
def _rewrite_prompt() -> str:
    """读取随wheel发布的版本化查询改写提示词。"""
    # 1、包资源API同时支持源码目录和安装后的wheel。
    return (
        files("aicare_agent_service.prompts")
        .joinpath("rag_query_rewrite.md")
        .read_text(encoding="utf-8")
    )


def build_rewrite_messages(redacted_question: str) -> list[BaseMessage]:
    """把系统规则与已经脱敏的用户查询分角色构造成模型输入。"""
    # 1、空文本不得进入模型；调用者需要先给用户澄清而不是生成任意查询。
    if not redacted_question.strip():
        raise ValueError("RAG_QUERY_EMPTY")
    # 2、用户文本只作为JSON数据放入HumanMessage，不能拼进系统指令。
    return [
        SystemMessage(content=_rewrite_prompt()),
        HumanMessage(
            content=json.dumps(
                {"question": redacted_question},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ),
    ]


class QueryRewriter:
    """执行一次有deadline的结构化改写，并以固定阈值决定是否采用。"""

    def __init__(
        self, *, model_provider: ChatModelProvider, confidence_threshold: float = 0.65
    ) -> None:
        """绑定模型Provider和确定性的最低采用置信度。"""
        # 1、阈值必须位于开区间，避免配置成永远采用或永远回退。
        if not 0 < confidence_threshold < 1:
            raise ValueError("RAG_REWRITE_CONFIDENCE_INVALID")
        # 2、保存Provider，不在构造阶段创建或调用模型。
        self._provider = model_provider
        self._confidence_threshold = confidence_threshold

    @staticmethod
    def output_schema() -> type[QueryRewriteDecision]:
        """暴露模型允许输出的严格Schema供契约测试和追踪检查。"""
        # 1、返回类而不是实例，LangChain可直接绑定为结构化工具Schema。
        return QueryRewriteDecision

    async def rewrite(self, question: str, *, deadline: float) -> QueryRewriteResult:
        """先脱敏，再调用DeepSeek；失败或低置信度使用脱敏原问题。"""
        # 1、任何模型、Embedding和Trace之前先执行现有确定性敏感信息替换。
        redacted = redact_sensitive_input(question).sanitized_text
        if not redacted:
            raise ValueError("RAG_QUERY_EMPTY")
        # 2、只捕获模型边界的可预期故障，异常正文绝不进入回退查询或结果。
        try:
            model = self._provider.create(ModelPurpose.ROUTING)
            structured = model.with_structured_output(
                QueryRewriteDecision,
                method="function_calling",
            )
            async with asyncio.timeout_at(deadline):
                raw = await structured.ainvoke(
                    build_rewrite_messages(redacted),
                    config={
                        "run_name": "rag.query.rewrite",
                        "tags": ["rag", "query-rewrite"],
                        "metadata": {
                            "node": "query_rewrite",
                            "prompt_version": "rag-query-rewrite-v1",
                            "data_classification": "redacted",
                        },
                    },
                )
            decision = QueryRewriteDecision.model_validate(raw)
        except (
            TimeoutError,
            openai.APIError,
            OutputParserException,
            ValidationError,
            ModelConfigurationError,
        ):
            return self._fallback(redacted)
        # 3、只有达到固定阈值的单条规范化文本才被采用，提示字段不生成业务过滤器。
        if decision.confidence < self._confidence_threshold:
            return self._fallback(redacted)
        return QueryRewriteResult(
            query=decision.normalized_query,
            redacted_original=redacted,
            language_hint=decision.language_hint,
            intent_hint=decision.intent_hint,
            confidence=decision.confidence,
            used_fallback=False,
        )

    @staticmethod
    def _fallback(redacted: str) -> QueryRewriteResult:
        """构造不包含Provider异常信息的固定安全回退结果。"""
        # 1、仅复用已经脱敏的原问题，不拼接错误类型、消息或模型输出。
        return QueryRewriteResult(
            query=redacted,
            redacted_original=redacted,
            confidence=0.0,
            used_fallback=True,
        )
