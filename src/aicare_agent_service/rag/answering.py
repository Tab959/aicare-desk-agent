"""使用有限候选生成带稳定引用标记的知识回答，并拒绝候选外引用。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from functools import lru_cache
from importlib.resources import files
from typing import Annotated

import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from aicare_agent_service.contracts import Citation
from aicare_agent_service.models.contracts import (
    ChatModelProvider,
    ModelConfigurationError,
    ModelPurpose,
)
from aicare_agent_service.rag.contracts import RagAnswer, RetrievedChunk
from aicare_agent_service.rag.errors import RagError, RagErrorCode
from aicare_agent_service.security.redaction import redact_sensitive_input

CitationMarker = Annotated[str, StringConstraints(pattern=r"^K[1-6]$")]


class GeneratedRagAnswer(BaseModel):
    """DeepSeek允许生成的正文和候选引用标记。"""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    content: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12_000)
    ]
    citation_markers: Annotated[tuple[CitationMarker, ...], Field(min_length=1, max_length=6)]


@lru_cache(maxsize=1)
def _answer_prompt() -> str:
    """读取随包发布的版本化知识回答提示词。"""
    # 1、使用包资源保证源码、wheel和LangGraph Server读取同一版本。
    return files("aicare_agent_service.prompts").joinpath("rag_answer.md").read_text("utf-8")


def build_answer_messages(
    *,
    query: str,
    candidates: Sequence[RetrievedChunk],
    previous_answer: str | None = None,
    repair_reason: str | None = None,
) -> list[BaseMessage]:
    """把脱敏问题与有限候选作为不可信数据构造成模型消息。"""
    # 1、候选最多六条，只提供回答需要的正文和稳定引用身份。
    evidence = [
        {
            "marker": f"K{index}",
            "documentId": chunk.metadata.document_id,
            "version": chunk.metadata.version,
            "titlePath": list(chunk.citation.title_path),
            "content": redact_sensitive_input(chunk.content).sanitized_text,
        }
        for index, chunk in enumerate(candidates[:6], start=1)
    ]
    # 2、修正只携带上次安全正文和有限原因码，不携带审核模型原始输出。
    payload = {
        "question": redact_sensitive_input(query).sanitized_text,
        "evidence": evidence,
        "previousAnswer": previous_answer,
        "repairReason": repair_reason,
    }
    # 3、规则与不可信数据分角色发送，文档正文不能成为系统指令。
    return [
        SystemMessage(content=_answer_prompt()),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
    ]


class RagAnswerGenerator:
    """执行单次结构化回答生成或受限修正。"""

    def __init__(self, model_provider: ChatModelProvider) -> None:
        """绑定统一模型Provider。"""
        # 1、模型在每次调用时按ANSWER用途创建，以应用冻结参数。
        self._provider = model_provider

    async def generate(
        self,
        *,
        query: str,
        candidates: Sequence[RetrievedChunk],
        deadline: float,
        previous_answer: str | None = None,
        repair_reason: str | None = None,
    ) -> RagAnswer:
        """生成只引用现有候选且引用标记实际出现在正文中的回答。"""
        # 1、没有候选时禁止调用模型。
        bounded_candidates = tuple(candidates[:6])
        if not bounded_candidates:
            raise RagError(RagErrorCode.INSUFFICIENT_EVIDENCE)
        # 2、结构化调用受整条子图绝对deadline限制，并附加低基数追踪元数据。
        try:
            model = self._provider.create(ModelPurpose.ANSWER)
            structured = model.with_structured_output(
                GeneratedRagAnswer,
                method="function_calling",
            )
            async with asyncio.timeout_at(deadline):
                raw = await structured.ainvoke(
                    build_answer_messages(
                        query=query,
                        candidates=bounded_candidates,
                        previous_answer=previous_answer,
                        repair_reason=repair_reason,
                    ),
                    config={
                        "run_name": "rag.answer.generate",
                        "tags": ["rag", "answer"],
                        "metadata": {
                            "node": "generate" if previous_answer is None else "repair",
                            "prompt_version": "rag-answer-v1",
                            "candidate_count": len(bounded_candidates),
                            "data_classification": "redacted",
                        },
                    },
                )
            generated = GeneratedRagAnswer.model_validate(raw)
        except (
            TimeoutError,
            openai.APIError,
            OutputParserException,
            ValidationError,
            ModelConfigurationError,
        ) as exc:
            raise RagError(RagErrorCode.MODEL_UNAVAILABLE) from exc
        # 3、引用标记必须存在、不能重复且必须逐一出现在回答正文。
        markers = tuple(dict.fromkeys(generated.citation_markers))
        if any(int(marker[1:]) > len(bounded_candidates) for marker in markers):
            raise RagError(RagErrorCode.INSUFFICIENT_EVIDENCE)
        if any(f"[{marker}]" not in generated.content for marker in markers):
            raise RagError(RagErrorCode.INSUFFICIENT_EVIDENCE)
        citations: tuple[Citation, ...] = tuple(
            bounded_candidates[int(marker[1:]) - 1].citation for marker in markers
        )
        # 4、最终正文再脱敏并交给RagAnswer执行引用充足性校验。
        return RagAnswer(
            content=redact_sensitive_input(generated.content).sanitized_text,
            sufficient_evidence=True,
            citations=citations,
        )
