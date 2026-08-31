"""使用结构化审核检查RAG回答的忠实度、引用覆盖和业务实时事实边界。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from enum import StrEnum
from functools import lru_cache
from importlib.resources import files

import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, ValidationError

from aicare_agent_service.models.contracts import (
    ChatModelProvider,
    ModelConfigurationError,
    ModelPurpose,
)
from aicare_agent_service.rag.contracts import RagAnswer, RetrievedChunk
from aicare_agent_service.rag.errors import RagError, RagErrorCode
from aicare_agent_service.security.redaction import redact_sensitive_input


class FaithfulnessReasonCode(StrEnum):
    """审核模型只能返回的低基数结论码。"""

    PASS = "PASS"
    MISSING_CITATION = "MISSING_CITATION"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    REALTIME_FACT_BOUNDARY = "REALTIME_FACT_BOUNDARY"
    DOCUMENT_INSTRUCTION_FOLLOWED = "DOCUMENT_INSTRUCTION_FOLLOWED"


class FaithfulnessDecision(BaseModel):
    """知识回答四项独立审核结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    citation_coverage: bool
    factually_supported: bool
    realtime_boundary_respected: bool
    document_instructions_ignored: bool
    reason_code: FaithfulnessReasonCode

    @property
    def passed(self) -> bool:
        """由代码计算全部门禁是否同时通过。"""
        # 1、四项布尔门禁和PASS原因码必须完全一致。
        return (
            self.citation_coverage
            and self.factually_supported
            and self.realtime_boundary_respected
            and self.document_instructions_ignored
            and self.reason_code is FaithfulnessReasonCode.PASS
        )


@lru_cache(maxsize=1)
def _faithfulness_prompt() -> str:
    """读取随包发布的版本化忠实度审核提示词。"""
    # 1、包资源确保部署后不依赖工作目录中的临时文件。
    return files("aicare_agent_service.prompts").joinpath("rag_faithfulness.md").read_text("utf-8")


def build_faithfulness_messages(
    *, query: str, candidates: Sequence[RetrievedChunk], answer: RagAnswer
) -> list[BaseMessage]:
    """构造只含脱敏有限证据和候选回答的审核输入。"""
    # 1、证据使用与生成阶段相同的K1至K6稳定标记。
    evidence = [
        {
            "marker": f"K{index}",
            "content": redact_sensitive_input(chunk.content).sanitized_text,
        }
        for index, chunk in enumerate(candidates[:6], start=1)
    ]
    # 2、审核输入不包含租户、用户、向量、ES响应或Provider异常。
    payload = {
        "question": redact_sensitive_input(query).sanitized_text,
        "evidence": evidence,
        "answer": answer.content,
    }
    # 3、系统规则和不可信数据分离。
    return [
        SystemMessage(content=_faithfulness_prompt()),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))),
    ]


class RagFaithfulnessVerifier:
    """执行单次DeepSeek结构化知识回答审核。"""

    def __init__(self, model_provider: ChatModelProvider) -> None:
        """绑定统一模型Provider。"""
        # 1、审核模型在调用时按REVIEW用途创建。
        self._provider = model_provider

    async def verify(
        self,
        *,
        query: str,
        candidates: Sequence[RetrievedChunk],
        answer: RagAnswer,
        deadline: float,
        repair_count: int,
    ) -> FaithfulnessDecision:
        """检查一次回答并返回不含自由文本的有限决策。"""
        # 1、严格结构化调用与整条子图共用绝对deadline。
        try:
            model = self._provider.create(ModelPurpose.REVIEW)
            structured = model.with_structured_output(
                FaithfulnessDecision,
                method="function_calling",
            )
            async with asyncio.timeout_at(deadline):
                raw = await structured.ainvoke(
                    build_faithfulness_messages(
                        query=query,
                        candidates=candidates,
                        answer=answer,
                    ),
                    config={
                        "run_name": "rag.answer.verify",
                        "tags": ["rag", "review"],
                        "metadata": {
                            "node": "verify",
                            "prompt_version": "rag-faithfulness-v1",
                            "candidate_count": min(len(candidates), 6),
                            "repair_count": repair_count,
                            "data_classification": "redacted",
                        },
                    },
                )
            return FaithfulnessDecision.model_validate(raw)
        except (
            TimeoutError,
            openai.APIError,
            OutputParserException,
            ValidationError,
            ModelConfigurationError,
        ) as exc:
            raise RagError(RagErrorCode.MODEL_UNAVAILABLE) from exc
