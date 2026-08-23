"""使用BGE兼容tokenizer把结构化Section切成有界、可复现的知识Chunk。"""

from __future__ import annotations

import hashlib
import hmac
import re
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Protocol

from aicare_agent_service.rag.contracts import KnowledgeChunk, ParsedSection
from aicare_agent_service.rag.text_normalization import normalize_text

_SENTENCE_END = re.compile(r"[。！？.!?](?:[\"'”’）】》])?\s*")
_PARAGRAPH_END = re.compile(r"\n\s*\n")


class OffsetTokenizer(Protocol):
    """Chunk所需的Hugging Face fast tokenizer最小协议。"""

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ChunkingLimits:
    """Chunk目标、硬上限和相邻窗口重叠token数量。"""

    target_tokens: int = 512
    max_tokens: int = 640
    overlap_tokens: int = 80

    def __post_init__(self) -> None:
        """拒绝无法前进或突破硬上限的切分预算。"""
        # 1、目标必须为正且不能超过硬上限。
        if self.target_tokens < 1 or self.max_tokens < self.target_tokens:
            raise ValueError("RAG_CHUNK_LIMIT_INVALID")
        # 2、重叠必须小于目标，确保每个窗口都能向后推进。
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.target_tokens:
            raise ValueError("RAG_CHUNK_OVERLAP_INVALID")


_DEFAULT_CHUNKING_LIMITS = ChunkingLimits()


def _token_offsets(tokenizer: OffsetTokenizer, text: str) -> tuple[tuple[int, int], ...]:
    """读取fast tokenizer的字符offset并拒绝不支持offset的实现。"""
    # 1、禁用特殊token，Chunk计数只覆盖真实索引文本。
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.get("offset_mapping")
    if not isinstance(offsets, list) or any(
        not isinstance(item, (list, tuple)) or len(item) != 2 for item in offsets
    ):
        raise ValueError("RAG_TOKENIZER_OFFSETS_REQUIRED")
    # 2、过滤零宽特殊offset并转换为不可变整数元组。
    return tuple((int(item[0]), int(item[1])) for item in offsets if int(item[1]) > int(item[0]))


def _boundary_tokens(text: str, offsets: tuple[tuple[int, int], ...]) -> tuple[set[int], set[int]]:
    """把段落和句子字符边界映射为token结束位置。"""
    # 1、token字符结束位置有序，可用二分把字符边界映射到窗口端点。
    token_ends = [end for _, end in offsets]

    def positions(pattern: re.Pattern[str]) -> set[int]:
        return {
            min(bisect_left(token_ends, match.end()) + 1, len(offsets))
            for match in pattern.finditer(text)
        }

    # 2、调用方优先段落边界，其次句子边界，最后按token硬切。
    return positions(_PARAGRAPH_END), positions(_SENTENCE_END)


def _choose_end(
    *,
    start: int,
    desired_end: int,
    hard_end: int,
    overlap_tokens: int,
    paragraph_ends: set[int],
    sentence_ends: set[int],
) -> int:
    """优先在接近目标的结构边界结束，并保证下一窗口能够推进。"""
    # 1、边界必须越过重叠区，否则下一窗口会停在原位置。
    minimum = start + overlap_tokens + 1
    for boundaries in (paragraph_ends, sentence_ends):
        candidates = [position for position in boundaries if minimum <= position <= desired_end]
        if candidates:
            return max(candidates)
    # 2、没有可用自然边界时按token目标兜底，仍不突破硬上限。
    return max(minimum, min(desired_end, hard_end))


def _context_prefix(document_title: str, title_path: tuple[str, ...]) -> str:
    """生成供检索使用但不改变引用路径的有限上下文前缀。"""
    # 1、文档标题始终存在，章节路径为空时不制造伪章节。
    prefix = f"文档：{normalize_text(document_title)}"
    if title_path:
        prefix += f"\n章节：{' > '.join(title_path)}"
    # 2、空行明确分隔上下文和原始正文。
    return prefix + "\n\n"


def _chunk_id(
    section: ParsedSection,
    *,
    ordinal: int,
    checksum: str,
    hmac_key: bytes,
) -> str:
    """用根密钥派生租户密钥后生成不可枚举的稳定Chunk ID。"""
    # 1、不同租户先派生不同HMAC密钥，阻止相同文档身份产生相同ID。
    tenant_key = hmac.new(hmac_key, section.metadata.tenant_id.encode(), hashlib.sha256).digest()
    # 2、身份、版本、结构、顺序和正文checksum全部进入签名消息。
    message = "\x1f".join(
        (
            section.metadata.document_id,
            str(section.metadata.version),
            *section.title_path,
            str(ordinal),
            checksum,
        )
    ).encode()
    return hmac.new(tenant_key, message, hashlib.sha256).hexdigest()


def chunk_sections(
    *,
    document_title: str,
    sections: tuple[ParsedSection, ...],
    tokenizer: OffsetTokenizer,
    hmac_key: bytes,
    limits: ChunkingLimits = _DEFAULT_CHUNKING_LIMITS,
) -> tuple[KnowledgeChunk, ...]:
    """按结构边界和token窗口切分Section并生成稳定Chunk。"""
    # 1、生产HMAC根密钥至少256 bit，所有Section必须属于同一文档版本。
    if len(hmac_key) < 32:
        raise ValueError("RAG_CHUNK_HMAC_KEY_TOO_SHORT")
    if not sections:
        return ()
    first_metadata = sections[0].metadata
    if any(section.metadata != first_metadata for section in sections):
        raise ValueError("RAG_CHUNK_DOCUMENT_MISMATCH")
    chunks: list[KnowledgeChunk] = []
    ordinal = 1
    # 2、每个Section继承标题路径，前缀token从正文预算中扣除。
    for section in sections:
        body = normalize_text(section.text)
        prefix = _context_prefix(document_title, section.title_path)
        prefix_tokens = len(_token_offsets(tokenizer, prefix))
        target_body = limits.target_tokens - prefix_tokens
        max_body = limits.max_tokens - prefix_tokens
        if target_body <= limits.overlap_tokens or max_body <= limits.overlap_tokens:
            raise ValueError("RAG_CHUNK_PREFIX_TOO_LARGE")
        offsets = _token_offsets(tokenizer, body)
        if not offsets:
            continue
        paragraph_ends, sentence_ends = _boundary_tokens(body, offsets)
        start = 0
        # 3、窗口优先在段落/句子结束，超长段落最终按token offset切分。
        while start < len(offsets):
            desired_end = min(start + target_body, len(offsets))
            hard_end = min(start + max_body, len(offsets))
            if desired_end == len(offsets):
                end = len(offsets)
            else:
                end = _choose_end(
                    start=start,
                    desired_end=desired_end,
                    hard_end=hard_end,
                    overlap_tokens=limits.overlap_tokens,
                    paragraph_ends=paragraph_ends,
                    sentence_ends=sentence_ends,
                )
            body_text = body[offsets[start][0] : offsets[end - 1][1]]
            checksum = hashlib.sha256(body_text.encode()).hexdigest()
            content = prefix + body_text
            token_count = len(_token_offsets(tokenizer, content))
            if token_count > limits.max_tokens:
                raise ValueError("RAG_CHUNK_TOKEN_LIMIT_EXCEEDED")
            chunks.append(
                KnowledgeChunk(
                    metadata=section.metadata,
                    chunk_id=_chunk_id(
                        section,
                        ordinal=ordinal,
                        checksum=checksum,
                        hmac_key=hmac_key,
                    ),
                    title_path=section.title_path,
                    ordinal=ordinal,
                    content=content,
                    token_count=token_count,
                    content_checksum=checksum,
                )
            )
            ordinal += 1
            if end == len(offsets):
                break
            start = end - limits.overlap_tokens
    # 4、返回不可变契约，正文只在配置的相邻重叠窗口重复。
    return tuple(chunks)
