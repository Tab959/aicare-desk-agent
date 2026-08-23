"""验证结构感知 Chunk 的 token 上限、覆盖、重叠和稳定身份。"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from aicare_agent_service.rag.chunking import ChunkingLimits, chunk_sections
from aicare_agent_service.rag.contracts import KnowledgeMetadata, ParsedSection


class CharacterTokenizer:
    """用一个字符对应一个token的确定性测试tokenizer。"""

    def __call__(self, text: str, **_: Any) -> dict[str, list[object]]:
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _metadata(*, version: int = 1) -> KnowledgeMetadata:
    return KnowledgeMetadata(
        tenant_id="tenant-1",
        knowledge_base_id="kb-1",
        document_id="doc-1",
        version=version,
        language="zh-CN",
        category="POLICY",
    )


def _section(text: str, *, version: int = 1) -> ParsedSection:
    return ParsedSection(
        metadata=_metadata(version=version),
        title_path=("交付政策", "CDK"),
        ordinal=1,
        text=text,
    )


def _body(content: str) -> str:
    return content.split("\n\n", maxsplit=1)[1]


def test_chunker_keeps_context_prefix_and_hard_token_limit() -> None:
    chunks = chunk_sections(
        document_title="商城帮助",
        sections=(_section("第一句。第二句。第三句。" * 20),),
        tokenizer=CharacterTokenizer(),
        hmac_key=b"k" * 32,
        limits=ChunkingLimits(target_tokens=80, max_tokens=100, overlap_tokens=12),
    )

    assert len(chunks) > 1
    assert all(
        chunk.content.startswith("文档：商城帮助\n章节：交付政策 > CDK\n\n") for chunk in chunks
    )
    assert all(chunk.token_count <= 100 for chunk in chunks)


def test_chunker_covers_normalized_body_and_only_repeats_overlap() -> None:
    text = "甲" * 70 + "。" + "乙" * 70 + "。" + "丙" * 70 + "。"
    chunks = chunk_sections(
        document_title="帮助",
        sections=(_section(text),),
        tokenizer=CharacterTokenizer(),
        hmac_key=b"k" * 32,
        limits=ChunkingLimits(target_tokens=70, max_tokens=80, overlap_tokens=10),
    )
    bodies = [_body(chunk.content) for chunk in chunks]

    rebuilt = bodies[0]
    for previous, current in pairwise(bodies):
        overlap = min(10, len(previous), len(current))
        assert previous[-overlap:] == current[:overlap]
        rebuilt += current[overlap:]
    assert rebuilt == text


def test_chunk_ids_and_checksums_are_stable_but_change_with_content_or_version() -> None:
    arguments = {
        "document_title": "帮助",
        "tokenizer": CharacterTokenizer(),
        "hmac_key": b"k" * 32,
        "limits": ChunkingLimits(target_tokens=100, max_tokens=120, overlap_tokens=20),
    }
    first = chunk_sections(sections=(_section("稳定正文。"),), **arguments)
    repeated = chunk_sections(sections=(_section("稳定正文。"),), **arguments)
    changed = chunk_sections(sections=(_section("变化正文。"),), **arguments)
    versioned = chunk_sections(sections=(_section("稳定正文。", version=2),), **arguments)

    assert first == repeated
    assert first[0].content_checksum != changed[0].content_checksum
    assert first[0].chunk_id != changed[0].chunk_id
    assert first[0].chunk_id != versioned[0].chunk_id


def test_long_unbroken_paragraph_falls_back_to_token_windows_without_loss() -> None:
    text = "超长段落" * 100
    chunks = chunk_sections(
        document_title="帮助",
        sections=(_section(text),),
        tokenizer=CharacterTokenizer(),
        hmac_key=b"k" * 32,
        limits=ChunkingLimits(target_tokens=90, max_tokens=100, overlap_tokens=15),
    )
    bodies = [_body(chunk.content) for chunk in chunks]
    rebuilt = bodies[0] + "".join(body[15:] for body in bodies[1:])

    assert rebuilt == text
    assert all(chunk.token_count <= 100 for chunk in chunks)
