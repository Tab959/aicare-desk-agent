"""提供知识库RAG的安全解析、Chunk、模型适配、严格契约和稳定错误。"""

from aicare_agent_service.rag.chunking import ChunkingLimits, chunk_sections
from aicare_agent_service.rag.contracts import (
    IndexedDocument,
    IndexResult,
    KnowledgeChunk,
    KnowledgeMetadata,
    ParsedSection,
    RagAnswer,
    RawKnowledgeDocument,
    RetrievalFilter,
    RetrievalQuery,
    RetrievalResult,
    RetrievedChunk,
)
from aicare_agent_service.rag.errors import RagError, RagErrorCode
from aicare_agent_service.rag.parsers import ParserLimits, parse_document

__all__ = [
    "ChunkingLimits",
    "IndexResult",
    "IndexedDocument",
    "KnowledgeChunk",
    "KnowledgeMetadata",
    "ParsedSection",
    "ParserLimits",
    "RagAnswer",
    "RagError",
    "RagErrorCode",
    "RawKnowledgeDocument",
    "RetrievalFilter",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievedChunk",
    "chunk_sections",
    "parse_document",
]
