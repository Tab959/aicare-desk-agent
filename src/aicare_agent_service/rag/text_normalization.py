"""提供知识文档正文的安全解码、控制字符检查和确定性规范化。"""

from __future__ import annotations

import re
import unicodedata

from aicare_agent_service.rag.errors import RagError, RagErrorCode

_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_ALLOWED_CONTROLS = {"\n", "\r", "\t"}


def decode_utf8(content: bytes) -> str:
    """严格解码UTF-8正文，不使用会掩盖伪造二进制内容的替换模式。"""
    # 1、非法UTF-8说明输入并非允许的纯文本格式。
    try:
        return content.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise RagError(RagErrorCode.UNSUPPORTED_FORMAT) from exc


def normalize_text(text: str) -> str:
    """拒绝危险控制字符并生成可复现的索引正文。"""
    # 1、除换行与制表符外，所有Unicode控制字符都拒绝进入解析结果。
    if any(unicodedata.category(char) == "Cc" and char not in _ALLOWED_CONTROLS for char in text):
        raise RagError(RagErrorCode.UNSUPPORTED_FORMAT)
    # 2、统一Unicode兼容形式、换行和每行末尾空白。
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    # 3、最多保留一个空行并移除正文两端空白，确保checksum可复现。
    return _EXCESS_BLANK_LINES.sub("\n\n", normalized).strip()
