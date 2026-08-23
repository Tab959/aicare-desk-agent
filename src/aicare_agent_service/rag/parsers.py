"""从受限bytes安全解析纯文本、Markdown、HTML、DOCX和文字型PDF。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from zipfile import BadZipFile, ZipFile

from bs4 import BeautifulSoup, Tag
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from aicare_agent_service.rag.contracts import ParsedSection, RawKnowledgeDocument
from aicare_agent_service.rag.errors import RagError, RagErrorCode
from aicare_agent_service.rag.text_normalization import decode_utf8, normalize_text

_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_PDF_MIME = "application/pdf"
_MARKDOWN_MIMES = {"text/markdown", "text/x-markdown"}
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_DOCX_MAX_ENTRIES = 2_048
_DOCX_MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_DOCX_MAX_RATIO = 100


@dataclass(frozen=True, slots=True)
class ParserLimits:
    """解析阶段的文件、页数和提取字符硬门禁。"""

    max_bytes: int = 20 * 1024 * 1024
    max_pages: int = 300
    max_characters: int = 2_000_000


_DEFAULT_PARSER_LIMITS = ParserLimits()


def _build_sections(
    document: RawKnowledgeDocument,
    blocks: list[tuple[tuple[str, ...], str]],
    limits: ParserLimits,
) -> tuple[ParsedSection, ...]:
    """规范化区块并创建连续有序的严格Section契约。"""
    # 1、逐块规范化，空区块不进入索引。
    normalized = [(path, normalize_text(text)) for path, text in blocks]
    normalized = [(path, text) for path, text in normalized if text]
    # 2、总字符数超限或完全没有正文时稳定失败。
    if sum(len(text) for _, text in normalized) > limits.max_characters:
        raise RagError(RagErrorCode.DOCUMENT_TOO_LARGE)
    if not normalized:
        raise RagError(RagErrorCode.NO_EXTRACTABLE_TEXT)
    # 3、ordinal从1连续编号，便于后续生成稳定Chunk身份。
    return tuple(
        ParsedSection(
            metadata=document.metadata,
            title_path=path,
            ordinal=index,
            text=text,
        )
        for index, (path, text) in enumerate(normalized, start=1)
    )


def _parse_plain(content: bytes) -> list[tuple[tuple[str, ...], str]]:
    """把UTF-8纯文本按段落边界解析。"""
    # 1、二进制格式魔数不得伪装成纯文本MIME。
    if content.startswith((b"%PDF-", b"PK\x03\x04")):
        raise RagError(RagErrorCode.UNSUPPORTED_FORMAT)
    # 2、保留段落内部换行，只在空行处分段。
    text = normalize_text(decode_utf8(content))
    return [((), block) for block in re.split(r"\n\s*\n", text) if block.strip()]


def _parse_markdown(content: bytes) -> list[tuple[tuple[str, ...], str]]:
    """按标题、列表、表格和代码围栏边界解析Markdown。"""
    # 1、严格解码后逐行扫描，维护最多六级标题路径。
    lines = normalize_text(decode_utf8(content)).splitlines()
    title_path: list[str] = []
    blocks: list[tuple[tuple[str, ...], str]] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            blocks.append((tuple(title_path), "\n".join(pending)))
            pending.clear()

    index = 0
    while index < len(lines):
        line = lines[index]
        heading = _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            title_path[level - 1 :] = [heading.group(2).strip()]
            index += 1
            continue
        if line.startswith("```"):
            flush()
            fenced = [line]
            index += 1
            while index < len(lines):
                fenced.append(lines[index])
                if lines[index].startswith("```"):
                    index += 1
                    break
                index += 1
            blocks.append((tuple(title_path), "\n".join(fenced)))
            continue
        if not line.strip():
            flush()
            index += 1
            continue
        pending.append(line)
        index += 1
    flush()
    return blocks


def _parse_html(content: bytes) -> list[tuple[tuple[str, ...], str]]:
    """移除主动内容并按HTML标题、段落、列表、代码和表格解析。"""
    # 1、要求HTML内容特征与MIME一致，避免任意纯文本被当作HTML。
    text = decode_utf8(content)
    lowered = text.lstrip().lower()
    if not (lowered.startswith("<!doctype html") or "<html" in lowered):
        raise RagError(RagErrorCode.UNSUPPORTED_FORMAT)
    # 2、主动内容和资源标签完整删除，解析器不请求其中的URL。
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup.find_all(["script", "style", "noscript", "template", "svg", "img", "iframe"]):
        tag.decompose()
    # 3、按文档顺序维护标题路径并抽取有限结构块。
    title_path: list[str] = []
    blocks: list[tuple[tuple[str, ...], str]] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table"]):
        assert isinstance(element, Tag)
        if element.name.startswith("h"):
            level = int(element.name[1])
            title_path[level - 1 :] = [element.get_text(" ", strip=True)]
            continue
        if element.find_parent(["table", "pre"]):
            continue
        if element.name == "table":
            rows = [
                " | ".join(cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"]))
                for row in element.find_all("tr")
            ]
            value = "\n".join(row for row in rows if row)
        else:
            value = element.get_text("\n" if element.name == "pre" else " ", strip=True)
            if element.name == "li":
                value = f"- {value}"
        if value:
            blocks.append((tuple(title_path), value))
    return blocks


def _validate_docx_archive(content: bytes) -> None:
    """在python-docx解压前阻断伪造、宏、外链和压缩炸弹。"""
    # 1、DOCX必须是含核心Office条目的ZIP包。
    try:
        with ZipFile(BytesIO(content)) as archive:
            entries = archive.infolist()
            names = {entry.filename for entry in entries}
            if "[Content_Types].xml" not in names or "word/document.xml" not in names:
                raise RagError(RagErrorCode.UNSUPPORTED_FORMAT)
            # 2、限制条目数、解压总量和高压缩比条目。
            total_size = sum(entry.file_size for entry in entries)
            if len(entries) > _DOCX_MAX_ENTRIES or total_size > _DOCX_MAX_UNCOMPRESSED_BYTES:
                raise RagError(RagErrorCode.DOCUMENT_TOO_LARGE)
            if any(
                entry.file_size > 1024 * 1024
                and entry.file_size / max(entry.compress_size, 1) > _DOCX_MAX_RATIO
                for entry in entries
            ):
                raise RagError(RagErrorCode.DOCUMENT_TOO_LARGE)
            # 3、宏和任何External关系都不允许进入知识解析链路。
            if any(name.lower().endswith("vbaproject.bin") for name in names):
                raise RagError(RagErrorCode.UNSUPPORTED_FORMAT)
            for name in names:
                if name.endswith(".rels") and b'TargetMode="External"' in archive.read(name):
                    raise RagError(RagErrorCode.UNSUPPORTED_FORMAT)
    except BadZipFile as exc:
        raise RagError(RagErrorCode.UNSUPPORTED_FORMAT) from exc


def _parse_docx(content: bytes) -> list[tuple[tuple[str, ...], str]]:
    """按DOCX正文顺序保留标题、段落和表格行。"""
    # 1、先完成ZIP安全门禁，再交给python-docx解析。
    _validate_docx_archive(content)
    try:
        document = Document(BytesIO(content))
    except Exception as exc:
        raise RagError(RagErrorCode.UNSUPPORTED_FORMAT) from exc
    # 2、直接遍历body子元素以保留段落与表格的原始顺序。
    title_path: list[str] = []
    blocks: list[tuple[tuple[str, ...], str]] = []
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style_name = paragraph.style.name if paragraph.style is not None else ""
            heading = re.fullmatch(r"Heading ([1-6])", style_name)
            if heading:
                level = int(heading.group(1))
                title_path[level - 1 :] = [text]
            else:
                blocks.append((tuple(title_path), text))
        elif child.tag.endswith("}tbl"):
            table = Table(child, document)
            for row in table.rows:
                value = " | ".join(cell.text.strip() for cell in row.cells)
                if value.strip(" |"):
                    blocks.append((tuple(title_path), value))
    return blocks


def _parse_pdf(content: bytes, limits: ParserLimits) -> list[tuple[tuple[str, ...], str]]:
    """解析未加密且包含文本层的有限页数PDF。"""
    # 1、魔数必须与PDF MIME匹配，解析异常转换为稳定格式错误。
    if not content.startswith(b"%PDF-"):
        raise RagError(RagErrorCode.UNSUPPORTED_FORMAT)
    try:
        reader = PdfReader(BytesIO(content), strict=True)
    except Exception as exc:
        raise RagError(RagErrorCode.UNSUPPORTED_FORMAT) from exc
    # 2、加密文档不尝试猜测密码，页数超限在正文提取前阻断。
    if reader.is_encrypted:
        raise RagError(RagErrorCode.NO_EXTRACTABLE_TEXT)
    if len(reader.pages) > limits.max_pages:
        raise RagError(RagErrorCode.DOCUMENT_TOO_LARGE)
    # 3、每页单独保留引用位置；无文本层最终由统一门禁返回需要OCR语义。
    blocks: list[tuple[tuple[str, ...], str]] = []
    try:
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                blocks.append(((f"第 {index} 页",), text))
    except Exception as exc:
        raise RagError(RagErrorCode.NO_EXTRACTABLE_TEXT) from exc
    return blocks


def parse_document(
    document: RawKnowledgeDocument,
    *,
    limits: ParserLimits = _DEFAULT_PARSER_LIMITS,
) -> tuple[ParsedSection, ...]:
    """根据允许MIME和内容特征从bytes解析严格Section。"""
    # 1、即使调用方绕过Pydantic构造，也再次执行原始字节上限门禁。
    if len(document.content) > limits.max_bytes:
        raise RagError(RagErrorCode.DOCUMENT_TOO_LARGE)
    # 2、按有限注册表选择解析器，不访问file_name路径或source_uri网络地址。
    media_type = document.media_type.lower().split(";", maxsplit=1)[0].strip()
    if media_type == "text/plain":
        blocks = _parse_plain(document.content)
    elif media_type in _MARKDOWN_MIMES:
        blocks = _parse_markdown(document.content)
    elif media_type == "text/html":
        blocks = _parse_html(document.content)
    elif media_type == _DOCX_MIME:
        if not document.content.startswith(b"PK\x03\x04"):
            raise RagError(RagErrorCode.UNSUPPORTED_FORMAT)
        blocks = _parse_docx(document.content)
    elif media_type == _PDF_MIME:
        blocks = _parse_pdf(document.content, limits)
    else:
        raise RagError(RagErrorCode.UNSUPPORTED_FORMAT)
    # 3、统一规范化、总字符限制和空正文门禁。
    return _build_sections(document, blocks, limits)
