"""定义 RAG 全链路可跨边界传递的稳定错误码和用户安全错误载荷。"""

from enum import StrEnum


class RagErrorCode(StrEnum):
    """RAG 解析、模型、索引和检索阶段允许暴露的稳定错误码。"""

    UNSUPPORTED_FORMAT = "RAG_UNSUPPORTED_FORMAT"
    DOCUMENT_TOO_LARGE = "RAG_DOCUMENT_TOO_LARGE"
    NO_EXTRACTABLE_TEXT = "RAG_NO_EXTRACTABLE_TEXT"
    MODEL_UNAVAILABLE = "RAG_MODEL_UNAVAILABLE"
    INDEX_UNAVAILABLE = "RAG_INDEX_UNAVAILABLE"
    RETRIEVAL_TIMEOUT = "RAG_RETRIEVAL_TIMEOUT"
    INSUFFICIENT_EVIDENCE = "RAG_INSUFFICIENT_EVIDENCE"
    INDEX_VERSION_CONFLICT = "RAG_INDEX_VERSION_CONFLICT"


_SAFE_MESSAGES: dict[RagErrorCode, tuple[bool, str]] = {
    RagErrorCode.UNSUPPORTED_FORMAT: (False, "暂不支持该知识文档格式。"),
    RagErrorCode.DOCUMENT_TOO_LARGE: (False, "知识文档超过允许的大小限制。"),
    RagErrorCode.NO_EXTRACTABLE_TEXT: (False, "知识文档中没有可提取的正文。"),
    RagErrorCode.MODEL_UNAVAILABLE: (True, "知识模型暂时不可用，请稍后重试。"),
    RagErrorCode.INDEX_UNAVAILABLE: (True, "知识检索服务暂时不可用，请稍后重试。"),
    RagErrorCode.RETRIEVAL_TIMEOUT: (True, "知识检索超时，请稍后重试。"),
    RagErrorCode.INSUFFICIENT_EVIDENCE: (False, "当前知识库中没有足够依据回答该问题。"),
    RagErrorCode.INDEX_VERSION_CONFLICT: (True, "知识文档版本已更新，请重新同步。"),
}


class RagError(RuntimeError):
    """只携带稳定错误码，不保存原始文件、路径、凭据、正文或异常堆栈。"""

    def __init__(self, code: RagErrorCode) -> None:
        super().__init__(code.value)
        self.code = code

    def to_safe_payload(self) -> dict[str, str | bool]:
        """转换为可记录和传输的固定安全结构。"""
        # 1、从静态表取得是否可重试及用户文案，不拼接下游异常。
        retryable, message = _SAFE_MESSAGES[self.code]
        # 2、只返回三个白名单字段。
        return {"code": self.code.value, "retryable": retryable, "message": message}
