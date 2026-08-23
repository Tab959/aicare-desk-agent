"""建立 Java run 身份与 LangGraph 持久化身份之间的稳定映射。

本文件只做两件事：把 Java 生成的 ``conversationId`` 原样用作 LangGraph
``thread_id``，以及对完整请求生成不保存正文的 SHA-256 幂等摘要。它不生成任何业务 ID。
"""

# hashlib 提供 SHA-256 等密码学哈希算法；摘要只能校验一致性，不能还原原文。
import hashlib

# json 用于把结构化请求转换成键顺序稳定的规范字符串。
import json

# RunnableConfig 是 LangGraph/LangChain 调用时传入 thread_id 等运行配置的字典类型。
from langchain_core.runnables import RunnableConfig

# AgentRunRequest 是 Java 发起一次 Agent 生成任务的已校验契约。
from aicare_agent_service.contracts.agent_run import AgentRunRequest


def build_thread_config(request: AgentRunRequest) -> RunnableConfig:
    """将 Java 会话标识原样映射为唯一 LangGraph ``thread_id``。

    参数 ``request`` 由 Java 提供；返回 LangGraph 标准配置字典。同一会话始终使用同一个
    conversationId，不能用每轮变化的 runId，否则无法恢复同一会话的 checkpoint。
    """
    # ``configurable`` 是 LangGraph 约定的配置命名空间；str 保证值为普通字符串。
    return {"configurable": {"thread_id": str(request.conversation_id)}}


def canonical_request_digest(contract_version: str, request: AgentRunRequest) -> str:
    """计算不保存请求正文的稳定幂等摘要。

    参数包含共享契约版本和完整请求；返回64位小写十六进制SHA-256。相同规范输入得到
    相同摘要，任何字段变化都会形成不同摘要，Redis只需保存摘要即可检测runId冲突。
    """
    # strip 删除版本字符串首尾空白，防止视觉相同的版本产生不同摘要。
    normalized_version = contract_version.strip()
    # 空版本无法区分不同契约的序列化语义，因此立即拒绝。
    if not normalized_version:
        raise ValueError("契约版本不能为空")

    # 组装规范载荷；by_alias=True 使用Java camelCase字段名，mode="json"转换为JSON安全值。
    canonical_payload = {
        "contractVersion": normalized_version,
        "request": request.model_dump(by_alias=True, mode="json"),
    }
    # separators 去掉无意义空格，sort_keys 固定字典键顺序，确保跨调用摘要稳定。
    canonical_json = json.dumps(
        canonical_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    # 哈希函数接收字节，因此先按UTF-8编码；hexdigest返回便于Redis保存的小写十六进制串。
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
