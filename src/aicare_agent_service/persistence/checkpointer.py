"""创建安全的 LangGraph Checkpoint 序列化器和进程级 Saver 资源。

开发/测试可显式使用内存Saver，生产必须使用加密的PostgreSQL AsyncPostgresSaver。
本文件只建立连接和序列化边界，不自动建表；数据库初始化由独立运维命令负责。
"""

# logging输出稳定运行告警，不应记录DSN、密钥或checkpoint正文。
import logging

# AsyncIterator用于asynccontextmanager的yield返回类型。
from collections.abc import AsyncIterator

# asynccontextmanager把异步生成器转换为可用``async with``管理的资源。
from contextlib import asynccontextmanager

# TypeAlias明确声明类型别名，而非普通运行时变量用途。
from typing import TypeAlias

# InMemorySaver仅供非生产开发；进程退出后状态丢失。
from langgraph.checkpoint.memory import InMemorySaver

# AsyncPostgresSaver是生产可持久化的异步PostgreSQL Checkpointer。
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# EncryptedSerializer在写入数据库前加密序列化内容。
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

# JsonPlusSerializer支持LangGraph消息及显式允许的自定义模型。
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from aicare_agent_service.config import CheckpointBackend, Environment, Settings
from aicare_agent_service.contracts.business_context import AgentBusinessContext
from aicare_agent_service.contracts.decisions import (
    AgentCode,
    Citation,
    EscalationSuggestion,
    HandoffSuggestion,
    Intent,
    MessageRole,
    RouteCode,
    RouteDecision,
    SafeConversationMessage,
    SafeToolResult,
    ToolResultStatus,
)
from aicare_agent_service.contracts.events import HandoffPriority
from aicare_agent_service.graph.state import AgentIdentity

# 当前模块logger名称使用__name__，便于按包路径过滤日志。
logger = logging.getLogger(__name__)

# Saver联合类型：调用者无需关心当前配置选择内存还是PostgreSQL实现。
CheckpointSaver: TypeAlias = InMemorySaver | AsyncPostgresSaver

# 反序列化白名单只含checkpoint状态确实需要的安全模型和枚举。
_CHECKPOINT_MODEL_ALLOWLIST = (
    AgentIdentity,
    AgentBusinessContext,
    SafeConversationMessage,
    RouteDecision,
    Citation,
    SafeToolResult,
    HandoffSuggestion,
    EscalationSuggestion,
    Intent,
    RouteCode,
    AgentCode,
    MessageRole,
    ToolResultStatus,
    HandoffPriority,
)


def _strict_json_serializer() -> JsonPlusSerializer:
    """只允许LangGraph内建安全类型和本项目明确列出的状态模型。"""
    # 禁止pickle回退，避免反序列化任意Python对象；只允许内建类型和明确白名单。
    return JsonPlusSerializer(
        pickle_fallback=False,
        allowed_msgpack_modules=_CHECKPOINT_MODEL_ALLOWLIST,
    )


def build_checkpoint_serializer(settings: Settings) -> EncryptedSerializer:
    """创建不依赖全局环境变量的 AES Checkpoint 序列化器。

    参数为已校验Settings；返回绑定严格JSON序列化器和AES密钥的EncryptedSerializer。
    密钥缺失或字节长度不符合AES要求时抛出ValueError。
    """
    # SecretStr避免普通打印泄露密钥，但这里需要显式取值传给加密器。
    secret = settings.checkpoint_encryption_key
    # 同时拒绝未配置和空字符串。
    if secret is None or not secret.get_secret_value():
        raise ValueError("PostgreSQL Checkpointer缺少LANGGRAPH_AES_KEY")

    # AES库需要字节密钥，因此按UTF-8编码配置字符串。
    key = secret.get_secret_value().encode("utf-8")
    # AES只接受128/192/256位密钥，分别等于16/24/32字节。
    if len(key) not in (16, 24, 32):
        raise ValueError("LANGGRAPH_AES_KEY必须为16、24或32字节")

    # 工厂方法创建使用PyCryptodome AES实现的加密序列化器。
    return EncryptedSerializer.from_pycryptodome_aes(
        serde=_strict_json_serializer(),
        key=key,
    )


@asynccontextmanager
async def checkpointer_resource(settings: Settings) -> AsyncIterator[CheckpointSaver]:
    """按配置持有进程级 Checkpointer 资源；数据库建表由独立初始化任务负责。

    ``async with checkpointer_resource(settings) as saver``进入资源；yield交付Saver；
    离开上下文后PostgreSQL连接自动关闭。内存后端在生产环境会被强制拒绝。
    """
    # 枚举使用is比较；显式MEMORY后端只允许非生产环境。
    if settings.checkpoint_backend is CheckpointBackend.MEMORY:
        # 防止生产重启后永久丢失会话状态。
        if settings.environment is Environment.PRODUCTION:
            raise ValueError("生产环境必须使用PostgreSQL Checkpointer")
        # 警告只说明后端类型，不输出状态正文。
        logger.warning("正在使用内存Checkpointer：状态非持久化，仅开发")
        # yield把带严格序列化器的Saver交给调用者，退出后继续执行下一行。
        yield InMemorySaver(serde=_strict_json_serializer())
        # 显式return阻止函数继续进入PostgreSQL分支。
        return

    # PostgreSQL DSN使用SecretStr保存；先验证非空。
    dsn_secret = settings.agent_postgres_dsn
    if dsn_secret is None or not dsn_secret.get_secret_value().strip():
        raise ValueError("PostgreSQL Checkpointer缺少AICARE_AGENT_POSTGRES_DSN")

    # PostgreSQL checkpoint正文必须经过AES加密和白名单序列化。
    serializer = build_checkpoint_serializer(settings)
    # 仅在创建连接时取出DSN明文，不记录到日志或错误消息。
    dsn = dsn_secret.get_secret_value()
    # Saver自身是异步上下文管理器，退出时自动关闭内部连接池/连接。
    async with AsyncPostgresSaver.from_conn_string(dsn, serde=serializer) as saver:
        # 不调用setup；生产建表必须通过显式init_db命令执行。
        yield saver
