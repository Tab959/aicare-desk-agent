"""向LangGraph Agent Server提供应用统一的加密PostgreSQL Checkpointer资源。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from aicare_agent_service.config import CheckpointBackend, get_settings
from aicare_agent_service.persistence.checkpointer import (
    CheckpointSaver,
    checkpointer_resource,
)


@asynccontextmanager
async def generate_server_checkpointer() -> AsyncIterator[CheckpointSaver]:
    """让Agent Server持有并注入与FastAPI相同的生产Checkpointer。"""
    # 1、Server入口禁止内存后端，避免调试线程与正式恢复语义不一致。
    settings = get_settings()
    if settings.checkpoint_backend is not CheckpointBackend.POSTGRES:
        raise ValueError("Agent Server必须使用PostgreSQL Checkpointer")
    # 2、复用AES序列化白名单和连接生命周期，不在Server入口重复实现连接逻辑。
    async with checkpointer_resource(settings) as saver:
        yield saver
