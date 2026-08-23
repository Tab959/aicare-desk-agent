"""组装FastAPI进程级持久化、RAG模型/索引与Java工具客户端生命周期。"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import cast

from fastapi import FastAPI

from aicare_agent_service.config import Settings
from aicare_agent_service.persistence.lifecycle import persistence_lifespan
from aicare_agent_service.rag.model_runtime import rag_lifespan
from aicare_agent_service.tools.java_client import (
    JavaToolClient,
    build_java_http_client,
    log_private_http_warning,
)


@asynccontextmanager
async def application_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """按持久化→RAG→Java启动，并按Java→RAG→持久化关闭。"""
    # 1、先进入Checkpointer和Redis生命周期，确保图运行基础资源已就绪。
    async with persistence_lifespan(app), rag_lifespan(app), AsyncExitStack() as stack:
        settings = cast(Settings, app.state.settings)
        app.state.java_http_client = None
        app.state.java_client = None

        # 2、开发测试可同时省略Java配置；半配置始终阻断，生产缺失已在创建应用时阻断。
        has_url = settings.java_base_url is not None
        has_token = settings.java_service_token is not None
        if has_url != has_token:
            raise ValueError("Java工具客户端配置不完整")
        if has_url:
            http_client = build_java_http_client(settings)
            app.state.java_http_client = http_client
            stack.push_async_callback(http_client.aclose)
            app.state.java_client = JavaToolClient(http_client, settings)
            log_private_http_warning(settings)

        # 3、yield期间路由和图运行复用唯一客户端，异常退出仍由资源栈清理。
        yield
