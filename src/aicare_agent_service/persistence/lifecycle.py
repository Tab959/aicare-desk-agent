"""管理 FastAPI 进程级持久化资源的创建、暴露与逆序关闭。

应用启动时先进入Checkpointer资源，再可选创建Redis连接、执行就绪审计并组装RunStore和
AgentRunLifecycle；应用退出时AsyncExitStack按注册相反顺序先关Redis、再关Checkpointer。
这里不执行数据库迁移，也不把Redis run ledger当作LangGraph Store。
"""

# AsyncIterator描述lifespan只yield一次控制权的异步生成器。
from collections.abc import AsyncIterator

# AsyncExitStack统一管理动态数量的异步资源；asynccontextmanager生成FastAPI lifespan上下文。
from contextlib import AsyncExitStack, asynccontextmanager

# cast只帮助静态类型检查器理解app.state动态属性，运行时不会转换对象。
from typing import cast

from fastapi import FastAPI
from redis.asyncio import Redis

from aicare_agent_service.config import Settings
from aicare_agent_service.persistence.checkpointer import checkpointer_resource
from aicare_agent_service.persistence.lifecycle_runner import AgentRunLifecycle
from aicare_agent_service.persistence.redis_readiness import enforce_redis_readiness
from aicare_agent_service.persistence.redis_run_store import RedisRunStore, build_redis_client


@asynccontextmanager
async def persistence_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """持有 Checkpointer 及可选 Redis RunStore，退出时按相反顺序关闭。

    参数是FastAPI应用；函数yield期间资源可通过``app.state``访问；正常或异常退出都会
    关闭已注册资源。没有Redis配置时仍可启动仅含Checkpointer的开发应用。
    """
    # FastAPI的state允许动态字段；cast声明此前app工厂已经放入Settings。
    settings = cast(Settings, app.state.settings)
    # AsyncExitStack退出时自动调用已进入上下文和回调的清理逻辑。
    async with AsyncExitStack() as stack:
        # 进入Checkpointer上下文并把Saver暴露给后续图编译或请求处理。
        app.state.checkpointer = await stack.enter_async_context(checkpointer_resource(settings))
        # 先初始化可选资源为None，使没有Redis时状态字段仍稳定存在。
        app.state.redis_client = None
        app.state.run_store = None
        app.state.run_lifecycle = None

        # 只有显式配置Agent Redis URL才启用生产式run协调能力。
        if settings.agent_redis_url is not None:
            # 创建带连接数、超时和健康检查边界的异步客户端。
            client = build_redis_client(
                settings.agent_redis_url,
                redis_mode=settings.redis_mode,
                sentinel_endpoints=settings.redis_sentinel_endpoints,
                sentinel_master_name=settings.redis_master_name,
                sentinel_username=settings.redis_sentinel_username,
                sentinel_password=settings.redis_sentinel_password,
                max_connections=settings.redis_max_connections,
                socket_timeout_seconds=settings.redis_socket_timeout_seconds,
                health_check_interval_seconds=settings.redis_health_check_interval_seconds,
            )
            # 暴露客户端便于应用内部健康检查，但浏览器和模型不能直接访问。
            app.state.redis_client = client
            # 回调后注册先执行，因此Redis会在较早进入的Checkpointer之前关闭。
            stack.push_async_callback(_close_redis, client)
            # 生产不合规时阻断启动；非生产只记录稳定代码告警。
            await enforce_redis_readiness(client, settings.environment)
            # RunStore维护幂等、租约、终态和清理guard，不保存checkpoint正文。
            app.state.run_store = RedisRunStore(
                client,
                lease_seconds=settings.run_lease_seconds,
                retention_seconds=settings.run_retention_seconds,
                cleanup_guard_seconds=settings.checkpoint_cleanup_guard_seconds,
            )
            # 生命周期编排器复用进程级RunStore和配置的心跳/总超时。
            app.state.run_lifecycle = AgentRunLifecycle(
                app.state.run_store,
                heartbeat_seconds=settings.run_heartbeat_seconds,
                timeout_seconds=settings.run_timeout_seconds,
                input_max_chars=settings.input_max_chars,
            )
        # 把控制权交还FastAPI；yield之后直到应用关闭才继续退出资源栈。
        yield


async def _close_redis(client: Redis) -> None:
    """关闭Redis客户端及连接池；参数是lifespan创建的进程级客户端。"""
    # aclose是异步关闭接口，await确保连接释放完成后再继续关闭其他资源。
    await client.aclose()
