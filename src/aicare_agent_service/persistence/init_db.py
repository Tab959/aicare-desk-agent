"""提供 LangGraph PostgreSQL Checkpointer 表结构的显式初始化 CLI。

该模块只应由运维命令运行，不在FastAPI启动时自动迁移。它复用正式加密Saver配置，
调用官方``setup()``创建或升级表，并在CLI边界隐藏DSN、凭据和数据库异常细节。
"""

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any

from aicare_agent_service.config import CheckpointBackend, Settings, get_settings
from aicare_agent_service.persistence.checkpointer import checkpointer_resource


async def initialize_database(settings: Settings) -> None:
    """仅在显式运维命令中创建或升级LangGraph Checkpointer表结构。"""
    # 初始化命令禁止在内存后端执行，因为内存Saver没有持久化表结构。
    if settings.checkpoint_backend is not CheckpointBackend.POSTGRES:
        raise ValueError("初始化数据库必须使用PostgreSQL Checkpointer")

    # async with确保setup完成或失败后都关闭PostgreSQL资源。
    async with checkpointer_resource(settings) as checkpointer:
        # setup是LangGraph官方Saver提供的幂等表结构初始化方法。
        await checkpointer.setup()


def _run_initialization(coroutine: Coroutine[Any, Any, None]) -> None:
    """在Windows显式CLI中使用psycopg兼容的Selector事件循环。"""
    # Linux/macOS使用Python默认事件循环执行协程直到完成。
    if sys.platform != "win32":
        asyncio.run(coroutine)
        # return防止非Windows继续导入和使用平台专用Selector配置。
        return

    # 仅在Windows分支解析Selector，避免非Windows导入平台专用实现。
    from selectors import SelectSelector

    # Runner负责创建、运行并关闭指定事件循环；lambda延迟构造SelectorEventLoop。
    with asyncio.Runner(loop_factory=lambda: asyncio.SelectorEventLoop(SelectSelector())) as runner:
        # run同步等待传入协程完成，便于普通CLI main调用异步代码。
        runner.run(coroutine)


def main() -> int:
    """输出不含连接信息和异常栈的稳定运维诊断。"""
    # CLI边界捕获异常并只输出稳定诊断，退出码0成功、1失败。
    try:
        _run_initialization(initialize_database(get_settings()))
    except ValueError:
        # 已知配置错误不打印具体DSN或密钥内容。
        print("PostgreSQL Checkpointer初始化失败：配置无效", file=sys.stderr)
        return 1
    except Exception:  # noqa: BLE001 - CLI边界必须隐藏所有基础设施细节。
        # 未知基础设施异常同样不输出异常正文和堆栈。
        print("PostgreSQL Checkpointer初始化失败：无法完成初始化", file=sys.stderr)
        return 1
    # 只有setup正常完成才输出成功信息。
    print("PostgreSQL Checkpointer初始化完成")
    return 0


if __name__ == "__main__":
    # 直接执行模块时把main返回值转换为进程退出码；被import时不会运行。
    raise SystemExit(main())
