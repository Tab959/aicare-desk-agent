"""提供 checkpoint 保留清理的显式、安全 CLI，默认仅 dry-run。

命令读取配置并组装Redis活跃状态、PostgreSQL目录和清理服务；只有传入``--apply``才
实际删除。输出仅包含模式和计数，异常边界不打印DSN、凭据、thread ID或数据库堆栈。
"""

import argparse
import asyncio
import sys
from collections.abc import Coroutine, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from aicare_agent_service.config import CheckpointBackend, get_settings
from aicare_agent_service.persistence.checkpoint_cleanup import (
    CheckpointCleanupService,
    PostgresCheckpointCatalog,
)
from aicare_agent_service.persistence.checkpointer import checkpointer_resource
from aicare_agent_service.persistence.redis_run_store import RedisRunStore, build_redis_client


def build_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器；返回值可在main或测试中重复使用。"""
    # ArgumentParser自动生成--help，并保存命令用途描述。
    parser = argparse.ArgumentParser(description="清理超过保留期且无活跃run的checkpoint")
    # store_true表示出现--apply时为True，省略时默认False，即安全dry-run。
    parser.add_argument("--apply", action="store_true", help="实际删除；省略时只做dry-run")
    # 返回尚未解析参数的parser对象，便于测试检查CLI定义。
    return parser


async def run_cleanup(apply: bool) -> None:
    """按配置执行一次清理；apply=False只统计，True才删除。"""
    # 获取缓存且已校验的应用Settings。
    settings = get_settings()
    # 清理需要读取真实LangGraph表，内存后端无法执行。
    if settings.checkpoint_backend is not CheckpointBackend.POSTGRES:
        raise ValueError("checkpoint清理必须使用PostgreSQL")
    # Redis用于确认活跃run和持有cleanup guard，缺失时不能安全删除。
    if settings.agent_redis_url is None:
        raise ValueError("checkpoint清理缺少Redis")

    # 创建独立短生命周期Redis客户端，不复用FastAPI进程资源。
    redis_client = build_redis_client(
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
    # try/finally保证后续任一步骤异常都关闭Redis连接池。
    try:
        # RunStore在此只使用活跃检查和guard接口。
        run_store = RedisRunStore(
            redis_client,
            lease_seconds=settings.run_lease_seconds,
            retention_seconds=settings.run_retention_seconds,
            cleanup_guard_seconds=settings.checkpoint_cleanup_guard_seconds,
        )
        # 进入PostgreSQL Saver上下文，离开时自动关闭连接。
        async with checkpointer_resource(settings) as checkpointer:
            # 目录读取安全标识，服务执行跨Redis/PostgreSQL编排。
            service = CheckpointCleanupService(
                PostgresCheckpointCatalog(checkpointer),
                run_store,
                delete_timeout_seconds=settings.checkpoint_cleanup_guard_seconds / 2,
            )
            # 截止时间=当前UTC时间-配置保留秒数；dry_run与apply逻辑取反。
            result = await service.run(
                cutoff=datetime.now(UTC) - timedelta(seconds=settings.checkpoint_retention_seconds),
                limit=settings.checkpoint_cleanup_batch_size,
                dry_run=not apply,
            )
    finally:
        # aclose异步释放Redis客户端和连接池。
        await redis_client.aclose()

    # 使用全大写固定模式，便于运维日志搜索。
    mode = "APPLY" if apply else "DRY_RUN"
    # 只打印聚合计数，不输出候选conversation/thread标识。
    print(
        f"Checkpoint清理完成 mode={mode} scanned={result.scanned} "
        f"eligible={result.eligible} active={result.active} "
        f"deleted={result.deleted} failed={result.failed}"
    )


def _run(coroutine: Coroutine[Any, Any, None]) -> None:
    """在普通同步CLI中运行异步清理协程，并兼容Windows psycopg事件循环。"""
    # 非Windows使用默认asyncio.run。
    if sys.platform != "win32":
        asyncio.run(coroutine)
        return
    # 只在Windows分支导入Selector，避免其他平台加载无关实现。
    from selectors import SelectSelector

    # Runner负责事件循环完整生命周期；lambda延迟创建SelectorEventLoop。
    with asyncio.Runner(loop_factory=lambda: asyncio.SelectorEventLoop(SelectSelector())) as runner:
        # 同步等待异步清理完成。
        runner.run(coroutine)


def main(argv: Sequence[str] | None = None) -> int:
    """解析可选参数序列并返回稳定进程退出码；None表示读取真实命令行。"""
    # parse_args把--apply转换为args.apply布尔字段。
    args = build_parser().parse_args(argv)
    # CLI边界统一隐藏内部异常细节。
    try:
        _run(run_cleanup(args.apply))
    except Exception:  # noqa: BLE001 - 运维CLI不得暴露连接串、凭据或数据库异常。
        # stderr用于失败诊断；消息只给操作方向，不带原始异常。
        print(
            "Checkpoint清理失败：请检查Agent持久化配置和基础设施状态。",
            file=sys.stderr,
        )
        # 非零退出码让调度器识别任务失败。
        return 1
    # 清理协程正常完成时返回成功退出码。
    return 0


if __name__ == "__main__":
    # 只有``python -m ...cleanup_checkpoints``直接运行时才退出进程。
    raise SystemExit(main())
