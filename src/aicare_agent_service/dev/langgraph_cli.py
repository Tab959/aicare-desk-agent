"""为 Windows 本地调试包装官方 LangGraph CLI，生产部署不经过该入口。"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from selectors import SelectSelector
from typing import Any

from langgraph_api import cli as langgraph_server_cli
from langgraph_cli.cli import cli as official_cli

_WINDOWS_LOOP_FACTORY_PATH = "aicare_agent_service.dev.langgraph_cli:windows_selector_loop_factory"


def current_utf8_mode() -> int:
    """读取解释器启动时确定的 UTF-8 模式标志。"""
    # 1、sys.flags 是只读启动快照；只有新进程才能真正改变 utf8_mode。
    return sys.flags.utf8_mode


def requires_utf8_restart() -> bool:
    """判断 Windows 当前解释器是否必须以 UTF-8 模式重启。"""
    # 1、Linux 生产容器默认 UTF-8，不增加额外进程跳转。
    # 2、Windows 非 UTF-8 进程无法可靠读取含中文注释的 .env。
    return sys.platform == "win32" and current_utf8_mode() == 0


def run_with_utf8() -> int:
    """以相同解释器和参数运行 UTF-8 子进程，并同步返回其退出码。"""
    # 1、复制环境而不修改父进程，只覆盖解释器启动所需的 PYTHONUTF8。
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    # 2、通过可导入模块恢复包装器，保留用户传给官方 CLI 的全部参数。
    arguments = [
        sys.executable,
        "-m",
        "aicare_agent_service.dev.langgraph_cli",
        *sys.argv[1:],
    ]
    # 3、同步等待子进程，使服务日志、Ctrl+C 和最终退出码仍归属当前控制台。
    try:
        return subprocess.call(arguments, env=environment)
    except KeyboardInterrupt:
        # 4、Ctrl+C 已同时通知子服务；父包装器使用通用 130 退出码且不打印堆栈。
        return 130


def windows_selector_loop_factory() -> asyncio.AbstractEventLoop:
    """为 Uvicorn 创建可被 psycopg 异步驱动使用的 Selector loop。"""
    # 1、自定义导入路径会被 Uvicorn 直接作为 asyncio.Runner.loop_factory 调用。
    # 2、显式绑定 SelectSelector，避免 Windows 默认回到 ProactorEventLoop。
    return asyncio.SelectorEventLoop(SelectSelector())


def install_langgraph_server_loop_factory() -> None:
    """让官方 LangGraph Server 将应用 Selector 工厂传给 Uvicorn。"""
    # 1、保存官方函数；包装器只补充 CLI 当前未暴露的 Uvicorn loop 参数。
    original_run_server = langgraph_server_cli.run_server

    def run_server_with_selector(*args: Any, **kwargs: Any) -> Any:
        """保留官方所有参数，并强制使用可跨重载进程导入的 loop 工厂。"""
        # 1、使用模块导入路径而不是临时 lambda，确保 Windows spawn 子进程可恢复。
        kwargs["loop"] = _WINDOWS_LOOP_FACTORY_PATH
        # 2、其余启动、配置和异常行为继续由官方实现负责。
        return original_run_server(*args, **kwargs)

    # 2、官方 dev 命令稍后从该模块导入 run_server，因此此处必须先完成替换。
    langgraph_server_cli.run_server = run_server_with_selector


def configure_cli_runtime() -> None:
    """在官方 CLI 创建事件循环前应用 Windows psycopg 兼容配置。"""
    # 1、Linux 生产容器使用平台默认事件循环，不做任何全局修改。
    if sys.platform != "win32":
        return
    # 2、Windows 控制台改用 UTF-8，避免官方帮助文本中的 emoji 触发 GBK 编码异常。
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8")
    # 3、在 Uvicorn 创建 loop 前切换 Selector；psycopg 异步连接不支持 Proactor。
    policy_factory = asyncio.WindowsSelectorEventLoopPolicy  # type: ignore[attr-defined]
    asyncio.set_event_loop_policy(policy_factory())
    # 4、Uvicorn 0.52 使用 loop_factory 而非全局 policy，必须同时注入显式工厂。
    install_langgraph_server_loop_factory()


def main() -> Any:
    """完成本机运行时配置后，把全部命令行参数交给官方 LangGraph CLI。"""
    # 1、Windows 编码模式只能在解释器启动时设置，因此必要时先原参数重启。
    if requires_utf8_restart():
        return run_with_utf8()
    # 2、设置事件循环策略，确保后续 CLI/Uvicorn 创建的是 Selector loop。
    configure_cli_runtime()
    # 3、官方 Click 命令继续负责参数解析、服务启动和退出码。
    return official_cli()


if __name__ == "__main__":
    main()
