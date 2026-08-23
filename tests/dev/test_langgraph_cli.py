"""验证 Windows 本地 Agent Server 启动包装器的事件循环与控制台边界。"""

from __future__ import annotations

import asyncio
import os
from selectors import SelectSelector
from typing import Any

import pytest

from aicare_agent_service.dev import langgraph_cli


def test_configure_runtime_uses_selector_policy_and_utf8_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, Any]] = []
    selector_policy = object()

    class ProbeStdout:
        def reconfigure(self, *, encoding: str) -> None:
            events.append(("encoding", encoding))

    monkeypatch.setattr(langgraph_cli.sys, "platform", "win32")
    monkeypatch.setattr(langgraph_cli.sys, "stdout", ProbeStdout())
    monkeypatch.setattr(
        langgraph_cli.asyncio,
        "WindowsSelectorEventLoopPolicy",
        lambda: selector_policy,
        raising=False,
    )
    monkeypatch.setattr(
        langgraph_cli.asyncio,
        "set_event_loop_policy",
        lambda policy: events.append(("policy", policy)),
    )
    monkeypatch.setattr(langgraph_cli, "install_langgraph_server_loop_factory", lambda: None)

    langgraph_cli.configure_cli_runtime()

    assert events == [("encoding", "utf-8"), ("policy", selector_policy)]


def test_configure_runtime_keeps_non_windows_event_loop_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(langgraph_cli.sys, "platform", "linux")
    monkeypatch.setattr(
        langgraph_cli.asyncio,
        "set_event_loop_policy",
        lambda _policy: pytest.fail("非 Windows 不得修改事件循环策略"),
    )

    langgraph_cli.configure_cli_runtime()


def test_windows_selector_loop_factory_returns_psycopg_compatible_loop() -> None:
    loop = langgraph_cli.windows_selector_loop_factory()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
        assert isinstance(loop._selector, SelectSelector)
    finally:
        loop.close()


def test_install_server_loop_factory_injects_uvicorn_loop_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def original_run_server(*args: Any, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "started"

    monkeypatch.setattr(langgraph_cli.langgraph_server_cli, "run_server", original_run_server)

    langgraph_cli.install_langgraph_server_loop_factory()
    result = langgraph_cli.langgraph_server_cli.run_server("127.0.0.1", 2025)

    assert result == "started"
    assert captured["loop"] == (
        "aicare_agent_service.dev.langgraph_cli:windows_selector_loop_factory"
    )


def test_main_configures_runtime_before_delegating_to_official_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(langgraph_cli, "requires_utf8_restart", lambda: False)
    monkeypatch.setattr(langgraph_cli, "configure_cli_runtime", lambda: events.append("configure"))
    monkeypatch.setattr(langgraph_cli, "official_cli", lambda: events.append("cli") or 0)

    result = langgraph_cli.main()

    assert result == 0
    assert events == ["configure", "cli"]


def test_requires_utf8_restart_only_for_non_utf8_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(langgraph_cli.sys, "platform", "win32")
    monkeypatch.setattr(langgraph_cli, "current_utf8_mode", lambda: 0)
    assert langgraph_cli.requires_utf8_restart() is True

    monkeypatch.setattr(langgraph_cli, "current_utf8_mode", lambda: 1)
    assert langgraph_cli.requires_utf8_restart() is False

    monkeypatch.setattr(langgraph_cli.sys, "platform", "linux")
    monkeypatch.setattr(langgraph_cli, "current_utf8_mode", lambda: 0)
    assert langgraph_cli.requires_utf8_restart() is False


def test_run_with_utf8_starts_same_module_without_leaking_other_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(langgraph_cli.sys, "executable", "python.exe")
    monkeypatch.setattr(langgraph_cli.sys, "argv", ["aicare-langgraph.exe", "dev", "--help"])
    monkeypatch.setenv("EXISTING_SETTING", "kept")

    def probe_call(argv: list[str], *, env: dict[str, str]) -> int:
        captured.update(argv=argv, env=env)
        return 23

    monkeypatch.setattr(langgraph_cli.subprocess, "call", probe_call)

    result = langgraph_cli.run_with_utf8()

    assert result == 23
    assert captured["argv"] == [
        "python.exe",
        "-m",
        "aicare_agent_service.dev.langgraph_cli",
        "dev",
        "--help",
    ]
    assert captured["env"]["PYTHONUTF8"] == "1"
    assert captured["env"]["EXISTING_SETTING"] == "kept"
    assert os.environ.get("PYTHONUTF8") != "1"


def test_run_with_utf8_converts_console_interrupt_to_standard_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        langgraph_cli.subprocess,
        "call",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    assert langgraph_cli.run_with_utf8() == 130
