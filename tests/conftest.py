import asyncio
import sys
from collections.abc import Callable, Iterator

import pytest

from aicare_agent_service.config import Environment, Settings, get_settings

LoopFactory = Callable[[], asyncio.AbstractEventLoop]


def build_pytest_loop_factories(platform: str) -> dict[str, LoopFactory]:
    """仅为Windows pytest-asyncio创建psycopg兼容的Selector事件循环。"""
    if platform == "win32":
        return {"selector": _new_windows_selector_loop}
    return {"platform-default": asyncio.new_event_loop}


def pytest_asyncio_loop_factories(
    config: pytest.Config,
    item: pytest.Item,
) -> dict[str, LoopFactory]:
    """pytest-asyncio hook：测试进程按平台创建loop，不改全局策略。"""
    del config, item
    return build_pytest_loop_factories(sys.platform)


def _new_windows_selector_loop() -> asyncio.AbstractEventLoop:
    """延迟解析Windows Selector实现，避免非Windows导入专用类型。"""
    from selectors import SelectSelector

    return asyncio.SelectorEventLoop(SelectSelector())


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()

    yield

    get_settings.cache_clear()


@pytest.fixture
def test_settings() -> Settings:
    return Settings(environment=Environment.TEST, _env_file=None)
