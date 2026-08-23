import asyncio
from selectors import SelectSelector

import conftest
import pytest


def test_windows_pytest_loop_factory_creates_selector_event_loops() -> None:
    factory = conftest.build_pytest_loop_factories("win32")["selector"]
    loop = factory()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
        assert isinstance(loop._selector, SelectSelector)
    finally:
        loop.close()


def test_non_windows_pytest_loop_factory_keeps_platform_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = lambda: asyncio.new_event_loop()
    monkeypatch.setattr(conftest.asyncio, "new_event_loop", factory)

    assert conftest.build_pytest_loop_factories("linux") == {"platform-default": factory}


@pytest.mark.asyncio
async def test_pytest_asyncio_uses_selector_loop_on_windows() -> None:
    loop = asyncio.get_running_loop()

    assert isinstance(loop, asyncio.SelectorEventLoop)
