"""显式验证Python生产客户端可连接真实Java Gateway且安全映射授权门禁。"""

import os
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from aicare_agent_service.config import Settings
from aicare_agent_service.tools.contracts import (
    GetWalletArguments,
    SearchGamesArguments,
    ToolName,
)
from aicare_agent_service.tools.java_client import (
    JavaToolAccessDeniedError,
    JavaToolClient,
    build_java_http_client,
)

if TYPE_CHECKING:
    from aicare_agent_service.graph.state import AgentIdentity


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("AICARE_AGENT_LIVE_JAVA") != "true",
    reason="requires explicit AICARE_AGENT_LIVE_JAVA=true and a running local Java service",
)
async def test_real_java_gateway_authenticates_then_rejects_missing_active_run() -> None:
    settings = Settings(
        java_base_url="http://localhost:8080",
        java_allow_private_http=True,
    )
    identity = cast(
        Any,
        SimpleNamespace(
            tenant_id="tenant-live-missing",
            customer_id="customer-live-missing",
            conversation_id="conversation-live-missing",
            run_id="run-live-missing",
            trigger_message_id="message-live-missing",
            trigger_sequence=1,
        ),
    )
    http_client = build_java_http_client(settings)

    try:
        with pytest.raises(JavaToolAccessDeniedError):
            await JavaToolClient(http_client, settings).execute_tool(
                identity=cast("AgentIdentity", identity),
                tool_name=ToolName.GET_WALLET,
                tool_call_id="tool-call-live-missing",
                arguments=GetWalletArguments(),
                deadline=datetime.now(UTC) + timedelta(seconds=10),
            )
    finally:
        await http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("AICARE_AGENT_LIVE_JAVA_SUCCESS") != "true",
    reason="requires the explicit full-chain Java smoke fixture",
)
async def test_python_client_reads_real_java_business_data_for_active_run() -> None:
    settings = Settings(
        java_base_url="http://localhost:8080",
        java_allow_private_http=True,
    )
    identity = cast(
        Any,
        SimpleNamespace(
            tenant_id="tenant-demo",
            customer_id="user-customer-001",
            conversation_id="conv-tool-live-001",
            run_id="run-tool-live-001",
            trigger_message_id="msg-tool-live-001",
            trigger_sequence=1,
        ),
    )
    http_client = build_java_http_client(settings)

    try:
        result = await JavaToolClient(http_client, settings).execute_tool(
            identity=cast("AgentIdentity", identity),
            tool_name=ToolName.SEARCH_GAMES,
            tool_call_id="tool-call-live-success",
            arguments=SearchGamesArguments(query="游戏", limit=3),
            deadline=datetime.now(UTC) + timedelta(seconds=10),
        )
    finally:
        await http_client.aclose()

    assert result.status == "SUCCESS"
    assert result.tool_name is ToolName.SEARCH_GAMES
    assert result.data.kind == "GAME_PAGE"
    assert len(result.data.items) <= 3
