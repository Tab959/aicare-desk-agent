"""验证Java工具客户端的身份封装、有限重试、协议门禁和安全错误映射。"""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from fastapi import FastAPI, Request, Response

from aicare_agent_service.config import Environment, Settings
from aicare_agent_service.tools import java_client as java_client_module
from aicare_agent_service.tools.contracts import (
    GetWalletArguments,
    SearchGamesArguments,
    ToolName,
)
from aicare_agent_service.tools.java_client import (
    JavaToolAccessDeniedError,
    JavaToolClient,
    JavaToolNotFoundError,
    JavaToolProtocolError,
    JavaToolUnavailableError,
    build_java_http_client,
    log_private_http_warning,
)

if TYPE_CHECKING:
    from aicare_agent_service.graph.state import AgentIdentity


def _settings(**updates: object) -> Settings:
    """构建启用HTTPS Java Gateway的测试配置。"""
    values: dict[str, object] = {
        "environment": Environment.TEST,
        "java_base_url": "https://java.internal.example",
        "java_service_token": "service-secret",
        "java_retry_after_max_seconds": 0.01,
        "_env_file": None,
    }
    values.update(updates)
    return Settings(**values)


def _identity() -> "AgentIdentity":
    """返回完整Java拥有身份。"""
    return cast(
        Any,
        SimpleNamespace(
            tenant_id="tenant-1",
            customer_id="customer-1",
            conversation_id="conversation-1",
            run_id="run-1",
            trigger_message_id="message-1",
            trigger_sequence=1,
        ),
    )


def _deadline(seconds: float = 5) -> datetime:
    """返回未来绝对截止时间。"""
    return datetime.now(UTC) + timedelta(seconds=seconds)


def _success_payload(request_id: str, tool_call_id: str = "call-1") -> dict[str, object]:
    """构建钱包工具的冻结成功响应。"""
    return {
        "contractVersion": "1.0",
        "requestId": request_id,
        "toolCallId": tool_call_id,
        "toolName": "get_wallet",
        "status": "SUCCESS",
        "observedAt": "2026-08-16T00:00:00Z",
        "data": {"kind": "WALLET", "availableBalance": "100.00", "currency": "CNY"},
    }


def test_http_client_factory_disables_environment_proxy_and_redirects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    sentinel = cast(Any, object())

    def fake_async_client(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(java_client_module.httpx, "AsyncClient", fake_async_client)

    assert build_java_http_client(_settings()) is sentinel
    assert captured["trust_env"] is False
    assert captured["follow_redirects"] is False
    assert captured["base_url"] == "https://java.internal.example/"
    assert (
        cast(dict[str, str], captured["headers"])["X-AICare-Agent-Service-Token"]
        == "service-secret"
    )


def test_private_http_warning_contains_only_stable_code(caplog: pytest.LogCaptureFixture) -> None:
    settings = _settings(
        java_base_url="http://192.168.1.10:8080",
        java_allow_private_http=True,
    )

    log_private_http_warning(settings)

    rendered = " ".join(record.getMessage() for record in caplog.records)
    assert rendered == "JAVA_TOOL_PRIVATE_HTTP_ENABLED"
    assert "192.168.1.10" not in rendered
    assert "service-secret" not in rendered


@pytest.mark.asyncio
async def test_client_revalidates_arguments_against_selected_tool_before_network() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("参数与工具不匹配时不得发送网络请求")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://java.internal.example"
    ) as http_client:
        with pytest.raises(JavaToolProtocolError):
            await JavaToolClient(http_client, _settings()).execute_tool(
                identity=_identity(),
                tool_name=ToolName.GET_WALLET,
                tool_call_id="call-1",
                arguments=SearchGamesArguments(query="game"),
                deadline=_deadline(),
            )


@pytest.mark.asyncio
async def test_invalid_correlation_id_is_safely_rejected_before_network() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("非法关联ID不得发送网络请求")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://java.internal.example"
    ) as http_client:
        with pytest.raises(JavaToolProtocolError):
            await JavaToolClient(http_client, _settings()).execute_tool(
                identity=_identity(),
                tool_name=ToolName.GET_WALLET,
                tool_call_id="",
                arguments=GetWalletArguments(),
                deadline=_deadline(),
            )


@pytest.mark.asyncio
async def test_execute_tool_sends_fixed_path_full_identity_and_correlation_headers() -> None:
    captured: dict[str, object] = {}
    app = FastAPI()

    @app.post("/api/internal/v1/agent/tools/get_wallet/invoke")
    async def invoke(request: Request) -> dict[str, object]:
        captured["path"] = request.url.path
        captured["token"] = request.headers["X-AICare-Agent-Service-Token"]
        captured["body"] = await request.json()
        request_id = request.headers["X-Request-Id"]
        return _success_payload(request_id)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="https://java.internal.example",
        headers={"X-AICare-Agent-Service-Token": "service-secret"},
    ) as http_client:
        result = await JavaToolClient(http_client, _settings()).execute_tool(
            identity=_identity(),
            tool_name=ToolName.GET_WALLET,
            tool_call_id="call-1",
            arguments=GetWalletArguments(),
            deadline=_deadline(),
        )

    assert result.data.kind == "WALLET"
    assert captured["path"] == "/api/internal/v1/agent/tools/get_wallet/invoke"
    assert captured["token"] == "service-secret"
    assert captured["body"] == {
        "contractVersion": "1.0",
        "toolCallId": "call-1",
        "identity": {
            "tenantId": "tenant-1",
            "customerId": "customer-1",
            "conversationId": "conversation-1",
            "runId": "run-1",
            "triggerMessageId": "message-1",
            "triggerSequence": 1,
        },
        "arguments": {},
    }


@pytest.mark.asyncio
async def test_retryable_status_retries_once_and_caps_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    delays: list[float] = []
    app = FastAPI()

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    @app.post("/api/internal/v1/agent/tools/get_wallet/invoke")
    async def invoke(request: Request) -> Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return Response(status_code=503, headers={"Retry-After": "60"})
        return Response(
            json.dumps(_success_payload(request.headers["X-Request-Id"])),
            media_type="application/json",
        )

    monkeypatch.setattr("aicare_agent_service.tools.java_client.asyncio.sleep", fake_sleep)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="https://java.internal.example",
    ) as http_client:
        await JavaToolClient(http_client, _settings()).execute_tool(
            identity=_identity(),
            tool_name=ToolName.GET_WALLET,
            tool_call_id="call-1",
            arguments=GetWalletArguments(),
            deadline=_deadline(),
        )

    assert attempts == 2
    assert delays == [0.01]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403, 409])
async def test_access_denied_statuses_are_safe_and_never_retried(status_code: int) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(status_code, text="secret raw response", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://java.internal.example"
    ) as http_client:
        with pytest.raises(JavaToolAccessDeniedError) as exc_info:
            await JavaToolClient(http_client, _settings()).execute_tool(
                identity=_identity(),
                tool_name=ToolName.GET_WALLET,
                tool_call_id="call-1",
                arguments=GetWalletArguments(),
                deadline=_deadline(),
            )

    assert attempts == 1
    assert "secret raw response" not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("exception_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_retryable_network_failure_retries_once_then_returns_safe_unavailable(
    exception_type: type[httpx.RequestError],
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise exception_type("socket secret", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://java.internal.example"
    ) as http_client:
        with pytest.raises(JavaToolUnavailableError) as exc_info:
            await JavaToolClient(http_client, _settings()).execute_tool(
                identity=_identity(),
                tool_name=ToolName.GET_WALLET,
                tool_call_id="call-1",
                arguments=GetWalletArguments(),
                deadline=_deadline(),
            )

    assert attempts == 2
    assert "socket secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_bad_request_is_protocol_error_and_is_not_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, text="validation detail secret", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://java.internal.example"
    ) as http_client:
        with pytest.raises(JavaToolProtocolError) as exc_info:
            await JavaToolClient(http_client, _settings()).execute_tool(
                identity=_identity(),
                tool_name=ToolName.GET_WALLET,
                tool_call_id="call-1",
                arguments=GetWalletArguments(),
                deadline=_deadline(),
            )

    assert attempts == 1
    assert "validation detail secret" not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("exception_type", [httpx.WriteTimeout, httpx.PoolTimeout])
async def test_non_retryable_transport_timeout_is_safely_mapped_once(
    exception_type: type[httpx.TimeoutException],
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise exception_type("transport secret", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://java.internal.example"
    ) as http_client:
        with pytest.raises(JavaToolUnavailableError) as exc_info:
            await JavaToolClient(http_client, _settings()).execute_tool(
                identity=_identity(),
                tool_name=ToolName.GET_WALLET,
                tool_call_id="call-1",
                arguments=GetWalletArguments(),
                deadline=_deadline(),
            )

    assert attempts == 1
    assert "transport secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_not_found_is_indistinguishable_and_never_reads_raw_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="resource and tenant secret", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://java.internal.example"
    ) as http_client:
        with pytest.raises(JavaToolNotFoundError) as exc_info:
            await JavaToolClient(http_client, _settings()).execute_tool(
                identity=_identity(),
                tool_name=ToolName.GET_WALLET,
                tool_call_id="call-1",
                arguments=GetWalletArguments(),
                deadline=_deadline(),
            )

    assert "resource and tenant secret" not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_attempts"),
    [(429, 2), (502, 2), (503, 2), (504, 2), (500, 1)],
)
async def test_unavailable_status_retry_policy_is_exact(
    status_code: int,
    expected_attempts: int,
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(status_code, request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://java.internal.example"
    ) as http_client:
        with pytest.raises(JavaToolUnavailableError):
            await JavaToolClient(http_client, _settings()).execute_tool(
                identity=_identity(),
                tool_name=ToolName.GET_WALLET,
                tool_call_id="call-1",
                arguments=GetWalletArguments(),
                deadline=_deadline(),
            )

    assert attempts == expected_attempts


@pytest.mark.asyncio
async def test_success_response_rejects_correlation_drift() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_success_payload("wrong-request-id"),
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://java.internal.example"
    ) as http_client:
        with pytest.raises(JavaToolProtocolError):
            await JavaToolClient(http_client, _settings()).execute_tool(
                identity=_identity(),
                tool_name=ToolName.GET_WALLET,
                tool_call_id="call-1",
                arguments=GetWalletArguments(),
                deadline=_deadline(),
            )


@pytest.mark.asyncio
async def test_success_response_rejects_oversized_body_before_schema_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"x" * 2049,
            headers={"Content-Type": "application/json"},
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://java.internal.example"
    ) as http_client:
        with pytest.raises(JavaToolProtocolError):
            await JavaToolClient(http_client, _settings(java_response_max_bytes=2048)).execute_tool(
                identity=_identity(),
                tool_name=ToolName.GET_WALLET,
                tool_call_id="call-1",
                arguments=GetWalletArguments(),
                deadline=_deadline(),
            )


@pytest.mark.asyncio
async def test_absolute_deadline_also_bounds_streaming_response_body() -> None:
    class SlowBody(httpx.AsyncByteStream):
        def __init__(self, content: bytes) -> None:
            self._content = content

        async def __aiter__(self):
            await asyncio.sleep(0.05)
            yield self._content

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            stream=SlowBody(json.dumps(_success_payload(request.headers["X-Request-Id"])).encode()),
            headers={"Content-Type": "application/json"},
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://java.internal.example"
    ) as http_client:
        with pytest.raises(JavaToolUnavailableError):
            await JavaToolClient(http_client, _settings()).execute_tool(
                identity=_identity(),
                tool_name=ToolName.GET_WALLET,
                tool_call_id="call-1",
                arguments=GetWalletArguments(),
                deadline=_deadline(0.01),
            )
