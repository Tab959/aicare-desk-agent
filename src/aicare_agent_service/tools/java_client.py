"""提供调用 Java 内部只读工具 API 的生产级异步客户端。

客户端只接受固定工具枚举和严格参数，统一执行身份封装、绝对截止时间、一次有限重试、
正文大小限制及安全错误映射；原始响应、凭据和底层异常不会进入模型消息。
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx
from pydantic import ValidationError

from aicare_agent_service.config import Settings, validate_java_transport
from aicare_agent_service.tools.contracts import (
    TOOL_CONTRACT_VERSION,
    AgentToolIdentity,
    AgentToolInvokeRequest,
    AgentToolInvokeResponse,
    ToolArguments,
    ToolInvocationResult,
    ToolName,
    validate_tool_arguments,
)

if TYPE_CHECKING:
    from aicare_agent_service.graph.state import AgentIdentity

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})
_ACCESS_DENIED_STATUS_CODES = frozenset({401, 403, 409})


class JavaToolClientError(RuntimeError):
    """Java工具调用的安全基类；仅保存稳定代码和固定消息。"""

    code = "TOOL_ERROR"
    safe_message = "工具调用失败，请稍后重试。"

    def __init__(self) -> None:
        super().__init__(self.safe_message)


class JavaToolAccessDeniedError(JavaToolClientError):
    """当前活动run、身份或权限校验未通过。"""

    code = "TOOL_ACCESS_DENIED"
    safe_message = "当前请求无法访问该业务信息。"


class JavaToolNotFoundError(JavaToolClientError):
    """资源不存在或对当前顾客不可见。"""

    code = "TOOL_NOT_FOUND"
    safe_message = "未找到可访问的业务信息。"


class JavaToolUnavailableError(JavaToolClientError):
    """Java服务、连接或限流暂时不可用。"""

    code = "TOOL_UNAVAILABLE"
    safe_message = "业务查询暂时不可用，请稍后重试。"


class JavaToolProtocolError(JavaToolClientError):
    """Java响应大小、格式、版本或关联字段不符合冻结契约。"""

    code = "TOOL_PROTOCOL_ERROR"
    safe_message = "业务查询响应异常，请稍后重试。"


class JavaToolClient:
    """复用进程级AsyncClient调用固定Java工具端点。"""

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._max_response_bytes = settings.java_response_max_bytes
        self._retry_after_max_seconds = settings.java_retry_after_max_seconds
        self._timeouts = (
            settings.java_connect_timeout_seconds,
            settings.java_read_timeout_seconds,
            settings.java_write_timeout_seconds,
            settings.java_pool_timeout_seconds,
        )

    async def execute_tool(
        self,
        *,
        identity: "AgentIdentity",
        tool_name: ToolName,
        tool_call_id: str,
        arguments: ToolArguments,
        deadline: datetime,
    ) -> ToolInvocationResult:
        """在绝对截止时间内执行一次只读工具，瞬时故障最多重试一次。"""
        # 1、仅从运行时身份和严格参数模型构造信封，模型不能提供内部字段。
        request_id = str(uuid4())
        try:
            validated_arguments = validate_tool_arguments(
                tool_name, arguments.model_dump(mode="json", by_alias=True)
            )
            payload = AgentToolInvokeRequest(
                contract_version=TOOL_CONTRACT_VERSION,
                tool_call_id=tool_call_id,
                identity=AgentToolIdentity(
                    tenant_id=identity.tenant_id,
                    customer_id=identity.customer_id,
                    conversation_id=identity.conversation_id,
                    run_id=identity.run_id,
                    trigger_message_id=identity.trigger_message_id,
                    trigger_sequence=identity.trigger_sequence,
                ),
                arguments=validated_arguments.model_dump(mode="json", by_alias=True),
            ).model_dump(mode="json", by_alias=True)
        except ValueError:
            raise JavaToolProtocolError from None
        path = f"/api/internal/v1/agent/tools/{tool_name.value}/invoke"

        # 2、只对冻结白名单中的瞬时错误执行一次重试。
        for attempt in range(2):
            try:
                response = await self._send_once(path, request_id, payload, deadline)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, TimeoutError):
                if attempt == 0 and self._remaining(deadline) > 0:
                    continue
                raise JavaToolUnavailableError from None
            except (httpx.TimeoutException, httpx.RequestError):
                raise JavaToolUnavailableError from None

            if response.status_code in _RETRYABLE_STATUS_CODES and attempt == 0:
                await response.aclose()
                await self._wait_before_retry(response, deadline)
                continue

            # 3、状态码先映射为安全异常；成功正文再执行严格校验。
            try:
                remaining = self._remaining(deadline)
                if remaining <= 0:
                    raise TimeoutError
                async with asyncio.timeout(remaining):
                    return await self._parse_response(
                        response,
                        request_id=request_id,
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                    )
            except (httpx.ReadTimeout, TimeoutError):
                if attempt == 0 and self._remaining(deadline) > 0:
                    continue
                raise JavaToolUnavailableError from None
            finally:
                await response.aclose()

        raise JavaToolUnavailableError

    async def _send_once(
        self,
        path: str,
        request_id: str,
        payload: dict[str, object],
        deadline: datetime,
    ) -> httpx.Response:
        """发送一次流式请求，避免在大小校验前载入完整正文。"""
        # 1、把各阶段超时压缩到当前剩余总预算内。
        remaining = self._remaining(deadline)
        if remaining <= 0:
            raise TimeoutError
        connect, read, write, pool = self._timeouts
        timeout = httpx.Timeout(
            connect=min(connect, remaining),
            read=min(read, remaining),
            write=min(write, remaining),
            pool=min(pool, remaining),
        )
        # 2、URL只由固定枚举路径生成，HTTPX按stream模式交回响应。
        request = self._client.build_request(
            "POST", path, headers={"X-Request-Id": request_id}, json=payload
        )
        request.extensions["timeout"] = timeout.as_dict()
        async with asyncio.timeout(remaining):
            return await self._client.send(request, stream=True)

    async def _parse_response(
        self,
        response: httpx.Response,
        *,
        request_id: str,
        tool_call_id: str,
        tool_name: ToolName,
    ) -> ToolInvocationResult:
        """映射状态码并校验成功信封的大小、类型及关联字段。"""
        # 1、错误状态不读取正文，避免原始错误内容进入异常或日志。
        if response.status_code in _ACCESS_DENIED_STATUS_CODES:
            raise JavaToolAccessDeniedError
        if response.status_code == 404:
            raise JavaToolNotFoundError
        if response.status_code == 429 or response.status_code >= 500:
            raise JavaToolUnavailableError
        if response.status_code < 200 or response.status_code >= 300:
            raise JavaToolProtocolError

        # 2、Content-Length与实际流式字节数均受配置上限约束。
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                if int(content_length) > self._max_response_bytes:
                    raise JavaToolProtocolError
            except ValueError:
                raise JavaToolProtocolError from None
        content = bytearray()
        async for chunk in response.aiter_bytes():
            content.extend(chunk)
            if len(content) > self._max_response_bytes:
                raise JavaToolProtocolError

        # 3、只接受JSON和冻结响应模型，并复核三个关联字段。
        if "application/json" not in response.headers.get("Content-Type", "").lower():
            raise JavaToolProtocolError
        try:
            result = AgentToolInvokeResponse.model_validate(json.loads(content))
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError):
            raise JavaToolProtocolError from None
        if (
            result.request_id != request_id
            or result.tool_call_id != tool_call_id
            or result.tool_name != tool_name
        ):
            raise JavaToolProtocolError
        return result

    async def _wait_before_retry(self, response: httpx.Response, deadline: datetime) -> None:
        """按Retry-After等待，但不超过配置上限和当前deadline。"""
        # 1、仅接受非负秒数，日期或畸形值退化为0。
        try:
            requested = max(0.0, float(response.headers.get("Retry-After", "0")))
        except ValueError:
            requested = 0.0
        delay = min(requested, self._retry_after_max_seconds, self._remaining(deadline))
        # 2、等待为0时直接进行下一次尝试。
        if delay > 0:
            await asyncio.sleep(delay)

    @staticmethod
    def _remaining(deadline: datetime) -> float:
        """返回UTC绝对截止时间的剩余秒数。"""
        normalized = deadline if deadline.tzinfo is not None else deadline.replace(tzinfo=UTC)
        return max(0.0, (normalized - datetime.now(UTC)).total_seconds())


def build_java_http_client(settings: Settings) -> httpx.AsyncClient:
    """根据Settings创建唯一进程级HTTPX客户端。"""
    # 1、配置存在时执行传输门禁，避免凭据发往公网HTTP。
    validate_java_transport(settings)
    if settings.java_base_url is None or settings.java_service_token is None:
        raise ValueError("Java工具客户端缺少必需配置")
    # 2、固定连接池、代理、重定向和服务认证头。
    return httpx.AsyncClient(
        base_url=str(settings.java_base_url),
        headers={
            "X-AICare-Agent-Service-Token": settings.java_service_token.get_secret_value(),
            "Accept": "application/json",
            "User-Agent": f"{settings.service_name}/{settings.service_version}",
        },
        limits=httpx.Limits(
            max_connections=settings.java_max_connections,
            max_keepalive_connections=settings.java_max_keepalive_connections,
        ),
        trust_env=False,
        follow_redirects=False,
    )


def log_private_http_warning(settings: Settings) -> None:
    """显式启用私网HTTP时记录不含主机、路径或凭据的稳定警告。"""
    if settings.java_base_url is not None and settings.java_base_url.scheme == "http":
        logger.warning("JAVA_TOOL_PRIVATE_HTTP_ENABLED")
