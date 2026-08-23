"""使用 Redis Lua 原子实现 Agent run ledger、租约、取消和清理保护。

Redis只保存Java身份、安全摘要、状态和时间，不保存用户/AI正文。所有跨多个key的状态迁移
由Lua在Redis服务端原子执行；Python负责参数校验、key哈希、脚本调用和稳定异常映射。
固定hash tag让相关key在同一Redis Cluster槽，但当前客户端仍明确不支持Cluster拓扑。
"""

# hashlib对业务ID和租约生成SHA-256，避免明文出现在Redis key或记录中。
import hashlib

# re编译固定hash tag、摘要和错误码格式。
import re

# secrets使用系统安全随机源生成不可预测租约/guard token。
import secrets

# UTC和datetime把Redis毫秒时间戳转换成带时区时间。
from datetime import UTC, datetime

# Any用于兼容redis-py脚本可能返回bytes/str/list等动态类型。
from typing import Any
from urllib.parse import unquote, urlsplit

from pydantic import SecretStr, ValidationError
from redis.asyncio import Redis
from redis.asyncio.sentinel import Sentinel
from redis.exceptions import RedisError

from aicare_agent_service.config import RedisMode
from aicare_agent_service.contracts.agent_run import AgentRunRequest
from aicare_agent_service.persistence.models import (
    InvalidRunTransitionError,
    RunBeginOutcome,
    RunBeginResult,
    RunLeaseLostError,
    RunRecord,
    RunStatus,
    RunStoreUnavailableError,
)
from aicare_agent_service.persistence.scripts import (
    ACQUIRE_CLEANUP_GUARD_LUA,
    BEGIN_RUN_LUA,
    CHECK_CONVERSATION_ACTIVE_LUA,
    RELEASE_CLEANUP_GUARD_LUA,
    RENEW_LEASE_LUA,
    REQUEST_CANCEL_LUA,
    TERMINATE_RUN_LUA,
)

# 三个预编译正则分别验证key前缀、SHA-256和稳定错误码。
_HASH_TAG_PATTERN = re.compile(r"\{[^{}]+\}")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ERROR_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")


def build_redis_client(
    # Redis URL用SecretStr传入，避免普通repr显示密码。
    redis_url: SecretStr,
    # ``*``之后的连接参数必须按名称传入，降低位置顺序误配风险。
    *,
    redis_mode: RedisMode = RedisMode.STANDALONE,
    sentinel_endpoints: tuple[tuple[str, int], ...] = (),
    sentinel_master_name: str | None = None,
    sentinel_username: str | None = None,
    sentinel_password: SecretStr | None = None,
    max_connections: int,
    socket_timeout_seconds: float,
    health_check_interval_seconds: int,
) -> Redis:
    """创建显式有界、需由应用 lifespan 关闭的异步 Redis 连接池。

    返回redis.asyncio.Redis；本函数只创建客户端，不主动连接。decode_responses让正常响应
    解码为字符串；连接数、连接/命令超时及空闲健康检查都来自集中配置。
    """
    if redis_mode is RedisMode.SENTINEL:
        parsed = urlsplit(redis_url.get_secret_value())
        if not sentinel_endpoints or sentinel_master_name is None:
            raise ValueError("Sentinel客户端缺少发现端点或master名称")
        if sentinel_username is None or sentinel_password is None:
            raise ValueError("Sentinel客户端缺少控制面ACL凭据")
        sentinel = Sentinel(
            sentinel_endpoints,
            min_other_sentinels=1,
            sentinel_kwargs={
                "username": sentinel_username,
                "password": sentinel_password.get_secret_value(),
                "socket_connect_timeout": socket_timeout_seconds,
                "socket_timeout": socket_timeout_seconds,
            },
            username=unquote(parsed.username) if parsed.username else None,
            password=unquote(parsed.password) if parsed.password else None,
            db=int(parsed.path.lstrip("/") or "0"),
            decode_responses=True,
            max_connections=max_connections,
            socket_connect_timeout=socket_timeout_seconds,
            socket_timeout=socket_timeout_seconds,
            health_check_interval=health_check_interval_seconds,
        )
        return sentinel.master_for(sentinel_master_name)
    # from_url解析协议、主机、端口、数据库和凭据；明文仅交给redis-py。
    return Redis.from_url(
        redis_url.get_secret_value(),
        # 将bytes响应自动解码为str，解析器仍保留bytes兼容以防Fake/特殊响应。
        decode_responses=True,
        max_connections=max_connections,
        socket_connect_timeout=socket_timeout_seconds,
        socket_timeout=socket_timeout_seconds,
        health_check_interval=health_check_interval_seconds,
    )


class RedisRunStore:
    """使用Redis Lua原子维护run幂等、conversation单飞和终态元数据。"""

    def __init__(
        # client是lifespan持有的异步Redis客户端。
        self,
        client: Redis,
        # 以下参数强制关键字传入。
        *,
        lease_seconds: int,
        retention_seconds: int,
        cleanup_guard_seconds: int = 60,
        key_prefix: str = "aicare:agent:{run-store}",
    ) -> None:
        """校验时限/key前缀并向客户端注册七个原子Lua脚本。"""
        # 租约和guard必须为正，记录保留期不得短于租约。
        if lease_seconds <= 0 or retention_seconds < lease_seconds or cleanup_guard_seconds <= 0:
            raise ValueError("Redis run租约和保留期配置无效")
        # 删除尾部冒号，避免后续拼接key时产生双冒号。
        normalized_prefix = key_prefix.rstrip(":")
        # 所有多key脚本要求固定{...} hash tag，以保证键落在同一槽。
        if not _HASH_TAG_PATTERN.search(normalized_prefix):
            raise ValueError("Redis RunStore key前缀必须包含固定hash tag")

        # 保存客户端和换算后的内部配置；Lua的PX单位是毫秒。
        self._client = client
        self._lease_ms = lease_seconds * 1000
        self._retention_seconds = retention_seconds
        self._cleanup_guard_seconds = cleanup_guard_seconds
        self._key_prefix = normalized_prefix
        # register_script返回可调用Script对象，并自动处理EVALSHA缓存缺失。
        self._begin_script = client.register_script(BEGIN_RUN_LUA)
        self._check_conversation_active_script = client.register_script(
            CHECK_CONVERSATION_ACTIVE_LUA
        )
        self._acquire_cleanup_guard_script = client.register_script(ACQUIRE_CLEANUP_GUARD_LUA)
        self._release_cleanup_guard_script = client.register_script(RELEASE_CLEANUP_GUARD_LUA)
        self._renew_script = client.register_script(RENEW_LEASE_LUA)
        self._request_cancel_script = client.register_script(REQUEST_CANCEL_LUA)
        self._terminate_script = client.register_script(TERMINATE_RUN_LUA)

    @property
    def key_prefix(self) -> str:
        """返回不含业务标识的固定Redis key命名空间。"""
        # 只返回固定命名空间，不含任何run或conversation业务标识。
        return self._key_prefix

    async def begin(self, run: AgentRunRequest, request_digest: str) -> RunBeginResult:
        """原子开始/恢复run，或返回进行中、已完成重放、冲突判定。"""
        # 入站摘要必须已是规范64位小写SHA-256。
        _require_sha256(request_digest)
        # 原始租约只返回当前执行者；Redis记录另存其摘要用于安全审计。
        lease_token = secrets.token_urlsafe(32)
        lease_token_digest = _sha256(lease_token)
        # Redis错误或脚本畸形统一映射为不含基础设施细节的稳定异常。
        try:
            # keys与Lua KEYS[1..4]顺序严格对应run、lease、active-run和cleanup guard。
            raw_result = await self._begin_script(
                keys=[
                    self._run_key(run.run_id),
                    self._lease_key(run.conversation_id),
                    self._active_run_key(run.conversation_id),
                    self._cleanup_guard_key(run.conversation_id),
                ],
                # args与Lua ARGV顺序严格对应摘要、token、时限和Java身份。
                args=[
                    request_digest,
                    lease_token,
                    lease_token_digest,
                    self._lease_ms,
                    self._retention_seconds,
                    run.tenant_id,
                    run.customer_id,
                    run.conversation_id,
                    run.run_id,
                    run.trigger_message_id,
                    run.trigger_sequence,
                    self._run_key(run.run_id),
                ],
                # 显式传client执行已注册脚本。
                client=self._client,
            )
            # 脚本结果必须是非空序列，第一项为稳定判定码。
            code = _script_code(raw_result)
            # 集合推导生成所有允许的枚举字符串，未知响应视为基础设施异常。
            if code not in {outcome.value for outcome in RunBeginOutcome}:
                raise RunStoreUnavailableError()

            # 把已验证字符串转换为RunBeginOutcome枚举。
            outcome = RunBeginOutcome(code)
            # Lua原子迁移后读取安全记录，供生命周期核对和重放。
            record = await self.get(run.run_id)
            # 只有首次开始或过期恢复才拥有刚生成的租约。
            owns_lease = outcome in {RunBeginOutcome.STARTED, RunBeginOutcome.RESUMED}
            # SecretStr保护原始租约，其他判定明确返回None。
            return RunBeginResult(
                outcome=outcome,
                record=record,
                lease_token=SecretStr(lease_token) if owns_lease else None,
            )
        except (RunStoreUnavailableError, RedisError):
            # ``from None``隐藏redis-py原始地址/命令异常上下文。
            raise RunStoreUnavailableError() from None

    async def get(self, run_id: str) -> RunRecord | None:
        """读取并严格解析一个run hash；不存在返回None，畸形记录安全失败。"""
        try:
            # run ID先哈希后进入key，不把Java业务ID暴露在Redis keyspace。
            raw = await self._client.hgetall(self._run_key(run_id))
        except RedisError:
            raise RunStoreUnavailableError() from None
        # 空字典代表key不存在或已过保留期。
        if not raw:
            return None
        # Pydantic再次验证所有摘要、状态和字段类型。
        try:
            return _parse_record(raw)
        except (KeyError, TypeError, ValueError, ValidationError):
            raise RunStoreUnavailableError() from None

    async def renew_lease(self, run_id: str, lease_token: str) -> None:
        """校验run所属会话后原子续租并刷新安全元数据TTL。"""
        # 必须先从run记录取conversation ID，不能让调用者另传不一致会话。
        conversation_id = await self._required_conversation_id(run_id)
        try:
            result = await self._renew_script(
                keys=[
                    self._run_key(run_id),
                    self._lease_key(conversation_id),
                    self._active_run_key(conversation_id),
                ],
                args=[lease_token, self._lease_ms, self._retention_seconds],
                client=self._client,
            )
        except RedisError:
            raise RunStoreUnavailableError() from None
        # 把Lua结果映射为成功、租约丢失、非法迁移或服务不可用。
        self._require_mutation_result(result, success_codes={"RENEWED"})

    async def request_cancel(self, run_id: str) -> None:
        """记录外部取消意图；只有当前lease持有者能随后写入CANCELLED。"""
        # 取消意图不要求调用者持有租约，Java可在外部请求停止当前run。
        try:
            result = await self._request_cancel_script(
                keys=[self._run_key(run_id)],
                args=[self._retention_seconds],
                client=self._client,
            )
        except RedisError:
            raise RunStoreUnavailableError() from None
        self._require_mutation_result(
            result,
            success_codes={"CANCEL_REQUESTED", "ALREADY_REQUESTED"},
        )

    async def is_cancel_requested(self, run_id: str) -> bool:
        """检查run hash是否存在取消时间字段。"""
        try:
            value = await self._client.hexists(
                self._run_key(run_id),
                "cancel_requested_at_ms",
            )
        except RedisError:
            raise RunStoreUnavailableError() from None
        # redis-py可能返回bool或0/1，bool统一转换。
        return bool(value)

    async def is_conversation_active(self, conversation_id: str) -> bool:
        """只返回会话是否仍有RUNNING run或有效lease，不暴露run正文。"""
        # Lua会同时检查active-run引用与lease，并自愈指向过期run的悬空active key。
        try:
            result = await self._check_conversation_active_script(
                keys=[
                    self._active_run_key(conversation_id),
                    self._lease_key(conversation_id),
                ],
                client=self._client,
            )
        except RedisError:
            raise RunStoreUnavailableError() from None
        # 脚本只返回0或1，转换为整数后判断。
        return int(result) == 1

    async def acquire_cleanup_guard(self, conversation_id: str) -> str | None:
        """在短窗口内阻止新run启动；活跃run存在时不获取。"""
        # 每次申请生成独立高熵token，释放时必须原样匹配。
        token = secrets.token_urlsafe(32)
        try:
            acquired = await self._acquire_cleanup_guard_script(
                keys=[
                    self._active_run_key(conversation_id),
                    self._lease_key(conversation_id),
                    self._cleanup_guard_key(conversation_id),
                ],
                args=[token, self._cleanup_guard_seconds],
                client=self._client,
            )
            # 0代表活跃、已有guard或竞争失败，不属于基础设施异常。
            if int(acquired) != 1:
                return None
            return token
        except RedisError:
            raise RunStoreUnavailableError() from None

    async def release_cleanup_guard(self, conversation_id: str, token: str) -> None:
        """通过比较后删除Lua释放guard，错误token不会删除他人保护。"""
        try:
            await self._release_cleanup_guard_script(
                keys=[self._cleanup_guard_key(conversation_id)],
                args=[token],
                client=self._client,
            )
        except RedisError:
            raise RunStoreUnavailableError() from None

    async def complete(
        self,
        run_id: str,
        lease_token: str,
        checkpoint_id: str,
        final_digest: str,
    ) -> None:
        """校验checkpoint/摘要后提交COMPLETED终态。"""
        # checkpoint ID不能为空或全空白。
        if not checkpoint_id.strip():
            raise ValueError("checkpoint ID不能为空")
        # 终态语义摘要必须为规范SHA-256。
        _require_sha256(final_digest)
        await self._terminate(
            run_id,
            lease_token,
            RunStatus.COMPLETED,
            checkpoint_id=checkpoint_id,
            final_digest=final_digest,
        )

    async def fail(self, run_id: str, lease_token: str, error_code: str) -> None:
        """校验稳定错误码后提交FAILED终态。"""
        if not _ERROR_CODE_PATTERN.fullmatch(error_code):
            raise ValueError("RunStore错误码格式无效")
        await self._terminate(
            run_id,
            lease_token,
            RunStatus.FAILED,
            error_code=error_code,
        )

    async def cancel(self, run_id: str, lease_token: str) -> None:
        """提交CANCELLED终态；必须仍持有匹配租约。"""
        await self._terminate(run_id, lease_token, RunStatus.CANCELLED)

    async def _terminate(
        self,
        run_id: str,
        lease_token: str,
        target_status: RunStatus,
        *,
        checkpoint_id: str = "",
        final_digest: str = "",
        error_code: str = "",
    ) -> None:
        """统一执行三种终态Lua迁移；可选字段只对相应状态生效。"""
        conversation_id = await self._required_conversation_id(run_id)
        # Lua同时验证状态和原始lease token，并原子写终态、删lease/active、设保留TTL。
        try:
            result = await self._terminate_script(
                keys=[
                    self._run_key(run_id),
                    self._lease_key(conversation_id),
                    self._active_run_key(conversation_id),
                ],
                args=[
                    lease_token,
                    target_status.value,
                    checkpoint_id,
                    final_digest,
                    error_code,
                    self._retention_seconds,
                    self._run_key(run_id),
                ],
                client=self._client,
            )
        except RedisError:
            raise RunStoreUnavailableError() from None
        self._require_mutation_result(
            result,
            success_codes={"TERMINATED", "ALREADY_TERMINAL"},
        )

    async def _required_conversation_id(self, run_id: str) -> str:
        """从run记录读取必需conversation ID，不存在视为非法迁移。"""
        try:
            value = await self._client.hget(self._run_key(run_id), "conversation_id")
        except RedisError:
            raise RunStoreUnavailableError() from None
        if value is None:
            raise InvalidRunTransitionError()
        return _text(value)

    @staticmethod
    def _require_mutation_result(result: Any, *, success_codes: set[str]) -> None:
        """把Lua稳定返回码映射为领域异常。"""
        code = _script_code(result)
        # 幂等终态ALREADY_TERMINAL可由调用方列入success_codes。
        if code in success_codes:
            return
        if code == "LEASE_LOST":
            raise RunLeaseLostError()
        if code in {"NOT_FOUND", "INVALID_TRANSITION"}:
            raise InvalidRunTransitionError()
        raise RunStoreUnavailableError()

    def _run_key(self, run_id: str) -> str:
        """生成不含明文run ID的hash key。"""
        return f"{self._key_prefix}:run:{_sha256(run_id)}"

    def _lease_key(self, conversation_id: str) -> str:
        """生成conversation单飞lease key。"""
        return f"{self._key_prefix}:conversation:{_sha256(conversation_id)}:lease"

    def _active_run_key(self, conversation_id: str) -> str:
        """生成无TTL活跃run引用key；终态时原子删除。"""
        return f"{self._key_prefix}:conversation:{_sha256(conversation_id)}:active-run"

    def _cleanup_guard_key(self, conversation_id: str) -> str:
        """生成短时checkpoint清理保护key。"""
        return f"{self._key_prefix}:conversation:{_sha256(conversation_id)}:cleanup-guard"


def _parse_record(raw: dict[Any, Any]) -> RunRecord:
    """将Redis hash响应转换为严格RunRecord。"""
    # 字典推导统一把bytes/str键值转换为文本。
    values = {_text(key): _text(value) for key, value in raw.items()}
    # 必填字段用[]访问，缺失立即KeyError；可选字段用get并转换None。
    return RunRecord(
        tenant_id=values["tenant_id"],
        customer_id=values["customer_id"],
        conversation_id=values["conversation_id"],
        run_id=values["run_id"],
        trigger_message_id=values["trigger_message_id"],
        trigger_sequence=int(values["trigger_sequence"]),
        request_digest=values["request_digest"],
        status=values["status"],
        lease_token_digest=values.get("lease_token_digest") or None,
        lease_expires_at=_datetime_from_ms(values.get("lease_expires_at_ms")),
        cancel_requested_at=_datetime_from_ms(values.get("cancel_requested_at_ms")),
        checkpoint_id=values.get("checkpoint_id") or None,
        final_digest=values.get("final_digest") or None,
        error_code=values.get("error_code") or None,
        started_at=_datetime_from_ms(values["started_at_ms"]),
        updated_at=_datetime_from_ms(values["updated_at_ms"]),
        completed_at=_datetime_from_ms(values.get("completed_at_ms")),
    )


def _datetime_from_ms(value: str | None) -> datetime | None:
    """把Redis毫秒字符串转换为UTC datetime；缺失保持None。"""
    if value is None:
        return None
    return datetime.fromtimestamp(int(value) / 1000, tz=UTC)


def _script_code(result: Any) -> str:
    """严格读取Lua数组结果第一项稳定代码。"""
    if not isinstance(result, (list, tuple)) or not result:
        raise RunStoreUnavailableError()
    return _text(result[0])


def _text(value: Any) -> str:
    """把Redis标量响应转为字符串。"""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    return str(value)


def _sha256(value: str) -> str:
    """返回字符串UTF-8字节的64位小写SHA-256。"""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_sha256(value: str) -> None:
    """拒绝不是规范小写SHA-256的摘要。"""
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError("RunStore摘要必须是小写SHA-256")
