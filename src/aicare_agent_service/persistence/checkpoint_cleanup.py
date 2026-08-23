"""实现 checkpoint 保留清理的候选发现、活跃核对与逐 thread 删除。

PostgreSQL目录只读取thread_id和最新checkpoint_id，不读取或解密状态正文；服务再通过
Redis确认会话无活跃run，取得短时guard后调用LangGraph官方``adelete_thread``删除。
单个thread失败只计数并留待下次重试，不中止整批任务。
"""

# asyncio提供单thread删除超时保护。
import asyncio

# dataclass生成不可变结果数据类。
from dataclasses import dataclass

# UTC和datetime用于统一比较带时区时间。
from datetime import UTC, datetime

# Any兼容第三方Saver内部游标接口；Protocol声明可替换依赖边界。
from typing import Any, Protocol

# uuid_utils可解析LangGraph当前使用的UUIDv6/v7时间戳。
from uuid_utils import UUID


# frozen防修改，slots限制动态字段并减少轻量候选对象开销。
@dataclass(frozen=True, slots=True)
class CheckpointCandidate:
    """一个超过保留截止时间的会话最新checkpoint候选。"""

    # LangGraph thread_id；本项目中等于Java conversationId。
    thread_id: str
    # 当前线程最新checkpoint标识。
    checkpoint_id: str
    # 从时间UUID解析出的UTC更新时间。
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class CheckpointCleanupResult:
    """一次清理批次的不可变计数汇总，所有字段默认从零开始。"""

    # 被目录返回并检查的候选数。
    scanned: int = 0
    # dry-run可删或apply已尝试删除的非活跃候选数。
    eligible: int = 0
    # 因活跃run、lease或guard而跳过的数量。
    active: int = 0
    # 成功调用官方线程删除接口的数量。
    deleted: int = 0
    # 单thread删除失败或超时的数量。
    failed: int = 0


class CheckpointCatalog(Protocol):
    """候选查询和线程删除的数据目录协议。"""

    async def list_candidates(
        self,
        cutoff: datetime,
        limit: int,
    ) -> list[CheckpointCandidate]:
        """返回截止时间前最多limit个候选。"""
        ...

    async def delete_thread(self, thread_id: str) -> None:
        """删除一个thread的全部LangGraph checkpoint。"""
        ...


class ConversationActivityStore(Protocol):
    """清理器所需的最小Redis活跃状态协议。"""

    async def is_conversation_active(self, conversation_id: str) -> bool:
        """判断会话是否有运行中run或有效租约。"""
        ...

    async def acquire_cleanup_guard(self, conversation_id: str) -> str | None:
        """无活跃执行时取得清理token，否则返回None。"""
        ...

    async def release_cleanup_guard(self, conversation_id: str, token: str) -> None:
        """用所有权token释放清理保护。"""
        ...


class PostgresCheckpointCatalog:
    """只从LangGraph表读取thread/checkpoint标识，删除委托官方Saver。"""

    def __init__(self, saver: Any, *, scan_limit: int = 10_000) -> None:
        """保存官方Saver并限制单次SQL最多扫描的线程数。"""
        # 扫描上限必须为正，避免LIMIT无效或查询范围失控。
        if scan_limit <= 0:
            raise ValueError("checkpoint候选扫描上限必须为正数")
        # 前导下划线表示只供目录实现内部使用。
        self._saver = saver
        self._scan_limit = scan_limit

    async def list_candidates(
        self,
        cutoff: datetime,
        limit: int,
    ) -> list[CheckpointCandidate]:
        """查询每个thread最新ID，解析时间后筛出过期候选。"""
        # SQL不选择checkpoint、metadata或blob正文，只按thread聚合最新ID。
        query = """
            SELECT thread_id, MAX(checkpoint_id) AS checkpoint_id
            FROM checkpoints
            GROUP BY thread_id
            ORDER BY MAX(checkpoint_id)
            LIMIT %s
        """
        # _cursor是官方AsyncPostgresSaver提供的异步游标上下文。
        async with self._saver._cursor() as cursor:
            # 参数化%s占位符，避免把limit拼接进SQL字符串。
            await cursor.execute(query, (self._scan_limit,))
            # 一次取回受scan_limit限制的安全标识行。
            rows = await cursor.fetchall()

        # 用列表累积真正早于cutoff且ID可解析的候选。
        candidates: list[CheckpointCandidate] = []
        # 每行通常是支持按列名读取的mapping row。
        for row in rows:
            # 未知/损坏ID采取保守跳过，不猜测时间并误删状态。
            try:
                updated_at = checkpoint_timestamp(str(row["checkpoint_id"]))
            except (KeyError, TypeError, ValueError):
                continue
            # 只有严格早于截止时间才进入候选。
            if updated_at < cutoff:
                candidates.append(
                    CheckpointCandidate(
                        thread_id=str(row["thread_id"]),
                        checkpoint_id=str(row["checkpoint_id"]),
                        updated_at=updated_at,
                    )
                )
            # limit约束最终候选数；可能小于底层scan_limit。
            if len(candidates) >= limit:
                break
        # 返回按SQL中最新checkpoint ID排序的候选列表。
        return candidates

    async def delete_thread(self, thread_id: str) -> None:
        """委托LangGraph官方Saver删除指定thread。"""
        await self._saver.adelete_thread(thread_id)


def checkpoint_timestamp(checkpoint_id: str) -> datetime:
    """从LangGraph当前使用的时间UUID中读取UTC时间；未知格式保守拒绝。"""
    # UUID构造器会验证文本格式；from None隐藏原始异常上下文。
    try:
        value = UUID(checkpoint_id)
    except (TypeError, ValueError):
        raise ValueError("checkpoint ID不是有效时间UUID") from None
    # 当前只接受包含毫秒时间语义的UUIDv6和UUIDv7。
    if value.version not in {6, 7}:
        raise ValueError("checkpoint ID不是受支持的时间UUID")
    # uuid_utils.timestamp为毫秒，除以1000传给datetime并明确UTC时区。
    return datetime.fromtimestamp(value.timestamp / 1000, tz=UTC)


class CheckpointCleanupService:
    """协调目录、Redis活跃状态、guard和逐线程限时删除。"""

    def __init__(
        self,
        catalog: CheckpointCatalog,
        activity_store: ConversationActivityStore,
        *,
        delete_timeout_seconds: float = 30,
    ) -> None:
        # 每次删除超时必须为正，并应短于Redis cleanup guard有效期。
        if delete_timeout_seconds <= 0:
            raise ValueError("checkpoint删除超时必须为正数")
        # 保存注入依赖，方便单元测试使用Fake而无需真实基础设施。
        self._catalog = catalog
        self._activity_store = activity_store
        self._delete_timeout_seconds = delete_timeout_seconds

    async def run(
        self,
        *,
        cutoff: datetime,
        limit: int,
        dry_run: bool,
    ) -> CheckpointCleanupResult:
        """执行一批dry-run或真实清理并返回计数，不返回会话正文。"""
        # naive datetime没有时区，跨部署地区比较可能误删，因此拒绝。
        if cutoff.tzinfo is None:
            raise ValueError("checkpoint清理截止时间必须包含时区")
        # 批次大小必须为正数。
        if limit <= 0:
            raise ValueError("checkpoint清理批次必须为正数")

        # 统一转换UTC后查询候选。
        candidates = await self._catalog.list_candidates(cutoff.astimezone(UTC), limit)
        # 以下计数器按候选处理结果累加。
        eligible = 0
        active = 0
        deleted = 0
        failed = 0
        # 串行逐thread处理，避免大量并发删除压垮数据库或争用guard。
        for candidate in candidates:
            # 首次无锁检查快速跳过明显活跃会话。
            if await self._activity_store.is_conversation_active(candidate.thread_id):
                active += 1
                continue
            # dry-run只统计，不获取guard、不执行删除。
            if dry_run:
                eligible += 1
                continue
            # apply模式在删除前原子取得guard，阻止新run与删除并发。
            guard_token = await self._activity_store.acquire_cleanup_guard(candidate.thread_id)
            if guard_token is None:
                active += 1
                continue
            # 成功取得guard后该候选才算本轮可执行。
            eligible += 1
            try:
                try:
                    # wait_for限制单thread删除时间，且超时应严格短于guard TTL。
                    await asyncio.wait_for(
                        self._catalog.delete_thread(candidate.thread_id),
                        timeout=self._delete_timeout_seconds,
                    )
                    # 只有官方删除正常返回才计为deleted。
                    deleted += 1
                except Exception:  # noqa: BLE001 - 单个thread失败需保留给下次重试。
                    # 不泄漏或中断整批；下次任务仍可重试该thread。
                    failed += 1
            finally:
                # 无论删除成功、失败还是超时，都尝试用原token释放guard。
                await self._activity_store.release_cleanup_guard(
                    candidate.thread_id,
                    guard_token,
                )
        # 构造不可变结果对象供CLI输出安全统计。
        return CheckpointCleanupResult(
            scanned=len(candidates),
            eligible=eligible,
            active=active,
            deleted=deleted,
            failed=failed,
        )
