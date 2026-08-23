"""审计 Redis RunStore 是否满足生产持久化、高可用与安全门槛。

审计只执行INFO、ROLE、ACL WHOAMI和CONFIG GET等最小只读命令，检查Redis 7+、AOF、
安全fsync、noeviction、非default ACL、可写master和至少一个副本。报告只含稳定代码，
不保留目标主机、账号、拓扑或命令原始响应；当前客户端明确不支持Redis Cluster。
"""

# logging在非生产环境记录稳定审计代码。
import logging

# dataclass用于不可变finding/report值对象。
from dataclasses import dataclass

# Protocol声明审计所需的最小Redis客户端方法。
from typing import Protocol

# RedisError覆盖redis-py连接、协议和命令异常。
from redis.exceptions import RedisError

from aicare_agent_service.config import Environment

# 使用模块路径作为logger名称。
logger = logging.getLogger(__name__)


class RedisAuditClient(Protocol):
    """只读审计所需Redis客户端协议，便于测试注入Fake。"""

    async def info(self, section: str) -> dict[str, object]:
        """读取指定INFO区段。"""
        ...

    async def execute_command(self, *args: str) -> object:
        """执行ROLE或ACL WHOAMI；*args收集不定数量的位置参数。"""
        ...

    async def config_get(self, parameter: str) -> dict[str, object]:
        """只读获取单个Redis配置项。"""
        ...


@dataclass(frozen=True)
class RedisReadinessFinding:
    """单个不合规项；只保存稳定代码，不保存基础设施原值。"""

    # 机器可识别的大写诊断代码。
    code: str


@dataclass(frozen=True)
class RedisReadinessReport:
    """一次Redis审计的不可变finding元组；空元组表示通过。"""

    # tuple不可变，避免门禁判断后被调用者修改。
    findings: tuple[RedisReadinessFinding, ...]


class RedisReadinessError(RuntimeError):
    """生产Redis实例未满足RunStore的持久化与高可用门槛。"""

    def __init__(self, report: RedisReadinessReport) -> None:
        """将报告中的稳定代码连接为安全异常消息。"""
        # 生成器表达式逐个读取code，中文顿号只用于可读展示。
        codes = "、".join(finding.code for finding in report.findings)
        # 不接收或拼接Redis响应原文。
        super().__init__(f"Redis生产环境门禁未通过：{codes}")


async def audit_redis_readiness(client: RedisAuditClient) -> RedisReadinessReport:
    """使用最小只读命令审计Redis，不保留或输出目标端的身份信息。"""
    # 任意命令失败都降为单个稳定不可用代码，避免部分数据被误判为合规。
    try:
        # server提供版本；persistence提供AOF；memory提供淘汰策略。
        server = await client.info("server")
        persistence = await client.info("persistence")
        memory = await client.info("memory")
        # replication和cluster用于判断主从与客户端拓扑兼容性。
        replication = await client.info("replication")
        cluster = await client.info("cluster")
        # ROLE返回当前节点角色；ACL WHOAMI确认不是共享default用户。
        role = await client.execute_command("ROLE")
        acl_user = await client.execute_command("ACL", "WHOAMI")
        # appendfsync必须是everysec或always，禁止no等不安全策略。
        appendfsync = await client.config_get("appendfsync")
    except (RedisError, OSError, ValueError, TypeError):
        # 报告不包含原始异常正文、地址或凭据。
        return _report("REDIS_AUDIT_UNAVAILABLE")

    # 按固定检查顺序累积所有不合规代码，一次报告完整问题集合。
    codes: list[str] = []
    # Redis主版本低于7或版本文本畸形均不受支持。
    if not _is_version_at_least(_text(server.get("redis_version")), 7):
        codes.append("REDIS_VERSION_UNSUPPORTED")
    # AOF字段在不同响应解码下可能是"1"或"yes"。
    if _text(persistence.get("aof_enabled")) not in {"1", "yes"}:
        codes.append("REDIS_AOF_DISABLED")
    # lower统一大小写后检查允许的落盘策略。
    if _text(appendfsync.get("appendfsync")).lower() not in {"everysec", "always"}:
        codes.append("REDIS_APPEND_FSYNC_UNSAFE")
    # noeviction防止内存压力静默淘汰run ledger或租约相关key。
    if _text(memory.get("maxmemory_policy")).lower() != "noeviction":
        codes.append("REDIS_MAXMEMORY_POLICY_UNSAFE")
    # 先安全转换ACL响应，再区分不可用和危险default用户。
    acl_identity = _text(acl_user)
    if not isinstance(acl_user, (str, bytes)) or not acl_identity:
        codes.append("REDIS_ACL_UNAVAILABLE")
    elif acl_identity.lower() == "default":
        codes.append("REDIS_DEFAULT_USER_FORBIDDEN")
    # ROLE响应可能是bytes/list，辅助函数只提取第一个角色字段。
    role_name = _role_name(role)
    # Cluster即使本身高可用，当前Redis.from_url单节点客户端也不能正确适配，必须阻断。
    if _text(cluster.get("cluster_enabled")) in {"1", "yes"}:
        codes.append("REDIS_CLUSTER_CLIENT_UNSUPPORTED")
    # 非Cluster要求可识别且当前节点为可写master。
    elif role_name is None:
        codes.append("REDIS_ROLE_UNAVAILABLE")
    elif role_name != "master":
        codes.append("REDIS_PRIMARY_REQUIRED")
    # 最后要求至少一个已连接副本；这里只证明复制存在，不宣称自动故障切换。
    elif not _has_connected_replica(replication):
        codes.append("REDIS_HA_UNAVAILABLE")
    # 空codes会产生findings=()，代表审计通过。
    return _report(*codes)


async def enforce_redis_readiness(
    client: RedisAuditClient,
    environment: Environment,
) -> RedisReadinessReport:
    """生产阻断不合规实例；非生产环境仅记录稳定代码告警。"""
    # 先执行纯审计获得结构化报告。
    report = await audit_redis_readiness(client)
    # 无finding直接返回，调用者可用于健康状态展示。
    if not report.findings:
        return report
    # 生产环境任一finding都抛错，阻断应用启动。
    if environment is Environment.PRODUCTION:
        raise RedisReadinessError(report)
    # 开发/测试环境允许启动，但只记录稳定代码列表。
    logger.warning(
        "Redis RunStore审计告警：%s",
        "、".join(finding.code for finding in report.findings),
    )
    return report


def _report(*codes: str) -> RedisReadinessReport:
    """把任意数量代码转换成不可变finding报告。"""
    # 生成器逐项创建finding，tuple立即冻结结果。
    return RedisReadinessReport(tuple(RedisReadinessFinding(code) for code in codes))


def _is_version_at_least(version: str, minimum_major: int) -> bool:
    """解析主版本并判断是否达到下限；畸形文本保守返回False。"""
    try:
        # maxsplit=1只需提取第一个点号前的主版本。
        return int(version.split(".", maxsplit=1)[0]) >= minimum_major
    except (AttributeError, ValueError):
        return False


def _role_name(role: object) -> str | None:
    """从ROLE的list/tuple响应提取小写角色名。"""
    # 非序列或空序列都是畸形响应。
    if not isinstance(role, (list, tuple)) or not role:
        return None
    # Redis角色在第一项；_text兼容bytes和str。
    role_name = _text(role[0]).lower()
    # 空字符串用or转换为None，便于上层统一判定不可用。
    return role_name or None


def _has_connected_replica(replication: dict[str, object]) -> bool:
    """判断INFO replication是否报告至少一个连接副本。"""
    try:
        # Redis仍使用connected_slaves字段名；转换为int后比较数量。
        return int(_text(replication.get("connected_slaves"))) >= 1
    except ValueError:
        return False


def _text(value: object) -> str:
    """把受支持的Redis标量安全转成文本，其他类型返回空字符串。"""
    # bytes按UTF-8解码；失败时不泄漏repr并保守返回空。
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return ""
    # str保持原样，int转成十进制字符串。
    if isinstance(value, (str, int)):
        return str(value)
    # list/dict/None等复杂值不参与文本比较。
    return ""
