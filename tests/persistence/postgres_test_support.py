"""真实PostgreSQL集成测试的临时连接配置辅助函数。"""

from urllib.parse import quote, unquote, urlsplit, urlunsplit


def prepare_postgres_test_connection(dsn: str) -> tuple[str, str]:
    """将密码从conninfo移除，交由当前测试进程的PGPASSWORD提供。"""
    parsed = urlsplit(dsn)
    if not parsed.scheme or parsed.username is None or parsed.password is None:
        raise ValueError("测试PostgreSQL DSN必须包含scheme、用户名和密码")
    if parsed.hostname is None:
        raise ValueError("测试PostgreSQL DSN必须包含主机")

    username = quote(unquote(parsed.username), safe="")
    hostname = parsed.hostname
    if ":" in hostname:
        hostname = f"[{hostname}]"
    netloc = f"{username}@{hostname}"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"

    sanitized_dsn = urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, ""))
    return sanitized_dsn, unquote(parsed.password)
