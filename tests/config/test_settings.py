from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from aicare_agent_service.config import (
    CheckpointBackend,
    Environment,
    ModelProviderName,
    RedisMode,
    Settings,
    get_settings,
    validate_production_settings,
)

ENVIRONMENT_VARIABLES = (
    "AICARE_AGENT_ENVIRONMENT",
    "AICARE_AGENT_HOST",
    "AICARE_AGENT_PORT",
    "AICARE_AGENT_LOG_LEVEL",
    "AICARE_AGENT_OUTPUT_MAX_CHARS",
    "AICARE_AGENT_MODEL_PROVIDER",
    "DEEPSEEK_API_KEY",
    "AICARE_AGENT_DEEPSEEK_MODEL",
    "AICARE_AGENT_DEEPSEEK_BASE_URL",
    "AICARE_AGENT_DEEPSEEK_MAX_RETRIES",
    "AICARE_AGENT_MODEL_TIMEOUT_SECONDS",
    "AICARE_AGENT_MODEL_MAX_OUTPUT_TOKENS",
    "AICARE_AGENT_SPECIALIST_TIMEOUT_SECONDS",
    "AICARE_AGENT_SPECIALIST_MAX_OUTPUT_TOKENS",
    "AICARE_AGENT_ANSWER_TIMEOUT_SECONDS",
    "AICARE_AGENT_ANSWER_MAX_OUTPUT_TOKENS",
    "AICARE_AGENT_POSTGRES_DSN",
    "AICARE_AGENT_CHECKPOINT_BACKEND",
    "AICARE_AGENT_REDIS_URL",
    "AICARE_AGENT_REDIS_MODE",
    "AICARE_AGENT_REDIS_SENTINELS",
    "AICARE_AGENT_REDIS_MASTER_NAME",
    "AICARE_AGENT_REDIS_SENTINEL_USERNAME",
    "AICARE_AGENT_REDIS_SENTINEL_PASSWORD",
    "AICARE_AGENT_RUN_LEASE_SECONDS",
    "AICARE_AGENT_RUN_RETENTION_SECONDS",
    "AICARE_AGENT_RUN_HEARTBEAT_SECONDS",
    "AICARE_AGENT_RUN_TIMEOUT_SECONDS",
    "AICARE_AGENT_CHECKPOINT_RETENTION_SECONDS",
    "AICARE_AGENT_CHECKPOINT_CLEANUP_BATCH_SIZE",
    "AICARE_AGENT_CHECKPOINT_CLEANUP_GUARD_SECONDS",
    "AICARE_AGENT_REDIS_MAX_CONNECTIONS",
    "AICARE_AGENT_REDIS_SOCKET_TIMEOUT_SECONDS",
    "AICARE_AGENT_REDIS_HEALTH_CHECK_INTERVAL_SECONDS",
    "LANGGRAPH_AES_KEY",
    "AICARE_AGENT_JAVA_BASE_URL",
    "AICARE_AGENT_JAVA_SERVICE_TOKEN",
    "AICARE_AGENT_JAVA_ALLOW_PRIVATE_HTTP",
    "AICARE_AGENT_JAVA_MAX_CONNECTIONS",
    "AICARE_AGENT_JAVA_MAX_KEEPALIVE_CONNECTIONS",
    "AICARE_AGENT_JAVA_CONNECT_TIMEOUT_SECONDS",
    "AICARE_AGENT_JAVA_READ_TIMEOUT_SECONDS",
    "AICARE_AGENT_JAVA_WRITE_TIMEOUT_SECONDS",
    "AICARE_AGENT_JAVA_POOL_TIMEOUT_SECONDS",
    "AICARE_AGENT_JAVA_RETRY_AFTER_MAX_SECONDS",
    "AICARE_AGENT_JAVA_RESPONSE_MAX_BYTES",
    "AICARE_AGENT_ELASTICSEARCH_NODES",
    "AICARE_AGENT_ELASTICSEARCH_USERNAME",
    "AICARE_AGENT_ELASTICSEARCH_PASSWORD",
    "AICARE_AGENT_ELASTICSEARCH_CA_CERT_PATH",
    "AICARE_AGENT_RAG_ENABLED",
    "AICARE_AGENT_RAG_MODEL_LOCK_PATH",
    "AICARE_AGENT_RAG_MODEL_DIR",
    "AICARE_AGENT_BGE_EMBEDDING_REVISION",
    "AICARE_AGENT_BGE_RERANKER_REVISION",
    "AICARE_AGENT_RABBITMQ_URL",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "AICARE_AGENT_TEST_POSTGRES_DSN",
    "AICARE_AGENT_TEST_REDIS_URL",
)


def test_sentinel_settings_parse_three_endpoints() -> None:
    settings = Settings(
        redis_mode="sentinel",
        redis_sentinels="redis-a:26379,redis-b:26380,redis-c:26381",
        redis_master_name="aicare-agent-master",
        redis_sentinel_username="aicare_sentinel",
        redis_sentinel_password="sentinel-secret",
        _env_file=None,
    )

    assert settings.redis_mode is RedisMode.SENTINEL
    assert settings.redis_sentinel_endpoints == (
        ("redis-a", 26379),
        ("redis-b", 26380),
        ("redis-c", 26381),
    )


def test_sentinel_mode_requires_complete_discovery_and_acl_configuration() -> None:
    with pytest.raises(ValueError, match="Sentinel模式缺少必需配置"):
        Settings(redis_mode="sentinel", _env_file=None)


@pytest.mark.parametrize(
    "invalid_endpoints",
    ["redis-a", "redis-a:not-a-port", "redis-a:26379,,redis-c:26381"],
)
def test_sentinel_endpoints_reject_malformed_values(invalid_endpoints: str) -> None:
    with pytest.raises(ValidationError):
        Settings(
            redis_mode="sentinel",
            redis_sentinels=invalid_endpoints,
            redis_master_name="aicare-agent-master",
            redis_sentinel_username="aicare_sentinel",
            redis_sentinel_password="sentinel-secret",
            _env_file=None,
        )


@pytest.fixture(autouse=True)
def isolate_settings_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for variable_name in ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable_name, raising=False)

    yield


def test_default_development_settings_do_not_require_secrets() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment is Environment.DEVELOPMENT
    assert settings.service_name == "aicare-agent-service"
    assert settings.service_version == "0.1.0"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8090
    assert settings.output_max_chars == 12000
    assert settings.model_provider is ModelProviderName.DEEPSEEK
    assert settings.deepseek_model == "deepseek-v4-pro"
    assert str(settings.deepseek_base_url) == "https://api.deepseek.com/"
    assert settings.deepseek_max_retries == 2
    assert settings.model_timeout_seconds == 30
    assert settings.model_max_output_tokens == 2048
    assert settings.specialist_timeout_seconds == 60
    assert settings.specialist_max_output_tokens == 4096
    assert settings.answer_timeout_seconds == 60
    assert settings.answer_max_output_tokens == 4096
    assert settings.deepseek_api_key is None
    assert settings.checkpoint_backend is CheckpointBackend.MEMORY
    assert settings.agent_redis_url is None
    assert settings.run_lease_seconds == 30
    assert settings.run_retention_seconds == 604800
    assert settings.run_heartbeat_seconds == 10
    assert settings.run_timeout_seconds == 120
    assert settings.checkpoint_retention_seconds == 604800
    assert settings.checkpoint_cleanup_batch_size == 500
    assert settings.checkpoint_cleanup_guard_seconds == 60
    assert settings.redis_max_connections == 20
    assert settings.redis_socket_timeout_seconds == 2
    assert settings.redis_health_check_interval_seconds == 30
    assert settings.checkpoint_encryption_key is None
    assert settings.langsmith_tracing is False


def test_fixed_environment_variable_names_override_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AICARE_AGENT_ENVIRONMENT", "test")
    monkeypatch.setenv("AICARE_AGENT_HOST", "0.0.0.0")
    monkeypatch.setenv("AICARE_AGENT_PORT", "9000")
    monkeypatch.setenv("AICARE_AGENT_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("AICARE_AGENT_OUTPUT_MAX_CHARS", "6000")
    monkeypatch.setenv("AICARE_AGENT_MODEL_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    monkeypatch.setenv("AICARE_AGENT_DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("AICARE_AGENT_DEEPSEEK_BASE_URL", "https://deepseek.example.com/v1")
    monkeypatch.setenv("AICARE_AGENT_DEEPSEEK_MAX_RETRIES", "4")
    monkeypatch.setenv("AICARE_AGENT_MODEL_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("AICARE_AGENT_MODEL_MAX_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("AICARE_AGENT_SPECIALIST_TIMEOUT_SECONDS", "45.5")
    monkeypatch.setenv("AICARE_AGENT_SPECIALIST_MAX_OUTPUT_TOKENS", "3072")
    monkeypatch.setenv("AICARE_AGENT_ANSWER_TIMEOUT_SECONDS", "50.5")
    monkeypatch.setenv("AICARE_AGENT_ANSWER_MAX_OUTPUT_TOKENS", "3584")
    monkeypatch.setenv(
        "AICARE_AGENT_POSTGRES_DSN",
        "postgresql://agent:password@localhost:5432/aicare_agent",
    )
    monkeypatch.setenv("AICARE_AGENT_CHECKPOINT_BACKEND", "postgres")
    monkeypatch.setenv("AICARE_AGENT_REDIS_URL", "redis://:password@localhost:6379/1")
    monkeypatch.setenv("AICARE_AGENT_RUN_LEASE_SECONDS", "45")
    monkeypatch.setenv("AICARE_AGENT_RUN_RETENTION_SECONDS", "1209600")
    monkeypatch.setenv("AICARE_AGENT_RUN_HEARTBEAT_SECONDS", "8")
    monkeypatch.setenv("AICARE_AGENT_RUN_TIMEOUT_SECONDS", "90")
    monkeypatch.setenv("AICARE_AGENT_REDIS_MAX_CONNECTIONS", "40")
    monkeypatch.setenv("AICARE_AGENT_REDIS_SOCKET_TIMEOUT_SECONDS", "3.5")
    monkeypatch.setenv("AICARE_AGENT_REDIS_HEALTH_CHECK_INTERVAL_SECONDS", "15")
    monkeypatch.setenv("LANGGRAPH_AES_KEY", "0123456789abcdef0123456789abcdef")
    monkeypatch.setenv("AICARE_AGENT_JAVA_BASE_URL", "http://localhost:8080")
    monkeypatch.setenv("AICARE_AGENT_JAVA_SERVICE_TOKEN", "java-secret")
    monkeypatch.setenv("AICARE_AGENT_ELASTICSEARCH_NODES", "http://localhost:9200")
    monkeypatch.setenv("AICARE_AGENT_RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "langsmith-secret")
    monkeypatch.setenv("LANGSMITH_PROJECT", "aicare-agent-test")

    settings = Settings(_env_file=None)

    assert settings.environment is Environment.TEST
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.log_level == "DEBUG"
    assert settings.output_max_chars == 6000
    assert settings.model_provider is ModelProviderName.DEEPSEEK
    assert settings.deepseek_model == "deepseek-v4-pro"
    assert str(settings.deepseek_base_url) == "https://deepseek.example.com/v1"
    assert settings.deepseek_max_retries == 4
    assert settings.model_timeout_seconds == 12.5
    assert settings.model_max_output_tokens == 1024
    assert settings.specialist_timeout_seconds == 45.5
    assert settings.specialist_max_output_tokens == 3072
    assert settings.answer_timeout_seconds == 50.5
    assert settings.answer_max_output_tokens == 3584
    assert settings.deepseek_api_key is not None
    assert settings.deepseek_api_key.get_secret_value() == "deepseek-secret"
    assert settings.agent_postgres_dsn is not None
    assert "aicare_agent" in settings.agent_postgres_dsn.get_secret_value()
    assert settings.checkpoint_backend is CheckpointBackend.POSTGRES
    assert settings.agent_redis_url is not None
    assert settings.agent_redis_url.get_secret_value().startswith("redis://")
    assert settings.run_lease_seconds == 45
    assert settings.run_retention_seconds == 1209600
    assert settings.run_heartbeat_seconds == 8
    assert settings.run_timeout_seconds == 90
    assert settings.redis_max_connections == 40
    assert settings.redis_socket_timeout_seconds == 3.5
    assert settings.redis_health_check_interval_seconds == 15
    assert settings.checkpoint_encryption_key is not None
    assert str(settings.java_base_url) == "http://localhost:8080/"
    assert settings.java_service_token is not None
    assert settings.java_service_token.get_secret_value() == "java-secret"
    assert tuple(str(node) for node in settings.elasticsearch_node_urls) == (
        "http://localhost:9200/",
    )
    assert settings.rabbitmq_url is not None
    assert settings.langsmith_tracing is True
    assert settings.langsmith_api_key is not None
    assert settings.langsmith_project == "aicare-agent-test"


@pytest.mark.parametrize("port", [0, 65536])
def test_port_must_be_in_tcp_range(port: int) -> None:
    with pytest.raises(ValidationError):
        Settings(port=port, _env_file=None)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("model_timeout_seconds", 0),
        ("model_timeout_seconds", -0.1),
        ("specialist_timeout_seconds", 0),
        ("answer_timeout_seconds", -1),
    ],
)
def test_model_timeouts_must_be_positive(field_name: str, invalid_value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field_name: invalid_value}, _env_file=None)


@pytest.mark.parametrize(
    "field_name",
    [
        "model_max_output_tokens",
        "specialist_max_output_tokens",
        "answer_max_output_tokens",
    ],
)
@pytest.mark.parametrize("invalid_value", [0, 8193])
def test_model_output_token_limits_must_be_in_supported_range(
    field_name: str,
    invalid_value: int,
) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field_name: invalid_value}, _env_file=None)


@pytest.mark.parametrize("invalid_value", [-1, 6])
def test_deepseek_retries_must_be_limited(invalid_value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(deepseek_max_retries=invalid_value, _env_file=None)


def test_unknown_model_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(model_provider="unknown-provider", _env_file=None)


def test_secret_values_are_not_exposed_by_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek")
    monkeypatch.setenv("AICARE_AGENT_POSTGRES_DSN", "postgresql://secret-agent-dsn")
    monkeypatch.setenv("AICARE_AGENT_REDIS_URL", "redis://:secret-redis@localhost:6379/1")
    monkeypatch.setenv("LANGGRAPH_AES_KEY", "secret-checkpoint-key")
    monkeypatch.setenv("AICARE_AGENT_JAVA_SERVICE_TOKEN", "secret-java-token")
    monkeypatch.setenv("AICARE_AGENT_RABBITMQ_URL", "amqp://secret-rabbitmq")
    monkeypatch.setenv("LANGSMITH_API_KEY", "secret-langsmith")

    rendered_settings = repr(Settings(_env_file=None))

    assert "secret-deepseek" not in rendered_settings
    assert "secret-agent-dsn" not in rendered_settings
    assert "secret-redis" not in rendered_settings
    assert "secret-checkpoint-key" not in rendered_settings
    assert "secret-java-token" not in rendered_settings
    assert "secret-rabbitmq" not in rendered_settings
    assert "secret-langsmith" not in rendered_settings


def test_production_requires_deepseek_postgres_redis_encryption_and_java_connections() -> None:
    settings = Settings(environment=Environment.PRODUCTION, _env_file=None)

    with pytest.raises(ValueError, match="生产环境缺少必需配置") as exc_info:
        validate_production_settings(settings)

    error_message = str(exc_info.value)
    assert "DEEPSEEK_API_KEY" in error_message
    assert "AICARE_AGENT_POSTGRES_DSN" in error_message
    assert "AICARE_AGENT_REDIS_URL" in error_message
    assert "LANGGRAPH_AES_KEY" in error_message
    assert "AICARE_AGENT_JAVA_BASE_URL" in error_message
    assert "AICARE_AGENT_JAVA_SERVICE_TOKEN" in error_message


def test_complete_production_settings_pass_validation(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    lock_path = tmp_path / "models.lock.json"
    lock_path.write_text("{}", encoding="utf-8")
    ca_path = tmp_path / "http_ca.crt"
    ca_path.write_text("ca", encoding="utf-8")
    settings = Settings(
        environment=Environment.PRODUCTION,
        deepseek_api_key="deepseek-secret",
        agent_postgres_dsn="postgresql://agent:password@postgres:5432/aicare_agent",
        checkpoint_backend="postgres",
        agent_redis_url="redis://:password@redis:6379/1",
        redis_mode="sentinel",
        redis_sentinels="sentinel-a:26379,sentinel-b:26380,sentinel-c:26381",
        redis_master_name="aicare-agent-master",
        redis_sentinel_username="aicare_sentinel",
        redis_sentinel_password="sentinel-secret",
        checkpoint_encryption_key="0123456789abcdef0123456789abcdef",
        java_base_url="http://platform-api:8080",
        java_service_token="java-secret",
        java_allow_private_http=True,
        rag_enabled=True,
        elasticsearch_nodes="https://elasticsearch:9200",
        elasticsearch_username="aicare_agent",
        elasticsearch_password="es-secret",
        elasticsearch_ca_cert_path=ca_path,
        rag_model_lock_path=lock_path,
        rag_model_dir=model_dir,
        bge_embedding_revision="a" * 40,
        bge_reranker_revision="b" * 40,
        rag_chunk_hmac_key="c" * 64,
        _env_file=None,
    )

    validate_production_settings(settings)


def test_production_rejects_private_http_without_explicit_deployment_switch() -> None:
    settings = Settings(
        environment=Environment.PRODUCTION,
        deepseek_api_key="deepseek-secret",
        agent_postgres_dsn="postgresql://agent:password@postgres:5432/aicare_agent",
        checkpoint_backend="postgres",
        agent_redis_url="redis://:password@redis:6379/1",
        redis_mode="sentinel",
        redis_sentinels="sentinel-a:26379,sentinel-b:26380,sentinel-c:26381",
        redis_master_name="aicare-agent-master",
        redis_sentinel_username="aicare_sentinel",
        redis_sentinel_password="sentinel-secret",
        checkpoint_encryption_key="0123456789abcdef0123456789abcdef",
        java_base_url="http://platform-api:8080",
        java_service_token="java-secret",
        _env_file=None,
    )

    with pytest.raises(ValueError, match="必须显式允许私网HTTP"):
        validate_production_settings(settings)


def test_production_rejects_public_http_even_when_private_switch_is_enabled() -> None:
    settings = Settings(
        environment=Environment.PRODUCTION,
        deepseek_api_key="deepseek-secret",
        agent_postgres_dsn="postgresql://agent:password@postgres:5432/aicare_agent",
        checkpoint_backend="postgres",
        agent_redis_url="redis://:password@redis:6379/1",
        redis_mode="sentinel",
        redis_sentinels="sentinel-a:26379,sentinel-b:26380,sentinel-c:26381",
        redis_master_name="aicare-agent-master",
        redis_sentinel_username="aicare_sentinel",
        redis_sentinel_password="sentinel-secret",
        checkpoint_encryption_key="0123456789abcdef0123456789abcdef",
        java_base_url="http://public.example.com:8080",
        java_service_token="java-secret",
        java_allow_private_http=True,
        _env_file=None,
    )

    with pytest.raises(ValueError, match="HTTP主机必须是私网地址"):
        validate_production_settings(settings)


def test_production_requires_sentinel_redis_discovery() -> None:
    settings = Settings(
        environment=Environment.PRODUCTION,
        deepseek_api_key="deepseek-secret",
        agent_postgres_dsn="postgresql://agent:password@postgres:5432/aicare_agent",
        checkpoint_backend="postgres",
        agent_redis_url="redis://:password@redis:6379/1",
        checkpoint_encryption_key="0123456789abcdef0123456789abcdef",
        java_base_url="http://platform-api:8080",
        java_service_token="java-secret",
        _env_file=None,
    )

    with pytest.raises(ValueError, match="生产环境必须使用Redis Sentinel"):
        validate_production_settings(settings)


def test_production_rejects_fake_model_provider() -> None:
    settings = Settings(
        environment=Environment.PRODUCTION,
        model_provider="fake",
        _env_file=None,
    )

    with pytest.raises(ValueError, match="生产环境禁止使用Fake模型"):
        validate_production_settings(settings)


def test_production_requires_https_deepseek_base_url() -> None:
    settings = Settings(
        environment=Environment.PRODUCTION,
        deepseek_api_key="deepseek-secret",
        deepseek_base_url="http://deepseek.example.com",
        agent_postgres_dsn="postgresql://agent:password@postgres:5432/aicare_agent",
        checkpoint_backend="postgres",
        agent_redis_url="redis://:password@redis:6379/1",
        checkpoint_encryption_key="0123456789abcdef0123456789abcdef",
        java_base_url="http://platform-api:8080",
        java_service_token="java-secret",
        _env_file=None,
    )

    with pytest.raises(ValueError, match="DeepSeek Base URL必须使用HTTPS"):
        validate_production_settings(settings)


def test_production_rejects_blank_secret_values() -> None:
    settings = Settings(
        environment=Environment.PRODUCTION,
        deepseek_api_key=" ",
        agent_postgres_dsn="",
        checkpoint_backend="memory",
        agent_redis_url=" ",
        checkpoint_encryption_key="\t",
        java_base_url="http://platform-api:8080",
        java_service_token="\t",
        _env_file=None,
    )

    with pytest.raises(ValueError) as exc_info:
        validate_production_settings(settings)

    error_message = str(exc_info.value)
    assert "DEEPSEEK_API_KEY" in error_message
    assert "AICARE_AGENT_POSTGRES_DSN" in error_message
    assert "AICARE_AGENT_REDIS_URL" in error_message
    assert "LANGGRAPH_AES_KEY" in error_message
    assert "AICARE_AGENT_JAVA_SERVICE_TOKEN" in error_message
    assert "AICARE_AGENT_JAVA_BASE_URL" not in error_message


def test_production_rejects_memory_checkpoint_backend() -> None:
    settings = Settings(
        environment=Environment.PRODUCTION,
        model_provider="deepseek",
        deepseek_api_key="deepseek-secret",
        checkpoint_backend="memory",
        agent_postgres_dsn="postgresql://agent:password@postgres:5432/aicare_agent",
        agent_redis_url="redis://:password@redis:6379/1",
        checkpoint_encryption_key="0123456789abcdef0123456789abcdef",
        java_base_url="http://platform-api:8080",
        java_service_token="java-secret",
        _env_file=None,
    )

    with pytest.raises(ValueError, match="生产环境必须使用PostgreSQL Checkpointer"):
        validate_production_settings(settings)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("run_lease_seconds", 4),
        ("run_lease_seconds", 301),
        ("run_retention_seconds", 3599),
        ("run_retention_seconds", 2592001),
        ("run_heartbeat_seconds", 0),
        ("run_timeout_seconds", 0),
        ("redis_max_connections", 0),
        ("redis_max_connections", 201),
        ("redis_socket_timeout_seconds", 0),
        ("redis_health_check_interval_seconds", -1),
    ],
)
def test_run_lifecycle_durations_have_safe_bounds(field_name: str, invalid_value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field_name: invalid_value}, _env_file=None)


def test_run_heartbeat_must_finish_before_redis_lease_expires() -> None:
    with pytest.raises(ValidationError, match="心跳间隔必须短于Redis租约"):
        Settings(run_lease_seconds=10, run_heartbeat_seconds=10, _env_file=None)


def test_get_settings_caches_until_explicitly_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AICARE_AGENT_PORT", "9001")
    first = get_settings()
    monkeypatch.setenv("AICARE_AGENT_PORT", "9002")

    assert get_settings() is first
    assert get_settings().port == 9001

    get_settings.cache_clear()

    assert get_settings().port == 9002


def test_input_security_and_routing_thresholds_have_safe_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.input_max_chars == 8000
    assert settings.route_direct_confidence == 0.8
    assert settings.route_clarify_confidence == 0.5


def test_input_security_and_routing_thresholds_support_environment_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AICARE_AGENT_INPUT_MAX_CHARS", "12000")
    monkeypatch.setenv("AICARE_AGENT_ROUTE_DIRECT_CONFIDENCE", "0.9")
    monkeypatch.setenv("AICARE_AGENT_ROUTE_CLARIFY_CONFIDENCE", "0.6")

    settings = Settings(_env_file=None)

    assert settings.input_max_chars == 12000
    assert settings.route_direct_confidence == 0.9
    assert settings.route_clarify_confidence == 0.6


@pytest.mark.parametrize("invalid_value", [255, 32769])
def test_input_max_chars_stays_within_supported_bounds(invalid_value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(input_max_chars=invalid_value, _env_file=None)


@pytest.mark.parametrize("invalid_value", [-0.01, 1.01])
def test_route_confidence_thresholds_stay_between_zero_and_one(
    invalid_value: float,
) -> None:
    with pytest.raises(ValidationError):
        Settings(route_direct_confidence=invalid_value, _env_file=None)

    with pytest.raises(ValidationError):
        Settings(route_clarify_confidence=invalid_value, _env_file=None)


@pytest.mark.parametrize(
    ("direct_confidence", "clarify_confidence"),
    [(0.5, 0.5), (0.4, 0.5)],
)
def test_direct_route_threshold_must_be_greater_than_clarify_threshold(
    direct_confidence: float,
    clarify_confidence: float,
) -> None:
    with pytest.raises(ValidationError, match="直接路由置信度必须高于澄清置信度"):
        Settings(
            route_direct_confidence=direct_confidence,
            route_clarify_confidence=clarify_confidence,
            _env_file=None,
        )
