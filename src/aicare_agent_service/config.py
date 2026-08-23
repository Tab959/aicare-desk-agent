"""集中声明、读取并验证 Agent 服务的全部运行配置。

Pydantic Settings按“显式构造参数→进程环境变量→当前目录.env→字段默认值”的方式获得值，
并转换成强类型对象。SecretStr保护密钥的普通显示，枚举/Field约束拒绝非法值，生产门禁
额外要求模型、PostgreSQL、Redis、AES和Java网关配置齐全。本文件不主动连接任何服务。
"""

# StrEnum让环境、Provider和后端选项既是受控枚举又可作为字符串使用。
from enum import StrEnum

# lru_cache缓存唯一Settings实例，避免每次请求重复读取环境和.env。
from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path

# Literal把日志级别限制在四个明确字符串中。
from typing import Literal

# AnyHttpUrl验证HTTP(S)地址；Field声明默认值/范围/环境别名；SecretStr隐藏敏感值；验证器处理跨字段规则。
from pydantic import AnyHttpUrl, Field, SecretStr, model_validator

# BaseSettings实现配置源读取；SettingsConfigDict配置.env和字段解析行为。
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Agent服务运行环境。"""

    # 单元/集成测试环境，可使用Fake模型和内存资源。
    TEST = "test"
    # 本地或开发部署环境，部分生产门禁只告警。
    DEVELOPMENT = "development"
    # 公网/正式环境，强制检查持久化、安全和外部连接配置。
    PRODUCTION = "production"


class ModelProviderName(StrEnum):
    """可由Agent服务选择的模型提供商。"""

    # 当前真实模型Provider，通过langchain-deepseek访问官方兼容API。
    DEEPSEEK = "deepseek"
    # 只允许测试环境使用的确定性脚本模型。
    FAKE = "fake"


class CheckpointBackend(StrEnum):
    """LangGraph Checkpointer后端；RunStore固定使用Redis。"""

    # 进程内Saver，重启丢失状态，只允许非生产调试。
    MEMORY = "memory"
    # 生产异步PostgreSQL Saver，checkpoint正文使用AES加密。
    POSTGRES = "postgres"


class RedisMode(StrEnum):
    """Redis RunStore连接拓扑。"""

    # 直接连接URL指定的单个Redis节点，保留给测试和开发环境。
    STANDALONE = "standalone"
    # 通过多个Sentinel发现当前可写master，并在故障转移后重新发现。
    SENTINEL = "sentinel"


class Settings(BaseSettings):
    """Agent服务配置，仅从构造参数、环境变量或本地.env文件读取。"""

    # model_config不是业务字段；它告诉Pydantic Settings如何查找和解析配置源。
    model_config = SettingsConfigDict(
        # 从启动命令当前工作目录读取.env；langgraph.json也显式指向同一文件。
        env_file=".env",
        # .env按UTF-8读取，允许中文注释。
        env_file_encoding="utf-8",
        # 空环境变量不覆盖字段默认值，便于尚未部署的可选能力在.env中保留空值。
        env_ignore_empty=True,
        # 忽略当前阶段尚未建模的额外env键，避免兼容配置阻断启动。
        extra="ignore",
        # 环境变量名称不区分大小写；项目仍统一使用大写名称。
        case_sensitive=False,
        # 允许按Python字段名或validation_alias填充值，方便测试直接构造Settings。
        populate_by_name=True,
    )

    # 服务运行环境；默认development。
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        validation_alias="AICARE_AGENT_ENVIRONMENT",
    )
    # 以下两个内部元数据当前没有环境变量别名，供健康检查和日志识别服务。
    service_name: str = "aicare-agent-service"
    service_version: str = "0.1.0"
    # FastAPI监听地址；默认只绑定本机，公网暴露需结合网关和安全配置。
    host: str = Field(default="127.0.0.1", validation_alias="AICARE_AGENT_HOST")
    # TCP端口限制为1到65535。
    port: int = Field(default=8090, ge=1, le=65535, validation_alias="AICARE_AGENT_PORT")
    # Literal拒绝任意日志级别字符串。
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        validation_alias="AICARE_AGENT_LOG_LEVEL",
    )
    # 单条用户输入允许进入安全预处理的最大字符数。
    input_max_chars: int = Field(
        default=8000,
        ge=256,
        le=32768,
        validation_alias="AICARE_AGENT_INPUT_MAX_CHARS",
    )
    # 三类最终终态向Java输出的用户可见文本总字符上限。
    output_max_chars: int = Field(
        default=12000,
        ge=256,
        le=32768,
        validation_alias="AICARE_AGENT_OUTPUT_MAX_CHARS",
    )
    # 达到该置信度的合法分类结果才能直接进入专业子图。
    route_direct_confidence: float = Field(
        default=0.8,
        ge=0,
        le=1,
        validation_alias="AICARE_AGENT_ROUTE_DIRECT_CONFIDENCE",
    )
    # 达到该置信度但未达到直接路由阈值时进入模板化澄清。
    route_clarify_confidence: float = Field(
        default=0.5,
        ge=0,
        le=1,
        validation_alias="AICARE_AGENT_ROUTE_CLARIFY_CONFIDENCE",
    )
    # ================================================================================
    # 模型Provider默认DeepSeek；Fake还会在工厂和生产门禁中二次限制。
    model_provider: ModelProviderName = Field(
        default=ModelProviderName.DEEPSEEK,
        validation_alias="AICARE_AGENT_MODEL_PROVIDER",
    )
    # 发送给DeepSeek API的模型ID；可配置为官方当前支持的v4-flash或v4-pro。
    deepseek_model: str = Field(
        default="deepseek-v4-pro",
        validation_alias="AICARE_AGENT_DEEPSEEK_MODEL",
    )
    # DeepSeek密钥可缺省以便无Key加载开发图；真正创建模型时再报稳定配置错误。
    deepseek_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="DEEPSEEK_API_KEY",
    )
    # DeepSeek OpenAI兼容API根地址；生产环境额外强制HTTPS。
    deepseek_base_url: AnyHttpUrl = Field(
        default=AnyHttpUrl("https://api.deepseek.com"),
        validation_alias="AICARE_AGENT_DEEPSEEK_BASE_URL",
    )
    # 网络/限流瞬时失败重试次数，限定0到5，避免无限重试。
    deepseek_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        validation_alias="AICARE_AGENT_DEEPSEEK_MAX_RETRIES",
    )
    # 路由、摘要、审核等通用低成本模型调用的超时秒数。
    model_timeout_seconds: float = Field(
        default=30,
        gt=0,
        validation_alias="AICARE_AGENT_MODEL_TIMEOUT_SECONDS",
    )
    # 通用用途最大输出token；当前项目上限限制8192以控制成本和延迟。
    # 这个值限制的是 单次模型调用 的输出，而非整个交互对话。如果一次交互涉及多次调用（如路由 + 审核 + 回复），每次调用各自独立受此限制。
    model_max_output_tokens: int = Field(
        default=2048,
        ge=1,
        le=8192,
        validation_alias="AICARE_AGENT_MODEL_MAX_OUTPUT_TOKENS",
    )
    # 售前/售后专业Agent的独立超时预算。
    specialist_timeout_seconds: float = Field(
        default=60,
        gt=0,
        validation_alias="AICARE_AGENT_SPECIALIST_TIMEOUT_SECONDS",
    )
    # 专业Agent独立输出token上限。
    specialist_max_output_tokens: int = Field(
        default=4096,
        ge=1,
        le=8192,
        validation_alias="AICARE_AGENT_SPECIALIST_MAX_OUTPUT_TOKENS",
    )
    # 最终自然语言回答的独立超时预算。
    answer_timeout_seconds: float = Field(
        default=60,
        gt=0,
        validation_alias="AICARE_AGENT_ANSWER_TIMEOUT_SECONDS",
    )
    # 最终回答独立输出token上限。
    answer_max_output_tokens: int = Field(
        default=4096,
        ge=1,
        le=8192,
        validation_alias="AICARE_AGENT_ANSWER_MAX_OUTPUT_TOKENS",
    )
    # ================================================================================
    # Agent专用Redis连接，用于run幂等、单飞、租约和清理guard。
    agent_redis_url: SecretStr | None = Field(
        default=None,
        validation_alias="AICARE_AGENT_REDIS_URL",
    )
    # Redis异步连接池上限，防止无界连接耗尽基础设施。
    redis_max_connections: int = Field(
        default=20,
        ge=1,
        le=200,
        validation_alias="AICARE_AGENT_REDIS_MAX_CONNECTIONS",
    )
    # Redis连接和命令socket超时秒数。
    redis_socket_timeout_seconds: float = Field(
        default=2,
        gt=0,
        le=30,
        validation_alias="AICARE_AGENT_REDIS_SOCKET_TIMEOUT_SECONDS",
    )
    # redis-py空闲连接健康检查间隔；0表示关闭周期检查。
    redis_health_check_interval_seconds: int = Field(
        default=30,
        ge=0,
        le=300,
        validation_alias="AICARE_AGENT_REDIS_HEALTH_CHECK_INTERVAL_SECONDS",
    )
    # Redis拓扑模式；生产部署使用Sentinel，开发和集成测试可保留standalone。
    redis_mode: RedisMode = Field(
        default=RedisMode.STANDALONE,
        validation_alias="AICARE_AGENT_REDIS_MODE",
    )
    # 逗号分隔Sentinel发现入口，格式为host:port；不在此字段保存密码。
    redis_sentinels: str | None = Field(
        default=None,
        validation_alias="AICARE_AGENT_REDIS_SENTINELS",
    )
    # Sentinel监控的逻辑master名称，不是容器名或主机名。
    redis_master_name: str | None = Field(
        default=None,
        min_length=1,
        validation_alias="AICARE_AGENT_REDIS_MASTER_NAME",
    )
    # Sentinel控制面使用独立ACL用户，不复用数据节点应用用户。
    redis_sentinel_username: str | None = Field(
        default=None,
        min_length=1,
        validation_alias="AICARE_AGENT_REDIS_SENTINEL_USERNAME",
    )
    # Sentinel控制面ACL密码使用SecretStr隐藏普通显示。
    redis_sentinel_password: SecretStr | None = Field(
        default=None,
        validation_alias="AICARE_AGENT_REDIS_SENTINEL_PASSWORD",
    )
    # conversation租约秒数，防止同一thread并发执行多个run。
    # 这里的"thread"指的是 对话线程（conversation thread），一个 thread 就是一个多轮对话会话。
    run_lease_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        validation_alias="AICARE_AGENT_RUN_LEASE_SECONDS",
    )
    # run ledger终态/恢复元数据保留期，默认7天。
    run_retention_seconds: int = Field(
        default=604800,
        ge=3600,
        le=2592000,
        validation_alias="AICARE_AGENT_RUN_RETENTION_SECONDS",
    )
    # 执行中续租并发送heartbeat的间隔，必须短于租约。
    # 每 10 秒发心跳，租约被刷新，一直持有锁
    run_heartbeat_seconds: float = Field(
        default=10,
        gt=0,
        le=60,
        validation_alias="AICARE_AGENT_RUN_HEARTBEAT_SECONDS",
    )
    # 单次Agent图执行总超时，包含模型和节点工作。
    run_timeout_seconds: float = Field(
        default=120,
        gt=0,
        le=600,
        validation_alias="AICARE_AGENT_RUN_TIMEOUT_SECONDS",
    )

    # ================================================================================
    # Agent专用PostgreSQL DSN，用于LangGraph checkpoint，不访问Java业务MySQL。
    agent_postgres_dsn: SecretStr | None = Field(
        default=None,
        validation_alias="AICARE_AGENT_POSTGRES_DSN",
    )
    # Checkpointer后端默认memory；生产必须显式切换postgres。
    checkpoint_backend: CheckpointBackend = Field(
        default=CheckpointBackend.MEMORY,
        validation_alias="AICARE_AGENT_CHECKPOINT_BACKEND",
    )
    # LangGraph checkpoint保留期，默认7天，最长90天。
    checkpoint_retention_seconds: int = Field(
        default=604800,
        ge=3600,
        le=7776000,
        validation_alias="AICARE_AGENT_CHECKPOINT_RETENTION_SECONDS",
    )
    # 每次清理最多处理的thread候选数。
    checkpoint_cleanup_batch_size: int = Field(
        default=500,
        ge=1,
        le=10000,
        validation_alias="AICARE_AGENT_CHECKPOINT_CLEANUP_BATCH_SIZE",
    )
    # 删除checkpoint期间阻止新run启动的Redis guard秒数。
    checkpoint_cleanup_guard_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        validation_alias="AICARE_AGENT_CHECKPOINT_CLEANUP_GUARD_SECONDS",
    )
    # AES checkpoint密钥，必须为16/24/32个UTF-8字节；不进入日志。
    checkpoint_encryption_key: SecretStr | None = Field(
        default=None,
        validation_alias="LANGGRAPH_AES_KEY",
    )
    # ================================================================================
    # Java内部Agent Gateway根地址；Python只经受控接口查询业务事实和返回建议。
    java_base_url: AnyHttpUrl | None = Field(
        default=None,
        validation_alias="AICARE_AGENT_JAVA_BASE_URL",
    )
    # Java ↔ Python内部服务认证token。
    java_service_token: SecretStr | None = Field(
        default=None,
        validation_alias="AICARE_AGENT_JAVA_SERVICE_TOKEN",
    )
    # 私网明文HTTP必须由部署者显式授权；公网地址即使打开本开关也会被拒绝。
    java_allow_private_http: bool = Field(
        default=False,
        validation_alias="AICARE_AGENT_JAVA_ALLOW_PRIVATE_HTTP",
    )
    # Java工具客户端连接池和四阶段超时均由部署配置统一限制。
    java_max_connections: int = Field(
        default=50, ge=1, le=500, validation_alias="AICARE_AGENT_JAVA_MAX_CONNECTIONS"
    )
    java_max_keepalive_connections: int = Field(
        default=20,
        ge=1,
        le=200,
        validation_alias="AICARE_AGENT_JAVA_MAX_KEEPALIVE_CONNECTIONS",
    )
    java_connect_timeout_seconds: float = Field(
        default=2, gt=0, le=30, validation_alias="AICARE_AGENT_JAVA_CONNECT_TIMEOUT_SECONDS"
    )
    java_read_timeout_seconds: float = Field(
        default=8, gt=0, le=60, validation_alias="AICARE_AGENT_JAVA_READ_TIMEOUT_SECONDS"
    )
    java_write_timeout_seconds: float = Field(
        default=5, gt=0, le=60, validation_alias="AICARE_AGENT_JAVA_WRITE_TIMEOUT_SECONDS"
    )
    java_pool_timeout_seconds: float = Field(
        default=2, gt=0, le=30, validation_alias="AICARE_AGENT_JAVA_POOL_TIMEOUT_SECONDS"
    )
    java_retry_after_max_seconds: float = Field(
        default=1, ge=0, le=5, validation_alias="AICARE_AGENT_JAVA_RETRY_AFTER_MAX_SECONDS"
    )
    java_response_max_bytes: int = Field(
        default=65_536,
        ge=1024,
        le=1_048_576,
        validation_alias="AICARE_AGENT_JAVA_RESPONSE_MAX_BYTES",
    )
    # ================================================================================
    # 是否启用生产RAG资源；生产门禁要求显式启用并完整配置。
    rag_enabled: bool = Field(default=False, validation_alias="AICARE_AGENT_RAG_ENABLED")
    # 逗号分隔的Elasticsearch节点；解析属性提供多节点URL元组。
    elasticsearch_nodes: str | None = Field(
        default=None,
        validation_alias="AICARE_AGENT_ELASTICSEARCH_NODES",
    )
    # Elasticsearch最小权限应用用户，不使用elastic超级用户。
    elasticsearch_username: str | None = Field(
        default=None,
        min_length=1,
        validation_alias="AICARE_AGENT_ELASTICSEARCH_USERNAME",
    )
    # Elasticsearch应用用户密码。
    elasticsearch_password: SecretStr | None = Field(
        default=None,
        validation_alias="AICARE_AGENT_ELASTICSEARCH_PASSWORD",
    )
    # 签发HTTP节点证书的CA文件；支持证书轮换时替换并重启进程。
    elasticsearch_ca_cert_path: Path | None = Field(
        default=None,
        validation_alias="AICARE_AGENT_ELASTICSEARCH_CA_CERT_PATH",
    )
    # 证书校验是固定安全门禁，配置为false会被模型验证器拒绝。
    elasticsearch_verify_certs: bool = Field(
        default=True,
        validation_alias="AICARE_AGENT_ELASTICSEARCH_VERIFY_CERTS",
    )
    # 所有知识物理索引和别名的受控前缀。
    elasticsearch_index_prefix: str = Field(
        default="aicare-kb",
        pattern=r"^[a-z][a-z0-9-]{2,31}$",
        validation_alias="AICARE_AGENT_ELASTICSEARCH_INDEX_PREFIX",
    )
    # ES异步连接池与网络超时预算。
    elasticsearch_connections_per_node: int = Field(
        default=10,
        ge=1,
        le=100,
        validation_alias="AICARE_AGENT_ELASTICSEARCH_CONNECTIONS_PER_NODE",
    )
    elasticsearch_request_timeout_seconds: float = Field(
        default=5,
        gt=0,
        le=60,
        validation_alias="AICARE_AGENT_ELASTICSEARCH_REQUEST_TIMEOUT_SECONDS",
    )
    # 模型锁、离线模型根目录和两个精确Git revision。
    rag_model_lock_path: Path | None = Field(
        default=None,
        validation_alias="AICARE_AGENT_RAG_MODEL_LOCK_PATH",
    )
    rag_model_dir: Path | None = Field(
        default=None,
        validation_alias="AICARE_AGENT_RAG_MODEL_DIR",
    )
    bge_embedding_revision: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{40}$",
        validation_alias="AICARE_AGENT_BGE_EMBEDDING_REVISION",
    )
    bge_reranker_revision: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{40}$",
        validation_alias="AICARE_AGENT_BGE_RERANKER_REVISION",
    )
    # Chunk ID使用独立HMAC根密钥派生租户密钥，不复用Checkpoint或服务认证密钥。
    rag_chunk_hmac_key: SecretStr | None = Field(
        default=None,
        min_length=32,
        validation_alias="AICARE_AGENT_RAG_CHUNK_HMAC_KEY",
    )
    # CPU部署固定FP32；后续如启用GPU需新增显式受测模式。
    rag_model_device: Literal["cpu"] = Field(
        default="cpu", validation_alias="AICARE_AGENT_RAG_MODEL_DEVICE"
    )
    rag_model_use_fp16: bool = Field(
        default=False, validation_alias="AICARE_AGENT_RAG_MODEL_USE_FP16"
    )
    # 生产进程禁止运行时下载，模型只允许由初始化命令预置。
    rag_allow_runtime_model_download: bool = Field(
        default=False,
        validation_alias="AICARE_AGENT_RAG_ALLOW_RUNTIME_MODEL_DOWNLOAD",
    )
    # Embedding、重排批量、共享推理并发和单次绝对deadline预算。
    rag_embedding_batch_size: int = Field(
        default=8, ge=1, le=64, validation_alias="AICARE_AGENT_RAG_EMBEDDING_BATCH_SIZE"
    )
    rag_reranker_batch_size: int = Field(
        default=8, ge=1, le=64, validation_alias="AICARE_AGENT_RAG_RERANKER_BATCH_SIZE"
    )
    rag_model_max_concurrency: int = Field(
        default=2, ge=1, le=16, validation_alias="AICARE_AGENT_RAG_MODEL_MAX_CONCURRENCY"
    )
    rag_model_deadline_seconds: float = Field(
        default=30,
        gt=0,
        le=300,
        validation_alias="AICARE_AGENT_RAG_MODEL_DEADLINE_SECONDS",
    )
    # ================================================================================
    # 预留知识事件RabbitMQ连接，原始文档和大正文不通过消息传输。
    rabbitmq_url: SecretStr | None = Field(
        default=None,
        validation_alias="AICARE_AGENT_RABBITMQ_URL",
    )
    # ================================================================================
    # 是否启用LangSmith追踪；没有Key时应关闭。
    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    # LangSmith API密钥，SecretStr防止普通显示泄漏。
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="LANGSMITH_API_KEY",
    )
    # LangSmith项目名，用于把开发、测试和生产Trace分组隔离。
    langsmith_project: str = Field(
        default="aicare-agent-service-dev",
        validation_alias="LANGSMITH_PROJECT",
    )

    # ================================================================================
    # after验证器在全部字段解析后检查心跳和租约两个配置的关系。
    @model_validator(mode="after")
    def require_heartbeat_before_lease_expiry(self) -> "Settings":
        """确保至少能在租约到期前尝试一次心跳，并返回已验证Settings。"""
        # 大于或等于租约会导致正常执行者尚未续租就失去所有权。
        if self.run_heartbeat_seconds >= self.run_lease_seconds:
            raise ValueError("Agent run心跳间隔必须短于Redis租约")
        # after验证器必须返回self。
        return self

    @model_validator(mode="after")
    def require_ordered_route_confidence_thresholds(self) -> "Settings":
        """保证直接路由阈值严格高于澄清阈值。"""
        # 1、拒绝重叠或倒置区间，确保每个置信度只有一个确定分支。
        if self.route_direct_confidence <= self.route_clarify_confidence:
            raise ValueError("直接路由置信度必须高于澄清置信度")
        # 2、返回当前实例，让Pydantic继续完成Settings构造。
        return self

    # after验证器确保选择Sentinel后发现、名称和控制面ACL配置同时完整。
    @model_validator(mode="after")
    def require_complete_sentinel_configuration(self) -> "Settings":
        """拒绝无法发现master或无法认证Sentinel的半配置状态。"""
        if self.redis_mode is not RedisMode.SENTINEL:
            return self
        required = (
            self.redis_sentinels,
            self.redis_master_name,
            self.redis_sentinel_username,
            self.redis_sentinel_password,
        )
        if any(_is_missing(value) for value in required):
            raise ValueError("Sentinel模式缺少必需配置")
        # 访问属性会完整解析每个端点，并把格式错误转为Pydantic ValidationError。
        endpoints = self.redis_sentinel_endpoints
        if not endpoints:
            raise ValueError("Sentinel模式缺少发现端点")
        return self

    @model_validator(mode="after")
    def validate_elasticsearch_transport(self) -> "Settings":
        """拒绝公网明文节点、URL凭据和格式不完整的节点列表。"""
        # 1、未配置节点时交给生产必填门禁处理。
        if self.elasticsearch_nodes is None:
            return self
        # 2、解析每个节点并拒绝把认证信息写进URL。
        for node in self.elasticsearch_node_urls:
            if node.username is not None or node.password is not None:
                raise ValueError("Elasticsearch节点URL禁止内嵌凭据")
            if node.scheme == "http" and not _is_private_host(node.host or ""):
                raise ValueError("公网Elasticsearch节点必须使用HTTPS")
        return self

    @model_validator(mode="after")
    def require_offline_verified_fp32_rag(self) -> "Settings":
        """固定生产RAG为证书校验、离线模型和CPU FP32模式。"""
        # 1、ES证书验证不能通过环境变量关闭。
        if not self.elasticsearch_verify_certs:
            raise ValueError("Elasticsearch必须校验证书")
        # 2、服务生命周期不能下载模型，防止revision和供应链漂移。
        if self.rag_allow_runtime_model_download:
            raise ValueError("RAG禁止运行时下载模型")
        # 3、当前CPU部署只允许FP32，后续其他设备需独立评估和测试。
        if self.rag_model_use_fp16:
            raise ValueError("CPU RAG模型必须使用FP32")
        return self

    @property
    def elasticsearch_node_urls(self) -> tuple[AnyHttpUrl, ...]:
        """把逗号分隔节点转换为非空、去重的强类型URL元组。"""
        # 1、空配置返回空元组，生产必填检查负责阻断。
        if self.elasticsearch_nodes is None:
            return ()
        # 2、逐项解析并保持部署顺序，重复节点视为配置错误。
        raw_nodes = [item.strip() for item in self.elasticsearch_nodes.split(",")]
        if not raw_nodes or any(not item for item in raw_nodes):
            raise ValueError("Elasticsearch节点列表格式错误")
        try:
            nodes = tuple(AnyHttpUrl(item) for item in raw_nodes)
        except ValueError as exc:
            raise ValueError("Elasticsearch节点列表格式错误") from exc
        if len(nodes) != len(set(map(str, nodes))):
            raise ValueError("Elasticsearch节点不能重复")
        return nodes

    @property
    def redis_sentinel_endpoints(self) -> tuple[tuple[str, int], ...]:
        """把逗号分隔的Sentinel入口转换为redis-py需要的(host, port)元组。"""
        if self.redis_sentinels is None:
            return ()
        endpoints: list[tuple[str, int]] = []
        for raw_endpoint in self.redis_sentinels.split(","):
            endpoint = raw_endpoint.strip()
            if not endpoint or endpoint.count(":") != 1:
                raise ValueError("Sentinel端点必须使用host:port格式")
            host, raw_port = endpoint.rsplit(":", 1)
            if not host.strip():
                raise ValueError("Sentinel主机不能为空")
            try:
                port = int(raw_port)
            except ValueError as exc:
                raise ValueError("Sentinel端口必须是整数") from exc
            if not 1 <= port <= 65535:
                raise ValueError("Sentinel端口必须在1到65535之间")
            endpoints.append((host.strip(), port))
        return tuple(endpoints)


def validate_production_settings(settings: Settings) -> None:
    """在生产环境启动前校验当前阶段必需的外部连接配置。

    参数为已完成字段校验的Settings；非生产立即返回。生产缺配置或使用不安全后端时
    抛ValueError，错误只列环境变量名称，不输出其值。
    """
    # 开发和测试不强制完整基础设施，便于离线单测和局部调试。
    if settings.environment is not Environment.PRODUCTION:
        return

    # Fake模型只能提供预制测试响应，生产禁止启用。
    if settings.model_provider is ModelProviderName.FAKE:
        raise ValueError("生产环境禁止使用Fake模型")

    # 元组逐项保存对外配置名和已经解析的值，便于生成缺失清单。
    required_values = (
        ("DEEPSEEK_API_KEY", settings.deepseek_api_key),
        ("AICARE_AGENT_POSTGRES_DSN", settings.agent_postgres_dsn),
        ("AICARE_AGENT_REDIS_URL", settings.agent_redis_url),
        ("LANGGRAPH_AES_KEY", settings.checkpoint_encryption_key),
        ("AICARE_AGENT_JAVA_BASE_URL", settings.java_base_url),
        ("AICARE_AGENT_JAVA_SERVICE_TOKEN", settings.java_service_token),
    )
    # 列表推导只保留缺失变量名称，不包含任何SecretStr明文。
    missing_variables = [name for name, value in required_values if _is_missing(value)]

    # 有任一缺失项就用顿号连接名称并阻断启动。
    if missing_variables:
        joined_variables = "、".join(missing_variables)
        raise ValueError(f"生产环境缺少必需配置：{joined_variables}")

    # 生产必须用持久化PostgreSQL Checkpointer，不能在重启后丢状态。
    if settings.checkpoint_backend is not CheckpointBackend.POSTGRES:
        raise ValueError("生产环境必须使用PostgreSQL Checkpointer")

    # 对外模型API在生产必须TLS加密传输。
    if settings.deepseek_base_url.scheme != "https":
        raise ValueError("生产环境DeepSeek Base URL必须使用HTTPS")

    # production必须通过Sentinel发现当前master，禁止固定地址形成切主后的连接单点。
    if settings.redis_mode is not RedisMode.SENTINEL:
        raise ValueError("生产环境必须使用Redis Sentinel")

    # Java内部链路默认要求HTTPS；仅显式授权的私网地址允许HTTP。
    validate_java_transport(settings)

    # 核心生产门禁通过后再检查RAG，避免掩盖更基础的连接或安全错误。
    rag_required_values = (
        ("AICARE_AGENT_RAG_ENABLED", settings.rag_enabled or None),
        ("AICARE_AGENT_ELASTICSEARCH_NODES", settings.elasticsearch_nodes),
        ("AICARE_AGENT_ELASTICSEARCH_USERNAME", settings.elasticsearch_username),
        ("AICARE_AGENT_ELASTICSEARCH_PASSWORD", settings.elasticsearch_password),
        ("AICARE_AGENT_ELASTICSEARCH_CA_CERT_PATH", settings.elasticsearch_ca_cert_path),
        ("AICARE_AGENT_RAG_MODEL_LOCK_PATH", settings.rag_model_lock_path),
        ("AICARE_AGENT_RAG_MODEL_DIR", settings.rag_model_dir),
        ("AICARE_AGENT_BGE_EMBEDDING_REVISION", settings.bge_embedding_revision),
        ("AICARE_AGENT_BGE_RERANKER_REVISION", settings.bge_reranker_revision),
        ("AICARE_AGENT_RAG_CHUNK_HMAC_KEY", settings.rag_chunk_hmac_key),
    )
    missing_rag_variables = [name for name, value in rag_required_values if _is_missing(value)]
    if missing_rag_variables:
        raise ValueError("生产环境缺少必需配置：" + "、".join(missing_rag_variables))

    # RAG产物必须在进程创建任何ES或模型资源之前真实存在。
    assert settings.elasticsearch_ca_cert_path is not None
    assert settings.rag_model_lock_path is not None
    assert settings.rag_model_dir is not None
    if not settings.elasticsearch_ca_cert_path.is_file():
        raise ValueError("Elasticsearch CA文件不存在")
    if not settings.rag_model_lock_path.is_file():
        raise ValueError("RAG模型锁文件不存在")
    if not settings.rag_model_dir.is_dir():
        raise ValueError("RAG模型目录不存在")


def validate_java_transport(settings: Settings) -> None:
    """校验Java Gateway传输协议，拒绝公网明文HTTP和未授权私网HTTP。"""
    # 1、未配置地址由生产必填检查负责；HTTPS无需额外授权。
    if settings.java_base_url is None or settings.java_base_url.scheme == "https":
        return
    # 2、HTTP只有打开独立部署开关后才继续检查主机范围。
    if not settings.java_allow_private_http:
        raise ValueError("Java Base URL使用HTTP时必须显式允许私网HTTP")
    # 3、允许回环、私网IP、.local和无点容器服务名，公网主机始终阻断。
    host = settings.java_base_url.host or ""
    if host == "localhost" or host.endswith(".local") or "." not in host:
        return
    try:
        address = ip_address(host)
    except ValueError as exc:
        raise ValueError("Java Base URL的HTTP主机必须是私网地址") from exc
    if not (address.is_private or address.is_loopback or address.is_link_local):
        raise ValueError("Java Base URL的HTTP主机必须是私网地址")


def _is_missing(value: object) -> bool:
    """统一判断普通可选值或SecretStr是否未配置/全空白。"""
    # None直接表示缺失。
    if value is None:
        return True
    # SecretStr必须显式取内部值再strip判断，不能用显示掩码判断。
    if isinstance(value, SecretStr):
        return not value.get_secret_value().strip()
    # 非None、非SecretStr对象视为存在；其类型/格式此前已由Pydantic验证。
    return False


def _is_private_host(host: str) -> bool:
    """判断主机是否为回环、私网IP或内部DNS名称。"""
    # 1、容器短名、localhost和.local后缀明确属于内部部署名称。
    if host == "localhost" or host.endswith(".local") or "." not in host:
        return True
    # 2、带点域名只有解析为私网IP时才允许明文HTTP。
    try:
        address = ip_address(host)
    except ValueError:
        return host.endswith(".internal")
    return address.is_private or address.is_loopback or address.is_link_local


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程级只读配置快照；测试和配置重载需显式清理缓存。

    ``@lru_cache(maxsize=1)``使首次调用创建Settings，后续复用同一实例。修改.env后必须
    重启进程，或在测试中调用``get_settings.cache_clear()``，否则仍会看到旧配置。
    """
    # 无参数构造触发Pydantic Settings按配置源优先级读取并验证全部字段。
    return Settings()
