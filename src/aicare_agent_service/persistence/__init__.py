"""Agent run 协调与 LangGraph 持久化能力的统一公开入口。

调用者可从本包导入checkpoint资源、Redis RunStore、生命周期编排、清理服务和稳定模型，
无需依赖内部文件布局。本模块只重导出名称，导入时不会连接Redis/PostgreSQL或执行清理。
"""

# 导出checkpoint候选、统计、服务、PostgreSQL目录和时间UUID解析函数。
from aicare_agent_service.persistence.checkpoint_cleanup import (
    CheckpointCandidate,
    CheckpointCleanupResult,
    CheckpointCleanupService,
    PostgresCheckpointCatalog,
    checkpoint_timestamp,
)

# 导出严格加密序列化器工厂和Saver异步资源上下文。
from aicare_agent_service.persistence.checkpointer import (
    build_checkpoint_serializer,
    checkpointer_resource,
)

# 导出Java conversationId→thread_id配置和规范请求摘要工具。
from aicare_agent_service.persistence.identity import (
    build_thread_config,
    canonical_request_digest,
)

# 导出Agent run执行、稳定异常、结果以及终态摘要函数。
from aicare_agent_service.persistence.lifecycle_runner import (
    AgentRunLifecycle,
    AgentRunLifecycleError,
    AgentRunLifecycleResult,
    RunAlreadyInProgressError,
    RunReplayUnavailableError,
    RunRequestConflictError,
    terminal_event_digest,
)

# 导出Redis ledger的状态、判定、记录和稳定存储异常。
from aicare_agent_service.persistence.models import (
    InvalidRunTransitionError,
    RunBeginOutcome,
    RunBeginResult,
    RunLeaseLostError,
    RunRecord,
    RunStatus,
    RunStoreError,
    RunStoreUnavailableError,
)

# RedisRunStore是具体生产实现，build_redis_client创建有界异步连接池。
from aicare_agent_service.persistence.redis_run_store import RedisRunStore, build_redis_client

# RunStore是生命周期编排依赖的抽象Protocol。
from aicare_agent_service.persistence.run_store import RunStore

# ``__all__``列出本包承诺稳定公开的名称；未列出的内部Lua等实现不属于公共API。
__all__ = [
    "AgentRunLifecycle",
    "AgentRunLifecycleError",
    "AgentRunLifecycleResult",
    "CheckpointCandidate",
    "CheckpointCleanupResult",
    "CheckpointCleanupService",
    "InvalidRunTransitionError",
    "PostgresCheckpointCatalog",
    "RedisRunStore",
    "RunAlreadyInProgressError",
    "RunBeginOutcome",
    "RunBeginResult",
    "RunLeaseLostError",
    "RunRecord",
    "RunReplayUnavailableError",
    "RunRequestConflictError",
    "RunStatus",
    "RunStore",
    "RunStoreError",
    "RunStoreUnavailableError",
    "build_checkpoint_serializer",
    "build_redis_client",
    "build_thread_config",
    "canonical_request_digest",
    "checkpoint_timestamp",
    "checkpointer_resource",
    "terminal_event_digest",
]
