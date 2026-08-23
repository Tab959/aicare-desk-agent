"""声明 Agent run 幂等、单飞和终态存储的异步协议。

生命周期编排器依赖这个Protocol而不是Redis具体实现，测试可注入内存Fake。该接口保存
run元数据与租约，不保存LangGraph会话状态；会话状态由PostgreSQL Checkpointer负责。
"""

# Protocol按方法形状定义接口，不要求实现类显式继承。
from typing import Protocol

# 入站run契约用于开始执行；结果和记录模型用于读取状态。
from aicare_agent_service.contracts.agent_run import AgentRunRequest
from aicare_agent_service.persistence.models import RunBeginResult, RunRecord


class RunStore(Protocol):
    """Agent run幂等、单飞租约和终态元数据存储接口。"""

    async def begin(self, run: AgentRunRequest, request_digest: str) -> RunBeginResult:
        """原子判定开始、恢复、进行中、重放或冲突，并可能返回租约。"""
        ...

    async def get(self, run_id: str) -> RunRecord | None:
        """按run ID读取安全元数据；不存在时返回None。"""
        ...

    async def renew_lease(self, run_id: str, lease_token: str) -> None:
        """仅允许当前token持有者续租。"""
        ...

    async def request_cancel(self, run_id: str) -> None:
        """记录外部取消意图，不直接提交CANCELLED终态。"""
        ...

    async def is_cancel_requested(self, run_id: str) -> bool:
        """查询run是否已有取消意图。"""
        ...

    async def is_conversation_active(self, conversation_id: str) -> bool:
        """供checkpoint清理器判断会话是否仍有活跃run或租约。"""
        ...

    async def acquire_cleanup_guard(self, conversation_id: str) -> str | None:
        """获取短时清理保护token；活跃或已被保护时返回None。"""
        ...

    async def release_cleanup_guard(self, conversation_id: str, token: str) -> None:
        """仅用匹配token释放会话清理保护。"""
        ...

    async def complete(
        self,
        run_id: str,
        lease_token: str,
        checkpoint_id: str,
        final_digest: str,
    ) -> None:
        """由当前租约持有者提交checkpoint ID和终态摘要，进入COMPLETED。"""
        # Ellipsis表示Protocol只声明方法，具体Redis实现负责Lua原子迁移。
        ...

    async def fail(self, run_id: str, lease_token: str, error_code: str) -> None:
        """以稳定错误码提交FAILED终态。"""
        ...

    async def cancel(self, run_id: str, lease_token: str) -> None:
        """由当前租约持有者提交CANCELLED终态。"""
        ...
