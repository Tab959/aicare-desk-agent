"""管理 BGE 模型、异步 Elasticsearch 客户端和 FastAPI RAG 资源生命周期。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

from elasticsearch import AsyncElasticsearch
from fastapi import FastAPI

from aicare_agent_service.config import Settings
from aicare_agent_service.rag.embeddings import BgeM3EmbeddingProvider
from aicare_agent_service.rag.model_lock import ModelLockError, verify_model_lock
from aicare_agent_service.rag.reranker import BgeReranker

T = TypeVar("T")


class RagStartupError(RuntimeError):
    """表示生产 RAG 依赖未就绪；消息只使用稳定错误码。"""


class BgeModelRuntime:
    """持有一个进程的 Embedding/Reranker 及其有界 CPU 推理执行器。"""

    def __init__(
        self,
        *,
        embedding_model: Any,
        reranker_model: Any,
        max_concurrency: int,
        deadline_seconds: float,
    ) -> None:
        """绑定两个已验证模型和共享推理预算。"""
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self._deadline_seconds = deadline_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="aicare-bge",
        )
        self._closed = False

    async def run(self, operation: Callable[[], T], *, deadline: float | None = None) -> T:
        """在有界执行器执行同步 CPU 推理，并以绝对超时丢弃迟到结果。"""
        # 1、关闭后的运行时拒绝新任务，防止使用已释放执行器。
        if self._closed:
            raise RuntimeError("RAG_MODEL_RUNTIME_CLOSED")
        # 2、配置上限与调用方绝对deadline取更早者，覆盖排队和执行全部时间。
        loop = asyncio.get_running_loop()
        effective_deadline = loop.time() + self._deadline_seconds
        if deadline is not None:
            effective_deadline = min(effective_deadline, deadline)
        async with asyncio.timeout_at(effective_deadline):
            async with self._semaphore:
                return await loop.run_in_executor(self._executor, operation)

    async def warm_up(self) -> None:
        """分别执行无业务正文的 Embedding 与重排热身。"""
        # 1、用固定无敏感短文本触发Embedding权重与tokenizer初始化。
        await self.run(
            lambda: self.embedding_model.encode(
                ["warmup"], batch_size=1, max_length=8, return_dense=True
            )
        )
        # 2、用固定短文本触发Reranker初始化，不记录输入或输出。
        await self.run(
            lambda: self.reranker_model.compute_score([["warmup", "warmup"]], normalize=True)
        )

    async def close(self) -> None:
        """停止接收新推理并释放工作线程和模型引用。"""
        # 1、先标记关闭，阻止关闭过程进入新的推理任务。
        self._closed = True
        # 2、不等待不可取消的第三方CPU任务，避免ASGI关闭无限阻塞。
        self._executor.shutdown(wait=False, cancel_futures=True)
        # 3、释放大模型引用，使Python/PyTorch可以回收内存。
        self.embedding_model = None
        self.reranker_model = None


@dataclass(frozen=True, slots=True)
class RagRuntimeResources:
    """应用级 RAG 资源：ES、唯一BGE运行时和两个生产适配器。"""

    elasticsearch: AsyncElasticsearch
    models: BgeModelRuntime
    embeddings: BgeM3EmbeddingProvider
    reranker: BgeReranker


def _load_bge_models(model_paths: dict[str, Path]) -> tuple[Any, Any]:
    """从已校验本地目录以 CPU FP32 和离线模式加载两个 BGE 模型。"""
    # 1、延迟导入重量级依赖，禁用RAG的进程不承担PyTorch导入成本。
    from FlagEmbedding import BGEM3FlagModel, FlagReranker  # type: ignore[import-untyped]

    # 2、local_files_only禁止第三方模型类在生产启动时访问Hugging Face。
    embedding_model = BGEM3FlagModel(
        str(model_paths["embedding"]),
        devices="cpu",
        use_fp16=False,
        trust_remote_code=False,
        local_files_only=True,
    )
    reranker_model = FlagReranker(
        str(model_paths["reranker"]),
        devices="cpu",
        use_fp16=False,
        trust_remote_code=False,
        local_files_only=True,
    )
    return embedding_model, reranker_model


async def create_rag_resources(settings: Settings) -> RagRuntimeResources:
    """校验离线模型，连接真实ES，加载并热身唯一模型运行时。"""
    # 1、先验证所有本地产物，任何缺失都不允许触发在线下载。
    assert settings.rag_model_lock_path is not None
    assert settings.rag_model_dir is not None
    assert settings.bge_embedding_revision is not None
    assert settings.bge_reranker_revision is not None
    assert settings.elasticsearch_username is not None
    assert settings.elasticsearch_password is not None
    assert settings.elasticsearch_ca_cert_path is not None
    runtime: BgeModelRuntime | None = None
    try:
        model_paths = verify_model_lock(
            lock_path=settings.rag_model_lock_path,
            model_root=settings.rag_model_dir,
            expected_revisions={
                "embedding": settings.bge_embedding_revision,
                "reranker": settings.bge_reranker_revision,
            },
        )
    except ModelLockError as exc:
        raise RagStartupError(exc.code) from exc
    if set(model_paths) != {"embedding", "reranker"}:
        raise RagStartupError("RAG_MODEL_LOCK_INCOMPLETE")

    # 2、创建TLS校验和最小权限认证的异步ES客户端并执行readiness探测。
    elasticsearch = AsyncElasticsearch(
        hosts=[str(node) for node in settings.elasticsearch_node_urls],
        basic_auth=(
            settings.elasticsearch_username,
            settings.elasticsearch_password.get_secret_value(),
        ),
        ca_certs=str(settings.elasticsearch_ca_cert_path),
        verify_certs=True,
        request_timeout=settings.elasticsearch_request_timeout_seconds,
        connections_per_node=settings.elasticsearch_connections_per_node,
        retry_on_timeout=False,
        max_retries=0,
    )
    try:
        if not await elasticsearch.ping():
            raise RagStartupError("RAG_INDEX_UNAVAILABLE")
        # 3、模型加载本身也是同步CPU工作，放入临时工作线程避免阻塞事件循环。
        embedding_model, reranker_model = await asyncio.to_thread(_load_bge_models, model_paths)
        runtime = BgeModelRuntime(
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            max_concurrency=settings.rag_model_max_concurrency,
            deadline_seconds=settings.rag_model_deadline_seconds,
        )
        await runtime.warm_up()
    except BaseException:
        if runtime is not None:
            await runtime.close()
        await elasticsearch.close()
        raise
    assert runtime is not None
    # 4、生产装配只暴露锁定BGE适配器，不注册Fake、零向量或远程LLM替代实现。
    embeddings = BgeM3EmbeddingProvider(
        runtime=runtime,
        model_id="BAAI/bge-m3",
        revision=settings.bge_embedding_revision,
        expected_revision=settings.bge_embedding_revision,
        batch_size=settings.rag_embedding_batch_size,
    )
    reranker = BgeReranker(
        runtime=runtime,
        model_id="BAAI/bge-reranker-v2-m3",
        revision=settings.bge_reranker_revision,
        expected_revision=settings.bge_reranker_revision,
        batch_size=settings.rag_reranker_batch_size,
    )
    return RagRuntimeResources(
        elasticsearch=elasticsearch,
        models=runtime,
        embeddings=embeddings,
        reranker=reranker,
    )


@asynccontextmanager
async def rag_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """按模型→ES的逆序关闭应用级RAG资源；禁用时不创建替代实现。"""
    # 1、所有环境都显式初始化state，调用方不会误用残留对象。
    settings = cast(Settings, app.state.settings)
    app.state.rag_resources = None
    if not settings.rag_enabled:
        yield
        return
    # 2、启用后必须创建真实资源，失败直接阻断应用启动。
    resources = await create_rag_resources(settings)
    app.state.rag_resources = resources
    try:
        yield
    finally:
        # 3、先停止模型工作线程，再关闭其下游ES连接池。
        await resources.models.close()
        await resources.elasticsearch.close()
        app.state.rag_resources = None
