"""执行RAG生产就绪检查，不修复模型、索引、别名或Mapping漂移。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from elasticsearch import AsyncElasticsearch

from aicare_agent_service.config import Settings
from aicare_agent_service.rag.elasticsearch_mapping import INDEX_SCHEMA_VERSION
from aicare_agent_service.rag.index_manager import ElasticsearchIndexManager
from aicare_agent_service.rag.model_lock import ModelLockError, verify_model_lock

ReadinessStatus = Literal["UP", "DOWN"]


class ReadyModelRuntime(Protocol):
    """就绪探针读取BGE运行时状态所需的最小端口。"""

    @property
    def ready(self) -> bool:
        """模型锁校验、加载和热身全部完成后返回True。"""
        ...


@dataclass(frozen=True, slots=True)
class RagReadinessReport:
    """RAG各生产依赖的低基数就绪结果。"""

    models: ReadinessStatus
    elasticsearch_cluster: ReadinessStatus
    index_template: ReadinessStatus
    aliases_mapping: ReadinessStatus

    @property
    def ready(self) -> bool:
        """所有检查项均为UP时才允许接收流量。"""
        # 1、任何单项DOWN都使RAG整体不可就绪。
        return all(
            value == "UP"
            for value in (
                self.models,
                self.elasticsearch_cluster,
                self.index_template,
                self.aliases_mapping,
            )
        )


class RagReadinessProbe:
    """验证模型产物、ES集群、模板以及当前全部租户别名和Mapping。"""

    def __init__(
        self,
        *,
        settings: Settings,
        client: AsyncElasticsearch | Any,
        models: ReadyModelRuntime,
        embedding_fingerprint: str,
    ) -> None:
        """绑定只读生产依赖与当前模型指纹。"""
        # 1、探针只保存已验证配置和在线资源，不持有管理凭据。
        self._settings = settings
        self._client = client
        self._models = models
        self._fingerprint = embedding_fingerprint
        assert settings.rag_chunk_hmac_key is not None
        # 2、IndexManager仅用于只读模板验证，不调用初始化或切换方法。
        self._manager = ElasticsearchIndexManager(
            client=client,
            index_prefix=settings.elasticsearch_index_prefix,
            tenant_hmac_key=settings.rag_chunk_hmac_key.get_secret_value().encode(),
            embedding_fingerprint=embedding_fingerprint,
        )

    async def check(self) -> RagReadinessReport:
        """独立检查四类依赖并返回不含异常明文的状态。"""
        # 1、本地模型必须仍匹配锁文件且已经完成运行时热身。
        models_status: ReadinessStatus = "DOWN"
        try:
            assert self._settings.rag_model_lock_path is not None
            assert self._settings.rag_model_dir is not None
            assert self._settings.bge_embedding_revision is not None
            assert self._settings.bge_reranker_revision is not None
            verify_model_lock(
                lock_path=self._settings.rag_model_lock_path,
                model_root=self._settings.rag_model_dir,
                expected_revisions={
                    "embedding": self._settings.bge_embedding_revision,
                    "reranker": self._settings.bge_reranker_revision,
                },
            )
            models_status = "UP" if self._models.ready else "DOWN"
        except (AssertionError, ModelLockError, OSError):
            models_status = "DOWN"

        # 2、集群必须可达且至少为yellow；red或格式异常均阻断。
        cluster_status: ReadinessStatus = "DOWN"
        try:
            cluster = await self._client.cluster.health(wait_for_status="yellow", timeout="2s")
            cluster_status = "UP" if cluster.get("status") in {"yellow", "green"} else "DOWN"
        except Exception:  # noqa: BLE001 - 第三方客户端异常不得穿透健康接口
            cluster_status = "DOWN"

        # 3、模板schema和当前Embedding指纹必须精确匹配。
        template_status: ReadinessStatus = "DOWN"
        try:
            await self._manager.validate_template()
            template_status = "UP"
        except Exception:  # noqa: BLE001 - 只返回稳定状态，不暴露ES异常
            template_status = "DOWN"

        # 4、至少一个当前租户索引必须具备成对读写别名和兼容Mapping。
        alias_status: ReadinessStatus = "DOWN"
        try:
            alias_status = "UP" if await self._validate_aliases_and_mappings() else "DOWN"
        except Exception:  # noqa: BLE001 - 权限、认证、超时和漂移统一fail-closed
            alias_status = "DOWN"
        return RagReadinessReport(
            models=models_status,
            elasticsearch_cluster=cluster_status,
            index_template=template_status,
            aliases_mapping=alias_status,
        )

    async def _validate_aliases_and_mappings(self) -> bool:
        """验证全部当前读别名都有唯一同索引写别名和正确Mapping元数据。"""
        # 1、读取受控前缀下全部read别名；没有初始化知识租户时不具备服务能力。
        response = await self._client.indices.get_alias(
            name=f"{self._settings.elasticsearch_index_prefix}-*-read",
            allow_no_indices=True,
            expand_wildcards="open",
        )
        if not response:
            return False
        # 2、每个物理索引只能出现一个read别名，并必须存在同命名空间write别名。
        for index_name, payload in response.items():
            read_aliases = [name for name in payload.get("aliases", {}) if name.endswith("-read")]
            if len(read_aliases) != 1:
                return False
            write_alias = read_aliases[0][:-5] + "-write"
            write_response = await self._client.indices.get_alias(
                index=index_name, name=write_alias
            )
            write_config = write_response[index_name]["aliases"][write_alias]
            if write_config.get("is_write_index") is not True:
                return False
            # 3、物理Mapping必须仍使用当前schema和Embedding指纹。
            mapping = await self._client.indices.get_mapping(index=index_name)
            meta = mapping[index_name]["mappings"]["_meta"]
            if (
                meta.get("aicare_schema_version") != INDEX_SCHEMA_VERSION
                or meta.get("embedding_fingerprint") != self._fingerprint
            ):
                return False
        return True
