"""显式初始化、验证、重建并原子切换每租户 Elasticsearch 知识索引。"""

from __future__ import annotations

from typing import Any

from elasticsearch import AsyncElasticsearch, NotFoundError

from aicare_agent_service.rag.elasticsearch_mapping import (
    INDEX_SCHEMA_VERSION,
    TenantIndexNames,
    build_index_template,
    build_tenant_index_names,
    derive_tenant_namespace_key,
    index_template_name,
)
from aicare_agent_service.rag.errors import RagError, RagErrorCode


class ElasticsearchIndexManager:
    """管理显式索引生命周期；应用运行路径只允许调用验证方法。"""

    def __init__(
        self,
        *,
        client: AsyncElasticsearch | Any,
        index_prefix: str,
        tenant_hmac_key: bytes,
        embedding_fingerprint: str,
    ) -> None:
        """绑定ES客户端、索引前缀、租户HMAC根密钥和模型指纹。"""
        # 1、通过名称构造器集中执行生产参数校验。
        build_tenant_index_names(
            prefix=index_prefix,
            tenant_id="validation-tenant",
            hmac_key=tenant_hmac_key,
            embedding_fingerprint=embedding_fingerprint,
            generation=1,
        )
        # 2、保存后续显式管理和运行时验证所需的固定配置。
        self._client = client
        self._prefix = index_prefix
        self._hmac_key = derive_tenant_namespace_key(tenant_hmac_key)
        self._fingerprint = embedding_fingerprint

    def names(self, tenant_id: str, *, generation: int = 1) -> TenantIndexNames:
        """构造指定租户和代次的安全索引名称。"""
        # 1、统一复用HMAC命名，调用方不能自行拼接租户索引名。
        return build_tenant_index_names(
            prefix=self._prefix,
            tenant_id=tenant_id,
            hmac_key=self._hmac_key,
            embedding_fingerprint=self._fingerprint,
            generation=generation,
        )

    async def install_template(self) -> None:
        """使用管理凭据显式安装或升级受版本控制的索引模板。"""
        # 1、生成与当前schema和Embedding指纹严格匹配的模板。
        body = build_index_template(
            prefix=self._prefix,
            embedding_fingerprint=self._fingerprint,
        )
        # 2、通过组合模板API写入；生产应用启动不调用本方法。
        await self._client.indices.put_index_template(
            name=index_template_name(self._prefix),
            index_patterns=body["index_patterns"],
            priority=body["priority"],
            version=body["version"],
            meta=body["_meta"],
            template=body["template"],
        )

    async def create_generation(self, tenant_id: str, *, generation: int) -> TenantIndexNames:
        """显式创建新物理索引但不自动切换线上别名。"""
        # 1、先验证全局模板存在且与当前运行时指纹一致。
        await self.validate_template()
        names = self.names(tenant_id, generation=generation)
        # 2、物理索引由模板应用严格Mapping，禁用自动创建时也可预测执行。
        await self._client.indices.create(index=names.physical_index)
        return names

    async def initialize_tenant(self, tenant_id: str) -> TenantIndexNames:
        """首次创建租户generation 1并一次性绑定读写别名。"""
        # 1、只允许显式初始化命令创建物理索引。
        names = await self.create_generation(tenant_id, generation=1)
        alias_filter = {"term": {"tenant_namespace": names.tenant_namespace}}
        # 2、同一个原子请求建立读写别名、过滤器和routing。
        await self._client.indices.update_aliases(
            actions=[
                {
                    "add": {
                        "index": names.physical_index,
                        "alias": names.read_alias,
                        "filter": alias_filter,
                        "routing": names.routing,
                    }
                },
                {
                    "add": {
                        "index": names.physical_index,
                        "alias": names.write_alias,
                        "filter": alias_filter,
                        "routing": names.routing,
                        "is_write_index": True,
                    }
                },
            ]
        )
        return names

    async def switch_generation(
        self,
        tenant_id: str,
        *,
        generation: int,
    ) -> TenantIndexNames:
        """把已完整构建的新代次原子切换为读写目标并保留旧索引。"""
        # 1、目标索引必须已存在并通过Mapping指纹验证。
        names = self.names(tenant_id, generation=generation)
        if not await self._client.indices.exists(index=names.physical_index):
            raise RagError(RagErrorCode.INDEX_UNAVAILABLE)
        await self._validate_mapping(names.physical_index)
        # 2、读取当前别名目标，用单个actions请求移除旧绑定并添加新绑定。
        actions: list[dict[str, Any]] = []
        for alias in (names.read_alias, names.write_alias):
            try:
                current = await self._client.indices.get_alias(name=alias)
            except NotFoundError:
                current = {}
            for index_name in current:
                actions.append({"remove": {"index": index_name, "alias": alias}})
        alias_filter = {"term": {"tenant_namespace": names.tenant_namespace}}
        actions.extend(
            [
                {
                    "add": {
                        "index": names.physical_index,
                        "alias": names.read_alias,
                        "filter": alias_filter,
                        "routing": names.routing,
                    }
                },
                {
                    "add": {
                        "index": names.physical_index,
                        "alias": names.write_alias,
                        "filter": alias_filter,
                        "routing": names.routing,
                        "is_write_index": True,
                    }
                },
            ]
        )
        await self._client.indices.update_aliases(actions=actions)
        # 3、旧物理索引不在切换路径删除，交由独立保留期清理流程处理。
        return names

    async def validate_template(self) -> None:
        """校验模板schema和Embedding指纹，不执行隐式创建或迁移。"""
        # 1、缺失模板直接阻断，生产启动不得自行修复基础设施。
        try:
            response = await self._client.indices.get_index_template(
                name=index_template_name(self._prefix)
            )
            templates = response["index_templates"]
            template = templates[0]["index_template"]
            mappings = template["template"]["mappings"]
        except (KeyError, IndexError, NotFoundError) as exc:
            raise RagError(RagErrorCode.INDEX_UNAVAILABLE) from exc
        # 2、schema或指纹漂移一律失败，避免不同向量空间混用。
        self._validate_mapping_meta(mappings.get("_meta", {}))

    async def validate_tenant(self, tenant_id: str) -> TenantIndexNames:
        """校验租户读写别名、write target和物理Mapping兼容性。"""
        # 1、运行路径只解析稳定别名，不依赖固定物理代次。
        names = self.names(tenant_id)
        try:
            read_targets = await self._client.indices.get_alias(name=names.read_alias)
            write_targets = await self._client.indices.get_alias(name=names.write_alias)
        except NotFoundError as exc:
            raise RagError(RagErrorCode.INDEX_UNAVAILABLE) from exc
        if len(read_targets) != 1 or len(write_targets) != 1:
            raise RagError(RagErrorCode.INDEX_UNAVAILABLE)
        write_index, write_payload = next(iter(write_targets.items()))
        alias_payload = write_payload.get("aliases", {}).get(names.write_alias, {})
        if alias_payload.get("is_write_index") is not True:
            raise RagError(RagErrorCode.INDEX_UNAVAILABLE)
        # 2、读写别名必须指向同一当前代次，随后校验物理Mapping。
        if next(iter(read_targets)) != write_index:
            raise RagError(RagErrorCode.INDEX_UNAVAILABLE)
        await self._validate_mapping(write_index)
        return names

    async def _validate_mapping(self, index_name: str) -> None:
        """读取单个物理索引Mapping并验证受控元数据。"""
        # 1、任何响应缺项都按索引不可用处理，不猜测默认schema。
        try:
            response = await self._client.indices.get_mapping(index=index_name)
            meta = response[index_name]["mappings"]["_meta"]
        except (KeyError, NotFoundError) as exc:
            raise RagError(RagErrorCode.INDEX_UNAVAILABLE) from exc
        # 2、复用严格meta门禁。
        self._validate_mapping_meta(meta)

    def _validate_mapping_meta(self, meta: dict[str, Any]) -> None:
        """比较Mapping元数据与当前运行时契约。"""
        # 1、旧schema或不同Embedding模型都不能继续读写。
        if (
            meta.get("aicare_schema_version") != INDEX_SCHEMA_VERSION
            or meta.get("embedding_fingerprint") != self._fingerprint
        ):
            raise RagError(RagErrorCode.INDEX_UNAVAILABLE)
