"""实现租户隔离、外部版本控制和逐项Bulk检查的 Elasticsearch 知识存储。"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from elasticsearch import AsyncElasticsearch, ConflictError, NotFoundError, TransportError

from aicare_agent_service.contracts import Citation
from aicare_agent_service.rag.contracts import (
    IndexedDocument,
    IndexResult,
    IndexStatus,
    KnowledgeMetadata,
    RetrievalQuery,
    RetrievedChunk,
)
from aicare_agent_service.rag.errors import RagError, RagErrorCode
from aicare_agent_service.rag.index_manager import ElasticsearchIndexManager


class ElasticsearchKnowledgeIndex:
    """通过受控别名提供幂等文档替换、版本墓碑和安全删除。"""

    def __init__(
        self,
        *,
        client: AsyncElasticsearch | Any,
        index_prefix: str,
        tenant_hmac_key: bytes,
        embedding_fingerprint: str,
    ) -> None:
        """绑定ES连接与索引兼容性参数。"""
        # 1、IndexManager集中验证命名、别名、schema与Embedding指纹。
        self._client = client
        self._manager = ElasticsearchIndexManager(
            client=client,
            index_prefix=index_prefix,
            tenant_hmac_key=tenant_hmac_key,
            embedding_fingerprint=embedding_fingerprint,
        )
        # 2、保存当前Embedding指纹用于写入marker和chunk。
        self._fingerprint = embedding_fingerprint

    async def replace_document(self, document: IndexedDocument) -> IndexResult:
        """以外部版本marker保护并完整替换一个文档的全部Chunk。"""
        # 1、索引模型指纹必须与运行时一致，且租户别名和Mapping必须有效。
        metadata = document.metadata
        if document.embedding_fingerprint != self._fingerprint:
            return self._result(document, IndexStatus.FAILED, RagErrorCode.INDEX_VERSION_CONFLICT)
        names = await self._manager.validate_tenant(metadata.tenant_id)
        marker_id = self._marker_id(names.tenant_namespace, metadata.document_id)
        existing = await self._read_marker(names.write_alias, names.routing, marker_id)
        checksum = self._document_checksum(document)
        if existing is not None:
            current_version = int(existing.get("version", 0))
            if current_version > metadata.version:
                return self._result(
                    document,
                    IndexStatus.SKIPPED,
                    RagErrorCode.INDEX_VERSION_CONFLICT,
                )
            if (
                current_version == metadata.version
                and existing.get("status") == "PUBLISHED"
                and existing.get("completed") is True
                and existing.get("document_checksum") == checksum
            ):
                return self._result(document, IndexStatus.SKIPPED)
            if current_version == metadata.version and existing.get("status") == "DELETED":
                return self._result(
                    document,
                    IndexStatus.SKIPPED,
                    RagErrorCode.INDEX_VERSION_CONFLICT,
                )
        # 2、先以external_gte写入未完成marker；更高版本已占位时ES返回409阻断旧事件。
        marker = self._marker_source(
            namespace=names.tenant_namespace,
            document_id=metadata.document_id,
            version=metadata.version,
            status="INDEXING",
            document_checksum=checksum,
            completed=False,
        )
        try:
            await self._client.index(
                index=names.write_alias,
                id=marker_id,
                routing=names.routing,
                document=marker,
                version=metadata.version,
                version_type="external_gte",
                require_alias=True,
            )
        except ConflictError:
            return self._result(
                document,
                IndexStatus.SKIPPED,
                RagErrorCode.INDEX_VERSION_CONFLICT,
            )
        # 3、Bulk写入确定性Chunk ID并逐项检查；部分成功不清理旧版本，可安全重试。
        operations: list[dict[str, Any]] = []
        for chunk in document.chunks:
            operations.extend(
                [
                    {"index": {"_id": chunk.chunk_id}},
                    self._chunk_source(names.tenant_namespace, chunk),
                ]
            )
        try:
            response = await self._client.bulk(
                index=names.write_alias,
                routing=names.routing,
                require_alias=True,
                refresh="wait_for",
                operations=operations,
            )
        except TransportError:
            return self._result(document, IndexStatus.FAILED, RagErrorCode.INDEX_UNAVAILABLE)
        if self._bulk_failed(response, expected_items=len(document.chunks)):
            return self._result(document, IndexStatus.FAILED, RagErrorCode.INDEX_UNAVAILABLE)
        # 4、Chunk完整可见后才把marker置为完成；若新版本抢先完成，旧版本不能覆盖它。
        marker["status"] = "PUBLISHED"
        marker["completed"] = True
        try:
            await self._client.index(
                index=names.write_alias,
                id=marker_id,
                routing=names.routing,
                document=marker,
                version=metadata.version,
                version_type="external_gte",
                require_alias=True,
                refresh="wait_for",
            )
        except ConflictError:
            await self._delete_chunk_versions(
                names.write_alias,
                names.routing,
                names.tenant_namespace,
                metadata.document_id,
                exact_version=metadata.version,
            )
            return self._result(
                document,
                IndexStatus.SKIPPED,
                RagErrorCode.INDEX_VERSION_CONFLICT,
            )
        # 5、只在完成marker后清理更旧Chunk，新版本永远不在删除范围内。
        await self._delete_chunk_versions(
            names.write_alias,
            names.routing,
            names.tenant_namespace,
            metadata.document_id,
            before_version=metadata.version,
        )
        return self._result(document, IndexStatus.INDEXED, indexed_chunks=len(document.chunks))

    async def delete_document(
        self,
        tenant_id: str,
        document_id: str,
        version: int,
    ) -> IndexResult:
        """写入版本墓碑并只删除不高于该事件版本的Chunk。"""
        # 1、输入身份和版本先确定性校验，随后验证租户别名与Mapping。
        if not tenant_id.strip() or not document_id.strip() or version < 1:
            raise ValueError("RAG_DELETE_ARGUMENT_INVALID")
        names = await self._manager.validate_tenant(tenant_id)
        marker_id = self._marker_id(names.tenant_namespace, document_id)
        existing = await self._read_marker(names.write_alias, names.routing, marker_id)
        if existing is not None:
            current_version = int(existing.get("version", 0))
            if current_version > version:
                return IndexResult(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    version=version,
                    status=IndexStatus.SKIPPED,
                    error_code=RagErrorCode.INDEX_VERSION_CONFLICT.value,
                )
            if current_version == version and existing.get("status") == "DELETED":
                return IndexResult(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    version=version,
                    status=IndexStatus.SKIPPED,
                )
        # 2、external_gte墓碑阻断任何更旧的replace/delete事件重新写回。
        marker = self._marker_source(
            namespace=names.tenant_namespace,
            document_id=document_id,
            version=version,
            status="DELETED",
            document_checksum=None,
            completed=True,
        )
        try:
            await self._client.index(
                index=names.write_alias,
                id=marker_id,
                routing=names.routing,
                document=marker,
                version=version,
                version_type="external_gte",
                require_alias=True,
                refresh="wait_for",
            )
        except ConflictError:
            return IndexResult(
                tenant_id=tenant_id,
                document_id=document_id,
                version=version,
                status=IndexStatus.SKIPPED,
                error_code=RagErrorCode.INDEX_VERSION_CONFLICT.value,
            )
        # 3、删除范围带租户namespace、doc_kind、document ID和version上限。
        await self._delete_chunk_versions(
            names.write_alias,
            names.routing,
            names.tenant_namespace,
            document_id,
            through_version=version,
        )
        return IndexResult(
            tenant_id=tenant_id,
            document_id=document_id,
            version=version,
            status=IndexStatus.DELETED,
        )

    async def sparse_search(self, query: RetrievalQuery) -> tuple[RetrievedChunk, ...]:
        """在租户read alias内执行SmartCN与standard多字段BM25召回。"""
        # 1、先验证租户alias和Mapping指纹，再由代码生成不可被模型覆盖的过滤条件。
        names = await self._manager.validate_tenant(query.tenant_id)
        filters = self._retrieval_filters(names.tenant_namespace, query)
        # 2、查询只返回回答所需白名单source，明确排除1024维Embedding。
        try:
            response = await self._client.search(
                index=names.read_alias,
                routing=names.routing,
                size=query.candidate_limit,
                query={
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query.text,
                                    "fields": ["content^2", "content.standard"],
                                    "type": "best_fields",
                                }
                            }
                        ],
                        "filter": filters,
                    }
                },
                source_includes=self._answer_source_fields(),
                source_excludes=("embedding",),
                track_total_hits=False,
            )
        except TransportError as exc:
            raise RagError(RagErrorCode.INDEX_UNAVAILABLE) from exc
        # 3、ES原始分数不进入安全契约，排序信息由返回顺序交给Python RRF。
        return self._parse_hits(response, query.tenant_id, names.tenant_namespace)

    async def dense_search(
        self,
        query: RetrievalQuery,
        vector: Any,
        *,
        num_candidates: int,
    ) -> tuple[RetrievedChunk, ...]:
        """在同一租户过滤条件下执行1024维cosine HNSW召回。"""
        # 1、查询向量和HNSW候选预算在请求ES前严格校验。
        values = tuple(float(value) for value in vector)
        if len(values) != 1024 or num_candidates < query.candidate_limit:
            raise ValueError("RAG_DENSE_QUERY_INVALID")
        names = await self._manager.validate_tenant(query.tenant_id)
        filters = self._retrieval_filters(names.tenant_namespace, query)
        # 2、kNN filter是召回前过滤，不能用召回后的post_filter代替租户边界。
        try:
            response = await self._client.search(
                index=names.read_alias,
                routing=names.routing,
                knn={
                    "field": "embedding",
                    "query_vector": list(values),
                    "k": query.candidate_limit,
                    "num_candidates": num_candidates,
                    "filter": filters,
                },
                size=query.candidate_limit,
                source_includes=self._answer_source_fields(),
                source_excludes=("embedding",),
                track_total_hits=False,
            )
        except TransportError as exc:
            raise RagError(RagErrorCode.INDEX_UNAVAILABLE) from exc
        # 3、与BM25共用安全解析，跨租户canary或source缺项直接阻断。
        return self._parse_hits(response, query.tenant_id, names.tenant_namespace)

    async def _read_marker(self, alias: str, routing: str, marker_id: str) -> dict[str, Any] | None:
        """读取文档版本marker；不存在时返回None。"""
        # 1、读取始终走受控write alias与租户routing，不扫描其他租户索引。
        try:
            response = await self._client.get(
                index=alias,
                id=marker_id,
                routing=routing,
                source_excludes=("embedding",),
            )
        except NotFoundError:
            return None
        # 2、只返回marker安全source，不向上暴露ES元数据。
        source = response.get("_source")
        if not isinstance(source, dict):
            raise RagError(RagErrorCode.INDEX_UNAVAILABLE)
        return source

    def _retrieval_filters(self, namespace: str, query: RetrievalQuery) -> list[dict[str, Any]]:
        """从运行时租户和严格业务Filter生成ES预过滤条件。"""
        # 1、租户、文档类型、发布状态和Embedding指纹是不可省略的固定条件。
        clauses: list[dict[str, Any]] = [
            {"term": {"tenant_namespace": namespace}},
            {"term": {"doc_kind": "chunk"}},
            {"term": {"status": "PUBLISHED"}},
            {"term": {"embedding_fingerprint": self._fingerprint}},
        ]
        # 2、有限业务过滤器只缩小范围，模型不能传入tenant或任意ES字段。
        field_map = {
            "knowledge_base_ids": "knowledge_base_id",
            "languages": "language",
            "categories": "category",
            "game_ids": "game_id",
            "purchase_methods": "purchase_method",
            "issue_types": "issue_type",
        }
        for contract_field, index_field in field_map.items():
            values = getattr(query.filters, contract_field)
            if values:
                clauses.append({"terms": {index_field: list(values)}})
        return clauses

    @staticmethod
    def _answer_source_fields() -> tuple[str, ...]:
        """返回检索阶段允许从ES读取的固定source白名单。"""
        # 1、向量、checksum、marker和ES内部元数据均不属于回答证据。
        return (
            "tenant_namespace",
            "knowledge_base_id",
            "document_id",
            "version",
            "language",
            "category",
            "game_id",
            "purchase_method",
            "issue_type",
            "chunk_id",
            "title_path",
            "content",
            "source_uri",
        )

    @staticmethod
    def _parse_hits(
        response: Any,
        tenant_id: str,
        namespace: str,
    ) -> tuple[RetrievedChunk, ...]:
        """把ES hits转换为严格安全Chunk，并重新核对租户namespace。"""
        # 1、兼容Python客户端ObjectApiResponse，但不向调用者返回原始对象。
        payload = response.body if hasattr(response, "body") else response
        try:
            hits = payload["hits"]["hits"]
        except (KeyError, TypeError) as exc:
            raise RagError(RagErrorCode.INDEX_UNAVAILABLE) from exc
        result: list[RetrievedChunk] = []
        for hit in hits:
            source = hit.get("_source", {})
            if source.get("tenant_namespace") != namespace:
                raise RagError(RagErrorCode.INDEX_UNAVAILABLE)
            # 2、重新构造固定Metadata和Citation，缺字段或非法值按索引漂移阻断。
            try:
                metadata = KnowledgeMetadata(
                    tenant_id=tenant_id,
                    knowledge_base_id=source["knowledge_base_id"],
                    document_id=source["document_id"],
                    version=source["version"],
                    language=source["language"],
                    category=source["category"],
                    game_id=source.get("game_id"),
                    purchase_method=source.get("purchase_method"),
                    issue_type=source.get("issue_type"),
                )
                citation = Citation(
                    document_id=metadata.document_id,
                    version=metadata.version,
                    title_path=tuple(source["title_path"]),
                    source_uri=source["source_uri"],
                )
                result.append(
                    RetrievedChunk(
                        chunk_id=source["chunk_id"],
                        metadata=metadata,
                        content=source["content"],
                        citation=citation,
                        fused_score=0.0,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RagError(RagErrorCode.INDEX_UNAVAILABLE) from exc
        # 3、返回顺序保持ES排名，RRF以位置计算rank。
        return tuple(result)

    async def _delete_chunk_versions(
        self,
        alias: str,
        routing: str,
        namespace: str,
        document_id: str,
        *,
        before_version: int | None = None,
        through_version: int | None = None,
        exact_version: int | None = None,
    ) -> None:
        """按互斥版本条件删除当前租户的Chunk，并检查任务级失败。"""
        # 1、三种版本范围必须且只能提供一种。
        supplied = sum(
            value is not None for value in (before_version, through_version, exact_version)
        )
        if supplied != 1:
            raise ValueError("RAG_DELETE_RANGE_INVALID")
        version_filter: dict[str, Any]
        if before_version is not None:
            version_filter = {"range": {"version": {"lt": before_version}}}
        elif through_version is not None:
            version_filter = {"range": {"version": {"lte": through_version}}}
        else:
            version_filter = {"term": {"version": exact_version}}
        # 2、所有删除条件都包含租户、文档和chunk类型，禁止仅按document ID删除。
        query = {
            "bool": {
                "filter": [
                    {"term": {"tenant_namespace": namespace}},
                    {"term": {"doc_kind": "chunk"}},
                    {"term": {"document_id": document_id}},
                    version_filter,
                ]
            }
        }
        try:
            response = await self._client.delete_by_query(
                index=alias,
                routing=routing,
                query=query,
                conflicts="proceed",
                refresh=True,
            )
        except TransportError as exc:
            raise RagError(RagErrorCode.INDEX_UNAVAILABLE) from exc
        # 3、逐任务检查失败数组；版本冲突可由幂等重试收敛，其他失败必须阻断。
        if response.get("failures"):
            raise RagError(RagErrorCode.INDEX_UNAVAILABLE)

    @staticmethod
    def _marker_id(namespace: str, document_id: str) -> str:
        """生成固定marker ID，避免原始document ID出现在ES文档ID。"""
        # 1、namespace已经是租户HMAC，继续绑定文档ID生成不可逆固定摘要。
        return "marker-" + hashlib.sha256(f"{namespace}:{document_id}".encode()).hexdigest()

    @staticmethod
    def _document_checksum(document: IndexedDocument) -> str:
        """计算同版本幂等判断所需的稳定文档摘要。"""
        # 1、按契约中的Chunk顺序组合身份和正文checksum。
        digest = hashlib.sha256()
        for chunk in document.chunks:
            digest.update(chunk.chunk_id.encode())
            digest.update(b":")
            digest.update(chunk.content_checksum.encode())
            digest.update(b"\n")
        # 2、输出固定SHA-256十六进制摘要。
        return digest.hexdigest()

    def _marker_source(
        self,
        *,
        namespace: str,
        document_id: str,
        version: int,
        status: str,
        document_checksum: str | None,
        completed: bool,
    ) -> dict[str, Any]:
        """构造严格Mapping允许的版本marker。"""
        # 1、marker只保存版本协调字段，不保存正文、向量或原始租户ID。
        return {
            "doc_kind": "version_marker",
            "tenant_namespace": namespace,
            "document_id": document_id,
            "version": version,
            "status": status,
            "document_checksum": document_checksum,
            "completed": completed,
            "embedding_fingerprint": self._fingerprint,
            "indexed_at": datetime.now(UTC).isoformat(),
        }

    def _chunk_source(self, namespace: str, chunk: Any) -> dict[str, Any]:
        """把严格Chunk契约转换为ES白名单source。"""
        # 1、业务过滤字段逐项映射，不使用model_dump把tenant_id意外写入ES。
        metadata = chunk.metadata
        source: dict[str, Any] = {
            "doc_kind": "chunk",
            "tenant_namespace": namespace,
            "knowledge_base_id": metadata.knowledge_base_id,
            "document_id": metadata.document_id,
            "version": metadata.version,
            "status": "PUBLISHED",
            "language": metadata.language,
            "category": metadata.category,
            "chunk_id": chunk.chunk_id,
            "ordinal": chunk.ordinal,
            "title_path": list(chunk.title_path),
            "content": chunk.content,
            "source_uri": f"aicare://knowledge/{metadata.document_id}",
            "content_checksum": chunk.content_checksum,
            "embedding_fingerprint": self._fingerprint,
            "indexed_at": datetime.now(UTC).isoformat(),
            "embedding": list(chunk.embedding),
        }
        # 2、可选过滤字段仅在存在时写入，保持dynamic strict兼容。
        for key in ("game_id", "purchase_method", "issue_type"):
            value = getattr(metadata, key)
            if value is not None:
                source[key] = value
        return source

    @staticmethod
    def _bulk_failed(response: Any, *, expected_items: int) -> bool:
        """逐项检查Bulk结果，拒绝仅依赖顶层errors标记。"""
        # 1、响应结构、数量或顶层errors异常都视为部分失败。
        payload = response.body if hasattr(response, "body") else response
        items = payload.get("items") if isinstance(payload, Mapping) else None
        if not isinstance(items, list) or len(items) != expected_items:
            return True
        # 2、每项必须只有受支持动作且状态小于300且没有error对象。
        for item in items:
            action = item.get("index") if isinstance(item, Mapping) else None
            if (
                not isinstance(action, Mapping)
                or int(action.get("status", 500)) >= 300
                or action.get("error") is not None
            ):
                return True
        return bool(payload.get("errors"))

    @staticmethod
    def _result(
        document: IndexedDocument,
        status: IndexStatus,
        error: RagErrorCode | None = None,
        *,
        indexed_chunks: int = 0,
    ) -> IndexResult:
        """把文档身份和稳定错误码封装为索引结果。"""
        # 1、绝不附带ES原始错误或响应正文。
        return IndexResult(
            tenant_id=document.metadata.tenant_id,
            document_id=document.metadata.document_id,
            version=document.metadata.version,
            status=status,
            indexed_chunks=indexed_chunks,
            error_code=error.value if error is not None else None,
        )
