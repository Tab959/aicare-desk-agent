"""提供使用独立管理凭据显式安装模板、初始化租户和切换索引代次的CLI。"""

from __future__ import annotations

import argparse
import asyncio

from elasticsearch import AsyncElasticsearch

from aicare_agent_service.config import Settings
from aicare_agent_service.rag.embeddings import model_fingerprint
from aicare_agent_service.rag.index_manager import ElasticsearchIndexManager


def _parser() -> argparse.ArgumentParser:
    """构造只允许显式基础设施动作的命令行参数。"""
    # 1、动作固定为模板、首次租户初始化、创建新代次和原子切换。
    parser = argparse.ArgumentParser(description="AICare Agent RAG索引管理")
    parser.add_argument("action", choices=("install-template", "init-tenant", "create", "switch"))
    parser.add_argument("--tenant-id")
    parser.add_argument("--generation", type=int)
    # 2、解析器本身不接收密码参数，凭据只能从进程环境注入。
    return parser


async def _run(args: argparse.Namespace, settings: Settings) -> None:
    """校验管理配置并执行一个明确的索引管理动作。"""
    # 1、管理CLI必须使用独立账号、TLS CA和锁定Embedding revision。
    if (
        settings.elasticsearch_admin_username is None
        or settings.elasticsearch_admin_password is None
        or settings.elasticsearch_ca_cert_path is None
        or settings.bge_embedding_revision is None
        or settings.rag_chunk_hmac_key is None
    ):
        raise ValueError("RAG_INDEX_ADMIN_CONFIGURATION_INCOMPLETE")
    fingerprint = model_fingerprint(
        "BAAI/bge-m3",
        settings.bge_embedding_revision,
        "dense:1024",
    )
    client = AsyncElasticsearch(
        hosts=[str(node) for node in settings.elasticsearch_node_urls],
        basic_auth=(
            settings.elasticsearch_admin_username,
            settings.elasticsearch_admin_password.get_secret_value(),
        ),
        ca_certs=str(settings.elasticsearch_ca_cert_path),
        verify_certs=True,
        request_timeout=settings.elasticsearch_request_timeout_seconds,
        retry_on_timeout=False,
        max_retries=0,
    )
    try:
        manager = ElasticsearchIndexManager(
            client=client,
            index_prefix=settings.elasticsearch_index_prefix,
            tenant_hmac_key=settings.rag_chunk_hmac_key.get_secret_value().encode(),
            embedding_fingerprint=fingerprint,
        )
        # 2、模板安装不需要租户，其余动作必须提供租户和对应代次。
        if args.action == "install-template":
            await manager.install_template()
            return
        if not args.tenant_id:
            raise ValueError("RAG_INDEX_TENANT_REQUIRED")
        if args.action == "init-tenant":
            await manager.initialize_tenant(args.tenant_id)
            return
        if args.generation is None or args.generation < 1:
            raise ValueError("RAG_INDEX_GENERATION_REQUIRED")
        if args.action == "create":
            await manager.create_generation(args.tenant_id, generation=args.generation)
            return
        await manager.switch_generation(args.tenant_id, generation=args.generation)
    finally:
        # 3、无论动作成功或失败都关闭管理连接池，不遗留后台连接。
        await client.close()


def main() -> None:
    """解析环境与参数并运行异步管理流程。"""
    # 1、Settings只读取项目唯一.env和当前进程环境，密码不出现在命令行历史。
    settings = Settings()
    args = _parser().parse_args()
    # 2、命令失败直接返回非零，不提供隐式创建或静默降级。
    asyncio.run(_run(args, settings))


if __name__ == "__main__":
    main()
