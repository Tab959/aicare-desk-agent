"""验证RAG就绪与故障矩阵均fail-closed且没有内存回退。"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from aicare_agent_service.config import Environment, Settings
from aicare_agent_service.rag.readiness import RagReadinessProbe


class ProbeIndices:
    """模拟模板、别名和Mapping的只读ES接口。"""

    def __init__(self, *, fingerprint: str, alias_ok: bool = True) -> None:
        self.fingerprint = fingerprint
        self.alias_ok = alias_ok

    async def get_index_template(self, *, name: str):
        return {
            "index_templates": [
                {
                    "index_template": {
                        "template": {
                            "mappings": {
                                "_meta": {
                                    "aicare_schema_version": 1,
                                    "embedding_fingerprint": self.fingerprint,
                                }
                            }
                        }
                    }
                }
            ]
        }

    async def get_alias(self, *, name: str, **kwargs):
        if name.endswith("-read"):
            return {"aicare-kb-tenant-model-g1": {"aliases": {"aicare-kb-tenant-model-read": {}}}}
        return {
            "aicare-kb-tenant-model-g1": {
                "aliases": {"aicare-kb-tenant-model-write": {"is_write_index": self.alias_ok}}
            }
        }

    async def get_mapping(self, *, index: str):
        return {
            index: {
                "mappings": {
                    "_meta": {
                        "aicare_schema_version": 1,
                        "embedding_fingerprint": self.fingerprint,
                    }
                }
            }
        }


class ProbeClient:
    """模拟ES集群和索引命名空间。"""

    def __init__(self, *, fingerprint: str, cluster_status: str = "green") -> None:
        self.indices = ProbeIndices(fingerprint=fingerprint)
        self.cluster = SimpleNamespace(health=self._health)
        self.cluster_status = cluster_status

    async def _health(self, **kwargs):
        return {"status": self.cluster_status}


@pytest.mark.asyncio
async def test_readiness_fails_closed_for_cluster_alias_or_model_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """任一生产依赖失败都返回DOWN，探针不会创建替代索引。"""
    lock = tmp_path / "models.lock.json"
    root = tmp_path / "models"
    root.mkdir()
    lock.write_text("{}", encoding="utf-8")
    settings = Settings(
        environment=Environment.TEST,
        rag_chunk_hmac_key="x" * 32,
        rag_model_lock_path=lock,
        rag_model_dir=root,
        bge_embedding_revision="a" * 40,
        bge_reranker_revision="b" * 40,
        _env_file=None,
    )
    monkeypatch.setattr(
        "aicare_agent_service.rag.readiness.verify_model_lock",
        lambda **kwargs: {"embedding": root, "reranker": root},
    )
    fingerprint = "f" * 64
    healthy = RagReadinessProbe(
        settings=settings,
        client=ProbeClient(fingerprint=fingerprint),
        models=SimpleNamespace(ready=True),
        embedding_fingerprint=fingerprint,
    )
    assert (await healthy.check()).ready is True

    cluster_down = RagReadinessProbe(
        settings=settings,
        client=ProbeClient(fingerprint=fingerprint, cluster_status="red"),
        models=SimpleNamespace(ready=True),
        embedding_fingerprint=fingerprint,
    )
    assert (await cluster_down.check()).elasticsearch_cluster == "DOWN"

    model_down = RagReadinessProbe(
        settings=settings,
        client=ProbeClient(fingerprint=fingerprint),
        models=SimpleNamespace(ready=False),
        embedding_fingerprint=fingerprint,
    )
    assert (await model_down.check()).models == "DOWN"

    drifted = RagReadinessProbe(
        settings=settings,
        client=ProbeClient(fingerprint="0" * 64),
        models=SimpleNamespace(ready=True),
        embedding_fingerprint=fingerprint,
    )
    report = await drifted.check()
    assert report.index_template == "DOWN"
    assert report.aliases_mapping == "DOWN"
