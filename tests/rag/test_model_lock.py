"""验证离线模型锁的严格解析、文件哈希与版本一致性。"""

import hashlib
import json
from pathlib import Path

import pytest

from aicare_agent_service.rag.model_lock import (
    ModelLockError,
    initialize_locked_models,
    verify_model_lock,
)


def _write_lock(tmp_path: Path, *, sha256: str, revision: str = "a" * 40) -> Path:
    """写入只包含一个小文件的锁定模型样例。"""
    manifest = {
        "schemaVersion": 1,
        "models": [
            {
                "role": "embedding",
                "modelId": "BAAI/bge-m3",
                "revision": revision,
                "license": "mit",
                "dimensions": 1024,
                "files": [{"path": "config.json", "sha256": sha256, "size": 2}],
            }
        ],
    }
    path = tmp_path / "models.lock.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_verify_model_lock_accepts_exact_revision_and_hash(tmp_path: Path) -> None:
    model_file = tmp_path / "models" / "BAAI--bge-m3" / "config.json"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"{}")
    lock_path = _write_lock(tmp_path, sha256=hashlib.sha256(b"{}").hexdigest())

    verified = verify_model_lock(
        lock_path=lock_path,
        model_root=tmp_path / "models",
        expected_revisions={"embedding": "a" * 40},
    )

    assert verified["embedding"] == model_file.parent


def test_verify_model_lock_rejects_tampered_file_without_leaking_path(tmp_path: Path) -> None:
    model_file = tmp_path / "models" / "BAAI--bge-m3" / "config.json"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"tampered")
    lock_path = _write_lock(tmp_path, sha256=hashlib.sha256(b"{}").hexdigest())

    with pytest.raises(ModelLockError) as exc_info:
        verify_model_lock(
            lock_path=lock_path,
            model_root=tmp_path / "models",
            expected_revisions={"embedding": "a" * 40},
        )

    assert exc_info.value.code == "RAG_MODEL_HASH_MISMATCH"
    assert str(tmp_path) not in str(exc_info.value)


def test_verify_model_lock_rejects_revision_mismatch(tmp_path: Path) -> None:
    lock_path = _write_lock(tmp_path, sha256="0" * 64)

    with pytest.raises(ModelLockError) as exc_info:
        verify_model_lock(
            lock_path=lock_path,
            model_root=tmp_path / "models",
            expected_revisions={"embedding": "b" * 40},
        )

    assert exc_info.value.code == "RAG_MODEL_REVISION_MISMATCH"


def test_verify_model_lock_rejects_path_traversal(tmp_path: Path) -> None:
    lock_path = _write_lock(tmp_path, sha256="0" * 64)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["models"][0]["files"][0]["path"] = "../secret.txt"
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelLockError) as exc_info:
        verify_model_lock(
            lock_path=lock_path,
            model_root=tmp_path / "models",
            expected_revisions={"embedding": "a" * 40},
        )

    assert exc_info.value.code == "RAG_MODEL_LOCK_INVALID"


def test_initializer_downloads_only_locked_files_at_exact_revision(tmp_path: Path) -> None:
    lock_path = _write_lock(tmp_path, sha256=hashlib.sha256(b"{}").hexdigest())
    calls: list[dict[str, object]] = []

    def downloader(**kwargs: object) -> str:
        calls.append(kwargs)
        target = Path(str(kwargs["local_dir"]))
        target.mkdir(parents=True)
        (target / "config.json").write_bytes(b"{}")
        return str(target)

    initialized = initialize_locked_models(
        lock_path=lock_path,
        model_root=tmp_path / "models",
        downloader=downloader,
    )

    assert initialized == {"embedding": tmp_path / "models" / "BAAI--bge-m3"}
    assert calls == [
        {
            "repo_id": "BAAI/bge-m3",
            "revision": "a" * 40,
            "allow_patterns": ["config.json"],
            "local_dir": str(tmp_path / "models" / "BAAI--bge-m3"),
        }
    ]
