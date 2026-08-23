"""解析并校验离线 BGE 模型锁，阻止版本漂移、文件篡改和运行时下载。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError
from pydantic.alias_generators import to_camel

Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
GitRevision = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{40}$")]


class ModelLockError(RuntimeError):
    """表示模型锁无效或本地模型与锁不一致；只暴露稳定错误码。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _LockModel(BaseModel):
    """模型锁内部基类，统一拒绝未知字段并接受 camelCase。"""

    model_config = ConfigDict(
        alias_generator=to_camel,
        extra="forbid",
        frozen=True,
        validate_by_alias=True,
        validate_by_name=False,
        hide_input_in_errors=True,
    )


class LockedFile(_LockModel):
    """一个模型文件的仓库相对路径、SHA-256 和字节数。"""

    path: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    sha256: Sha256
    size: Annotated[int, Field(strict=True, ge=0)]


class LockedModel(_LockModel):
    """一个模型角色对应的仓库、精确 revision 和文件集合。"""

    role: Literal["embedding", "reranker"]
    model_id: Literal["BAAI/bge-m3", "BAAI/bge-reranker-v2-m3"]
    revision: GitRevision
    license: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    dimensions: Literal[1024]
    files: Annotated[tuple[LockedFile, ...], Field(min_length=1)]


class ModelLock(_LockModel):
    """models.lock.json 顶层结构及两个受控 BGE 角色。"""

    schema_version: Literal[1]
    models: Annotated[tuple[LockedModel, ...], Field(min_length=1, max_length=2)]


def load_model_lock(lock_path: Path) -> ModelLock:
    """读取严格模型锁；任何解析细节都转换为不泄漏路径的稳定错误。"""
    # 1、读取并解析本地锁文件，不尝试联网补全缺失内容。
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        manifest = ModelLock.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ModelLockError("RAG_MODEL_LOCK_INVALID") from exc
    # 2、要求每个角色只出现一次，避免后出现的条目覆盖先出现条目。
    roles = [model.role for model in manifest.models]
    if len(roles) != len(set(roles)):
        raise ModelLockError("RAG_MODEL_LOCK_INVALID")
    return manifest


def verify_model_lock(
    *, lock_path: Path, model_root: Path, expected_revisions: dict[str, str]
) -> dict[str, Path]:
    """校验 revision、相对路径、大小及 SHA-256，返回已验证模型目录。"""
    # 1、加载严格清单，并先比较部署配置要求的精确 revision。
    manifest = load_model_lock(lock_path)
    verified: dict[str, Path] = {}
    for model in manifest.models:
        if expected_revisions.get(model.role) != model.revision:
            raise ModelLockError("RAG_MODEL_REVISION_MISMATCH")
        model_directory = model_root / model.model_id.replace("/", "--")
        # 2、逐个限制为仓库内普通相对路径，再以流式方式计算文件哈希。
        for locked_file in model.files:
            relative_path = PurePosixPath(locked_file.path)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ModelLockError("RAG_MODEL_LOCK_INVALID")
            local_file = model_directory.joinpath(*relative_path.parts)
            try:
                stat = local_file.stat()
            except OSError as exc:
                raise ModelLockError("RAG_MODEL_FILE_MISSING") from exc
            if stat.st_size != locked_file.size:
                raise ModelLockError("RAG_MODEL_HASH_MISMATCH")
            digest = hashlib.sha256()
            try:
                with local_file.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
            except OSError as exc:
                raise ModelLockError("RAG_MODEL_FILE_UNREADABLE") from exc
            if digest.hexdigest() != locked_file.sha256:
                raise ModelLockError("RAG_MODEL_HASH_MISMATCH")
        # 3、仅在该角色的全部文件通过后暴露目录。
        verified[model.role] = model_directory
    return verified


def initialize_locked_models(
    *,
    lock_path: Path,
    model_root: Path,
    downloader: Callable[..., str] | None = None,
) -> dict[str, Path]:
    """只下载锁中列出的精确revision文件，并在返回前执行完整哈希校验。"""
    # 1、先解析受版本控制的锁，不从远端API生成或更新revision。
    manifest = load_model_lock(lock_path)
    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download
    # 2、每个角色下载到确定目录，allow_patterns禁止拉取未锁文件。
    expected_revisions: dict[str, str] = {}
    for model in manifest.models:
        target = model_root / model.model_id.replace("/", "--")
        downloader(
            repo_id=model.model_id,
            revision=model.revision,
            allow_patterns=[file.path for file in model.files],
            local_dir=str(target),
        )
        expected_revisions[model.role] = model.revision
    # 3、下载结果只有在大小和SHA-256全部匹配后才可用于生产启动。
    return verify_model_lock(
        lock_path=lock_path,
        model_root=model_root,
        expected_revisions=expected_revisions,
    )
