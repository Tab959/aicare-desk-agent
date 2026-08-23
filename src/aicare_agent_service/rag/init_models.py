"""提供离线模型初始化命令；生产应用生命周期本身绝不下载模型。"""

import argparse
from pathlib import Path

from aicare_agent_service.rag.model_lock import initialize_locked_models


def main() -> None:
    """解析显式锁和目标目录并初始化全部锁定模型文件。"""
    # 1、要求部署者显式指定受版本控制的锁和持久模型目录。
    parser = argparse.ArgumentParser(description="Initialize locked AICare BGE models")
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    arguments = parser.parse_args()
    # 2、下载精确revision与文件白名单并执行完整哈希校验。
    initialized = initialize_locked_models(
        lock_path=arguments.lock,
        model_root=arguments.model_dir,
    )
    # 3、只输出初始化角色，不打印本地绝对路径或远端认证信息。
    print("initialized=" + ",".join(sorted(initialized)))


if __name__ == "__main__":
    main()
