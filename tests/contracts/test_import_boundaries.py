"""验证契约与图运行时在全新进程中不存在依赖导入顺序的循环。"""

import subprocess
import sys


def test_graph_context_can_be_imported_first_in_a_fresh_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from aicare_agent_service.graph.context import AgentRuntimeContext",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
