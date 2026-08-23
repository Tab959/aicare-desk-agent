"""验证内部工具 OpenAPI 与 Python 工具目录保持一致。"""

import json
from pathlib import Path

from aicare_agent_service.tools.contracts import TOOL_CONTRACT_VERSION, ToolName


def test_openapi_declares_version_and_all_twenty_tool_names() -> None:
    contract_path = (
        Path(__file__).resolve().parents[2] / "resources" / "docs" / "api" / "agent-tools-v1.yaml"
    )
    document = json.loads(contract_path.read_text(encoding="utf-8"))

    assert document["info"]["version"] == TOOL_CONTRACT_VERSION
    assert set(document["components"]["schemas"]["ToolName"]["enum"]) == {
        item.value for item in ToolName
    }
    assert len(document["components"]["schemas"]["ToolArguments"]["anyOf"]) == 20
    assert document["paths"]["/api/internal/v1/agent/tools/{toolName}/invoke"]["post"]
