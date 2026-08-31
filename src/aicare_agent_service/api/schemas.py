"""定义健康接口使用的 HTTP 响应数据结构。

这些 Pydantic 模型负责运行时数据校验和 OpenAPI Schema 生成，只描述接口形状，
不读取配置、不处理请求，也不执行任何外部服务检查。
"""

# ``Literal`` 把字符串类型进一步限制为列出的固定字面量，能防止任意状态文本进入响应。
from typing import Literal

# ``BaseModel`` 是 Pydantic 数据模型基类；``ConfigDict`` 用于声明模型级行为。
from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """存活接口的基础响应模型。

    语法提示：``class Child(Parent)`` 表示继承；HealthResponse 会获得 BaseModel 的校验、序列化等能力。
    """

    # ``frozen=True`` 使模型实例创建后不可修改，避免响应在传递过程中被意外篡改。
    model_config = ConfigDict(frozen=True)

    # ``Literal["UP", "DOWN"]`` 表示只能取这两个字符串之一。
    status: Literal["UP", "DOWN"]
    # ``str`` 表示必须是字符串；Pydantic 会负责输入校验。
    service: str
    # 服务版本使用字符串，以兼容语义版本和预发布后缀。
    version: str


class ReadinessChecks(BaseModel):
    """就绪探针中每一类本地检查的状态集合。"""

    # 明细模型同样冻结，保证构造后只读。
    model_config = ConfigDict(frozen=True)

    # 当前阶段只检查配置是否成功解析；后续外部依赖检查不能在不更新契约时偷偷加入。
    configuration: Literal["UP", "DOWN"]
    # RAG关闭时明确标记DISABLED；启用后由四项生产检查汇总。
    elasticsearch: Literal["UP", "DOWN", "DISABLED"]
    # 锁文件、revision、校验和、模型加载和热身状态。
    rag_models: Literal["UP", "DOWN", "DISABLED"]
    # ES集群至少达到yellow才可接收检索流量。
    elasticsearch_cluster: Literal["UP", "DOWN", "DISABLED"]
    # 版本化模板schema和Embedding指纹状态。
    index_template: Literal["UP", "DOWN", "DISABLED"]
    # 当前租户读写别名、write target和物理Mapping状态。
    aliases_mapping: Literal["UP", "DOWN", "DISABLED"]


class ReadinessResponse(HealthResponse):
    """在基础存活字段上增加就绪检查明细。

    继承意味着该类自动拥有 ``status``、``service`` 和 ``version``，这里只声明新增字段。
    """

    # 嵌套模型让 JSON 中的 ``checks`` 保持结构化，而不是使用含义模糊的字典。
    checks: ReadinessChecks
