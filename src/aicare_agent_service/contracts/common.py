"""定义 Java/Python 线契约共同使用的常量、约束类型和模型基类。

“线契约”指通过 HTTP/NDJSON 传输的 JSON 结构。本模块统一规定版本头、非空文本、正整数序号、
camelCase 字段名、不可变模型和未知字段拒绝规则，其他 contracts 文件都建立在这些基础上。
"""

# ``Annotated`` 可以给原始类型附加校验元数据，而不改变它在类型检查器眼中的基础类型。
from typing import Annotated

# BaseModel 提供校验/序列化；ConfigDict 配置模型；Field 和 StringConstraints 描述字段约束。
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# ``to_camel`` 把 snake_case（如 run_id）转换为 camelCase（如 runId）。
from pydantic.alias_generators import to_camel

# Java 与 Python 通过这个 HTTP 请求头声明共享契约版本。
CONTRACT_HEADER_NAME = "X-Contract-Version"
# 当前只支持版本 1；字符串类型与 HTTP Header 的文本语义一致。
CONTRACT_HEADER_VERSION = "1"

# 类型别名：运行时基础类型仍是 str，但 Pydantic 会先去除首尾空白，再要求至少一个字符。
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
# ``strict=True`` 禁止把字符串 "1" 自动转成整数；``ge=1`` 表示 greater than or equal to 1。
PositiveSequence = Annotated[int, Field(strict=True, ge=1)]


class WireContractModel(BaseModel):
    """所有 Java/Python JSON 线模型的公共父类。

    子类只需声明 Python 风格的 snake_case 字段；这个基类会要求线上 JSON 使用 camelCase。
    """

    # 模型配置是类级字段，Pydantic 在创建子类时读取它；它不属于业务 JSON。
    model_config = ConfigDict(
        # 自动为每个 snake_case 字段生成 camelCase 输入/输出别名。
        alias_generator=to_camel,
        # 输入出现模型没有声明的字段时立即报错，防止契约悄悄漂移。
        extra="forbid",
        # 实例构造后不能修改字段，避免事件或请求身份在处理中被篡改。
        frozen=True,
        # 允许使用别名（例如 runId）进行校验。
        validate_by_alias=True,
        # 不允许线请求用 Python 字段名（例如 run_id）绕过 camelCase 契约。
        validate_by_name=False,
        # 校验异常只显示字段和错误类型，禁止把畸形消息中的秘密渲染到日志或响应文本。
        hide_input_in_errors=True,
    )
