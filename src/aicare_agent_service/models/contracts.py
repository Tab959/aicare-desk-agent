"""声明模型用途、固定参数和 Provider 抽象契约。

这些类型隔离业务图与 DeepSeek 等具体厂商：节点只说明“为何使用模型”，
Provider 再为对应用途创建模型。这样可集中控制温度、超时、输出长度并方便测试替换。
"""

# dataclass 适合定义仅保存数据的轻量不可变对象。
from dataclasses import dataclass

# StrEnum 的成员既是枚举又是字符串，便于配置、日志和序列化使用。
from enum import StrEnum

# Protocol 定义结构化接口；runtime_checkable 允许在运行时用 isinstance 检查该接口。
from typing import Protocol, runtime_checkable

# BaseChatModel 是所有 LangChain 聊天模型实现的共同抽象基类。
from langchain_core.language_models import BaseChatModel


# 枚举限制模型用途只能从以下固定成员中选择，避免到处传任意字符串。
class ModelPurpose(StrEnum):
    """客服工作流中的模型用途。"""

    # 低随机性的意图识别与节点路由用途。
    ROUTING = "routing"
    # 售前、售后等专业 Agent 推理用途。
    SPECIALIST = "specialist"
    # 面向顾客生成最终自然语言回答的用途。
    ANSWER = "answer"
    # 对长会话或历史内容进行压缩摘要的用途。
    SUMMARY = "summary"
    # 对候选回答做安全性、忠实度等复核的用途。
    REVIEW = "review"


# frozen=True 使参数不可修改；slots=True 减少实例开销并阻止动态增加字段。
@dataclass(frozen=True, slots=True)
class ModelProfile:
    """由用途决定且不可在运行中修改的模型参数。"""

    # 采样温度；越低通常越稳定，越高通常越有随机性。
    temperature: float
    # 单次模型请求允许等待的最大秒数。
    timeout_seconds: float
    # 模型单次回答允许生成的最大 token 数。
    max_output_tokens: int


# 继承 ValueError，表示调用者传入的模型用途或配置值不合法。
class ModelConfigurationError(ValueError):
    """模型配置缺失或不受支持。"""


# 装饰器让 Protocol 除静态类型检查外，还支持必要时使用 isinstance(..., ChatModelProvider)。
@runtime_checkable
class ChatModelProvider(Protocol):
    """按用途创建LangChain聊天模型的统一契约。"""

    # Provider 实现必须提供同名、兼容签名的方法。
    def create(self, purpose: ModelPurpose) -> BaseChatModel:
        """创建用途参数已冻结的聊天模型。

        参数 ``purpose`` 决定模型配置；返回任意符合 BaseChatModel 的实例。
        协议只规定行为，不负责创建具体对象。
        """
        # Ellipsis 表示这里只声明接口，真实方法体由 DeepSeek/Fake Provider 实现。
        ...
