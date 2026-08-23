"""模型抽象层的统一公开入口。

本包把模型用途、Provider 协议、DeepSeek 实现、Fake 测试替身和工厂函数集中导出。
业务节点应依赖这里的抽象，而不是自行读取密钥或直接实例化具体厂商模型。
"""

# 导入所有模型实现共同依赖的用途枚举、参数配置、异常和 Provider 协议。
from aicare_agent_service.models.contracts import (
    ChatModelProvider,
    ModelConfigurationError,
    ModelProfile,
    ModelPurpose,
)

# DeepSeekModelProvider 是当前生产模型提供者实现。
from aicare_agent_service.models.deepseek import DeepSeekModelProvider

# 工厂函数根据 Settings 选择并构造允许的 Provider。
from aicare_agent_service.models.factory import create_model_provider

# Fake 模型相关类型只用于确定性单元测试和离线开发验证。
from aicare_agent_service.models.fake import (
    FakeModelProvider,
    FakeModelScriptExhaustedError,
    ScriptedFakeChatModel,
)

# ``__all__`` 明确 models 包稳定公开的名称，避免调用者依赖内部实现细节。
__all__ = [
    "ChatModelProvider",
    "DeepSeekModelProvider",
    "FakeModelProvider",
    "FakeModelScriptExhaustedError",
    "ModelConfigurationError",
    "ModelProfile",
    "ModelPurpose",
    "ScriptedFakeChatModel",
    "create_model_provider",
]
