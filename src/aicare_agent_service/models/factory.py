"""根据应用配置选择模型 Provider 的唯一工厂入口。

业务代码只调用 ``create_model_provider``，不直接分支创建 DeepSeek 或 Fake Provider。
工厂同时执行环境门禁，防止生产/开发环境意外启用只供测试使用的确定性 Fake 模型。
"""

# Environment、ModelProviderName 是受控枚举；Settings 是已完成校验的应用配置。
from aicare_agent_service.config import Environment, ModelProviderName, Settings

# Provider 协议作为统一返回类型，配置错误则使用明确领域异常。
from aicare_agent_service.models.contracts import (
    ChatModelProvider,
    ModelConfigurationError,
)

# 导入生产 DeepSeek Provider 和测试 Fake Provider 的具体实现。
from aicare_agent_service.models.deepseek import DeepSeekModelProvider
from aicare_agent_service.models.fake import FakeModelProvider


def create_model_provider(settings: Settings) -> ChatModelProvider:
    """根据受控配置创建模型 Provider，不读取模型生成的 Provider 名称。

    参数是已校验 Settings，返回满足 ChatModelProvider 协议的实例。
    不支持或不安全的配置会抛出 ModelConfigurationError。
    """
    # 枚举成员是单例，使用 ``is`` 明确判断配置要求 DeepSeek。
    if settings.model_provider is ModelProviderName.DEEPSEEK:
        # DeepSeek Provider 构造时会进一步检查 API Key 是否有效存在。
        return DeepSeekModelProvider(settings)

    # Fake Provider 只允许显式配置，不作为任何环境的隐式降级方案。
    if settings.model_provider is ModelProviderName.FAKE:
        # 二次校验运行环境，避免测试替身在开发或生产环境伪造真实回答。
        if settings.environment is not Environment.TEST:
            raise ModelConfigurationError("Fake模型Provider仅允许测试环境")
        # 测试环境可返回尚未配置脚本的 Provider，由具体测试按用途注入响应。
        return FakeModelProvider()

    # 防御配置枚举未来扩展但工厂尚未实现对应 Provider 的情况。
    raise ModelConfigurationError("不支持的模型Provider配置")
