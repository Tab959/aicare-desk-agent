"""实现 DeepSeek 模型用途参数解析与 LangChain 模型创建。

本文件把强类型 Settings 转换为用途固定的 ``ModelProfile``，再创建 ``ChatDeepSeek``。
创建对象本身不会发起网络请求；实际调用发生在图节点执行 ``invoke/ainvoke`` 时。
当前显式关闭 thinking，以兼容需要结构化工具调用的既定流程。
"""

# ChatDeepSeek 是 langchain-deepseek 提供的 DeepSeek 聊天模型适配器。
from langchain_deepseek import ChatDeepSeek

# Settings 集中保存已校验的密钥、模型名、端点、超时和重试等配置。
from aicare_agent_service.config import Settings

# 导入模型配置异常、用途参数数据类和用途枚举。
from aicare_agent_service.models.contracts import (
    ModelConfigurationError,
    ModelProfile,
    ModelPurpose,
)


def resolve_model_profile(settings: Settings, purpose: ModelPurpose) -> ModelProfile:
    """根据用途生成已通过 Settings 约束的模型参数。

    ``settings`` 提供配置上限，``purpose`` 指定场景；返回不可变 ModelProfile。
    未知用途会抛出 ModelConfigurationError，而不会静默使用默认参数。
    """
    # 即使有类型标注，运行时仍可能被动态调用传入错误对象，因此显式校验枚举实例。
    if not isinstance(purpose, ModelPurpose):
        # 立即拒绝任意字符串等未受控用途。
        raise ModelConfigurationError("不支持的模型用途")

    # 集合字面量配合 ``in`` 判断三类偏确定性的任务，统一使用零温度配置。
    if purpose in {ModelPurpose.ROUTING, ModelPurpose.SUMMARY, ModelPurpose.REVIEW}:
        # 每次返回一个新的不可变用途参数对象。
        return ModelProfile(
            # 零温度降低分类、摘要和审核结果的随机漂移。
            temperature=0,
            # 这些轻量用途复用通用模型超时。
            timeout_seconds=settings.model_timeout_seconds,
            # 输出上限来自集中配置，避免节点自行写死。
            max_output_tokens=settings.model_max_output_tokens,
        )

    # ``is`` 对枚举单例做身份比较，明确匹配专业 Agent 用途。
    if purpose is ModelPurpose.SPECIALIST:
        return ModelProfile(
            # 少量温度让专业回复保留一定语言自然度。
            temperature=0.2,
            # 专业 Agent 使用独立超时预算。
            timeout_seconds=settings.specialist_timeout_seconds,
            # 专业 Agent 使用独立输出 token 上限。
            max_output_tokens=settings.specialist_max_output_tokens,
        )

    # 最终答案生成使用自己的资源预算。
    if purpose is ModelPurpose.ANSWER:
        return ModelProfile(
            # 最终回答也使用低温度，兼顾稳定性和自然表达。
            temperature=0.2,
            # 答案请求超时来自 ANSWER 专属配置。
            timeout_seconds=settings.answer_timeout_seconds,
            # 最终答案输出长度由 ANSWER 专属配置限制。
            max_output_tokens=settings.answer_max_output_tokens,
        )

    # 防御未来新增枚举成员却忘记配置 Profile 的情况。
    raise ModelConfigurationError("不支持的模型用途")


# 普通类持有 Settings，并实现 ChatModelProvider 约定的 create 方法。
class DeepSeekModelProvider:
    """使用受控Settings创建DeepSeek聊天模型，不执行网络调用。"""

    # 构造方法在 ``DeepSeekModelProvider(settings)`` 时自动执行；``-> None`` 表示不返回业务值。
    def __init__(self, settings: Settings) -> None:
        # 先取出 SecretStr；直接保存该对象不会在 repr 中暴露明文。
        api_key = settings.deepseek_api_key
        # ``or`` 短路判断同时覆盖未配置、空字符串和全空格密钥。
        if api_key is None or not api_key.get_secret_value().strip():
            # 在真正构造模型前用清晰领域异常报告缺失配置。
            raise ModelConfigurationError("模型配置缺少DEEPSEEK_API_KEY")
        # 前导下划线表示该字段仅供类内部使用，调用者不应直接修改。
        self._settings = settings

    # purpose 决定本次模型的温度、超时和输出长度；返回具体 ChatDeepSeek 实例。
    def create(self, purpose: ModelPurpose) -> ChatDeepSeek:
        """创建显式绑定模型、连接参数与用途Profile的ChatDeepSeek。"""
        # 先把用途转换为已经校验且不可变的参数集合。
        profile = resolve_model_profile(self._settings, purpose)
        # 仅实例化 LangChain 模型适配器；此处不会发送网络请求。
        return ChatDeepSeek(
            # 绑定集中配置的 DeepSeek 模型名称。
            model=self._settings.deepseek_model,
            # SecretStr 交给官方适配器用于请求鉴权。
            api_key=self._settings.deepseek_api_key,
            # HttpUrl 转为字符串后作为兼容 OpenAI 协议的 API 根地址。
            api_base=str(self._settings.deepseek_base_url),
            # 使用当前用途 Profile 中的采样温度。
            temperature=profile.temperature,
            # LangChain 参数名为 max_tokens，对应本项目 max_output_tokens。
            max_tokens=profile.max_output_tokens,
            # 单次 HTTP 请求超时秒数。
            timeout=profile.timeout_seconds,
            # 瞬时网络或限流错误的最大重试次数。
            max_retries=self._settings.deepseek_max_retries,
            # extra_body 传递 DeepSeek 特有参数；当前关闭思考模式以兼容结构化工具调用。
            extra_body={"thinking": {"type": "disabled"}},
        )
