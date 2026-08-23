"""组装供 Agent Chat UI 调试真实模型的开发图入口。

模块导入时只构建图，不立即读取 DeepSeek 密钥或创建模型；真正执行节点时才调用
``create_answer_model``。这种延迟创建方式保证缺少密钥时开发服务器仍能加载图清单。
"""

# get_settings 读取并校验应用配置；其结果会被缓存，调用者无需自行解析环境变量。
from aicare_agent_service.config import get_settings

# 导入只负责搭建最小模型调用流程的工厂函数。
from aicare_agent_service.dev.model_playground import build_model_playground_graph

# ModelPurpose 表示模型用途，create_model_provider 根据配置选择 DeepSeek 或测试替身。
from aicare_agent_service.models import ModelPurpose, create_model_provider


def create_answer_model():
    """按运行创建真实答案模型，使无 Key 时仍可加载开发服务器。

    函数没有传参；返回一个按 ANSWER 用途配置好的 LangChain 聊天模型。
    若密钥缺失，错误发生在用户真正发起调试请求时，而不是服务加载图时。
    """
    # 获取当前进程的强类型 Settings 配置对象。
    settings = get_settings()
    # 先创建配置指定的 Provider，再让 Provider 按“最终答案”用途创建模型。
    return create_model_provider(settings).create(ModelPurpose.ANSWER)


# 把模型工厂函数注入最小图；这里只传函数本身，没有加括号，因此不会立即创建模型。
graph = build_model_playground_graph(create_answer_model)
