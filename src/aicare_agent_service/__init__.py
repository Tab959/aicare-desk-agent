"""AICareDesk Python Agent 服务的顶层包信息。

该文件只公开当前Python包版本，供日志、健康检查或打包工具读取；导入本包不会创建
FastAPI应用、连接模型或访问任何基础设施。实际ASGI入口位于``aicare_agent_service.main``。
"""

# ``__all__``声明本模块稳定公开的名称；星号导入时只会导出__version__。
__all__ = ["__version__"]

# Python包版本，应与pyproject.toml中的project.version保持一致。
__version__ = "0.1.0"
