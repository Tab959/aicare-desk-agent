"""提供 Uvicorn 等 ASGI 服务器加载的 FastAPI 应用对象。

常用入口是``uvicorn aicare_agent_service.main:app``：冒号左侧是模块路径，右侧是本文件
中的全局变量名。导入模块时调用应用工厂一次；真实模型不会在这里创建或发起网络请求。
"""

# create_app集中组装路由、配置和持久化lifespan，避免入口文件堆积业务逻辑。
from aicare_agent_service.api.app import create_app

# 模块导入时创建一个进程级FastAPI实例，ASGI服务器会把HTTP请求交给它处理。
app = create_app()
