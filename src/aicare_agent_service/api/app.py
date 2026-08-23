"""创建并组装 Agent Service 的 FastAPI 应用。

该模块位于程序启动链路的最外层：读取或接收配置、执行生产配置校验、注册资源生命周期和路由。
它只负责应用装配，不在这里实现健康检查业务，也不会直接连接 PostgreSQL、Redis 或模型服务。
"""

# ``FastAPI`` 是 ASGI Web 应用类；实例最终会被 Uvicorn 等服务器加载。
from fastapi import FastAPI

# ``as`` 给导入对象起本地别名，明确这是健康检查路由，而不是整个应用。
from aicare_agent_service.api.health import router as health_router
from aicare_agent_service.api.lifecycle import application_lifespan

# ``Settings`` 用于类型标注；后两个函数分别读取配置和执行生产环境约束校验。
from aicare_agent_service.config import Settings, get_settings, validate_production_settings

# lifespan 是异步上下文管理器，负责在应用启动/关闭时持有和释放持久化资源。


def create_app(settings: Settings | None = None) -> FastAPI:
    """创建一个配置完整的 FastAPI 应用实例。

    Args:
        settings: 可选的配置对象。测试可以显式传入；传入 ``None`` 时从环境变量和 ``.env`` 读取。

    Returns:
        已注册生命周期、健康路由和 OpenAPI 元数据的 ``FastAPI`` 实例。

    Raises:
        ValueError: 生产环境缺少密钥、外部服务地址或使用了不允许的后端时抛出。

    语法提示：``Settings | None`` 是 Python 3.10+ 的联合类型，表示参数可以是两种类型之一；
    ``= None`` 则给参数设置默认值，所以调用方可以直接写 ``create_app()``。
    """
    # 条件表达式语法是 ``真值 if 条件 else 假值``。
    # 未注入配置时调用带缓存的 ``get_settings``；测试注入配置时直接复用对象。
    resolved_settings = get_settings() if settings is None else settings
    # 在创建应用和外部资源之前失败，避免生产服务带着不完整配置启动。
    validate_production_settings(resolved_settings)

    # 调用类会创建对象；以下关键字参数使每个设置的含义比按位置传参更清楚。
    application = FastAPI(
        # OpenAPI 文档中的服务标题来自统一配置。
        title=resolved_settings.service_name,
        # OpenAPI 文档和健康响应共享同一个服务版本。
        version=resolved_settings.service_version,
        # Swagger UI 的访问路径。
        docs_url="/docs",
        # OpenAPI JSON 契约的访问路径。
        openapi_url="/openapi.json",
        # FastAPI 会在启动时进入该异步上下文，并在关闭时退出它。
        lifespan=application_lifespan,
    )
    # ``state`` 是 FastAPI/Starlette 提供的应用级对象存储；资源生命周期和路由可读取同一配置。
    application.state.settings = resolved_settings
    # 注册整个健康路由器，使其中声明的三个 GET 路径真正出现在应用中。
    application.include_router(health_router)
    # ``return`` 把组装好的对象交给调用方或 ASGI 入口。
    return application
