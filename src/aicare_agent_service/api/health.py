"""提供 Agent Service 的存活与配置就绪 HTTP 探针。

该模块定义三个只读 GET 接口。存活探针不调用外部依赖；就绪探针在RAG启用后实时检查
Elasticsearch连接，失败返回503且不暴露连接、认证或异常细节。
"""

# ``cast`` 只帮助静态类型检查器理解对象类型，运行时不会转换或校验数据。
from typing import Literal, cast

from elastic_transport import TransportError

# ``APIRouter`` 用于按模块组织路由；``Request`` 表示当前 HTTP 请求及其所属应用。
from fastapi import APIRouter, Request, Response, status

# 圆括号允许把较长的导入列表分成多行，末尾逗号便于以后追加名字。
from aicare_agent_service.api.schemas import (
    # 基础存活响应模型。
    HealthResponse,
    # 就绪检查明细模型。
    ReadinessChecks,
    # 带明细的就绪响应模型。
    ReadinessResponse,
)

# 路由从应用 state 中读取这个强类型配置对象。
from aicare_agent_service.config import Settings
from aicare_agent_service.rag.model_runtime import RagRuntimeResources

# 创建路由容器；``tags`` 会把这些接口归入 OpenAPI 文档的 health 分组。
router = APIRouter(tags=["health"])


def _request_settings(request: Request) -> Settings:
    """从当前请求所属的 FastAPI 应用读取配置。

    Args:
        request: FastAPI 自动注入的当前请求对象。

    Returns:
        ``create_app`` 保存到 ``application.state.settings`` 的配置对象。

    语法提示：函数名前导下划线表示模块内部辅助函数；这只是命名约定，不是访问权限控制。
    """
    # ``request.app`` 指向处理请求的应用；``state`` 保存应用级共享对象。
    # ``cast(Settings, value)`` 不创建新对象，只告诉类型检查器这里必然是 Settings。
    return cast(Settings, request.app.state.settings)


def _health_response(settings: Settings) -> HealthResponse:
    """根据配置构造统一的存活响应。

    Args:
        settings: 当前应用配置，提供服务名和版本号。

    Returns:
        状态固定为 ``UP`` 的不可变 ``HealthResponse``。
    """
    # Pydantic 模型像普通类一样通过关键字参数实例化，并会立即校验字段类型和值域。
    return HealthResponse(
        # 代码执行到这里说明进程能够处理请求，因此存活状态为 UP。
        status="UP",
        # 服务名来自配置，避免在多个接口中重复硬编码。
        service=settings.service_name,
        # 版本同样来自配置，便于部署诊断。
        version=settings.service_version,
    )


# 装饰器在函数定义阶段把路径、HTTP 方法和响应模型登记到 router。
@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """返回通用存活探针；``async def`` 表示这是可被事件循环等待的协程函数。"""
    # 内层先取配置，外层再构造响应；Python 会从最内层函数调用开始求值。
    return _health_response(_request_settings(request))


# 兼容 Java Agent Gateway 约定的带版本前缀健康路径，响应结构与通用路径一致。
@router.get("/api/v1/agent/health", response_model=HealthResponse)
async def agent_health(request: Request) -> HealthResponse:
    """返回 Agent 内部 API 路径下的存活探针。"""
    # 复用同一辅助函数，保证两个健康路径不会产生字段差异。
    return _health_response(_request_settings(request))


# ``response_model`` 让 FastAPI 校验返回值，并用它生成 OpenAPI Schema。
@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(request: Request, response: Response) -> ReadinessResponse:
    """返回配置和启用后的Elasticsearch真实就绪状态。"""
    # 1、未启用RAG时明确返回DISABLED，不创建替代索引实现。
    settings = _request_settings(request)
    elasticsearch_status: Literal["UP", "DOWN", "DISABLED"] = "DISABLED"
    if settings.rag_enabled:
        # 2、启用后资源缺失、ping异常或ping失败均按fail-closed返回DOWN。
        resources = cast(
            RagRuntimeResources | None,
            getattr(request.app.state, "rag_resources", None),
        )
        try:
            is_elasticsearch_ready = resources is not None and await resources.elasticsearch.ping()
        except (TransportError, OSError):
            is_elasticsearch_ready = False
        elasticsearch_status = "UP" if is_elasticsearch_ready else "DOWN"
        if not is_elasticsearch_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    # 3、总状态与外部依赖状态一致，响应不携带连接或异常细节。
    return ReadinessResponse(
        status="DOWN" if elasticsearch_status == "DOWN" else "UP",
        service=settings.service_name,
        version=settings.service_version,
        checks=ReadinessChecks(configuration="UP", elasticsearch=elasticsearch_status),
    )
