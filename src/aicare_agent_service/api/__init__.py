"""API 包的公开入口。

这个文件把创建 FastAPI 应用所需的 ``create_app`` 函数重新导出。
调用方因此可以从 ``aicare_agent_service.api`` 导入它，而不必知道函数实际位于 ``app.py``。
"""

# ``from ... import ...`` 会把另一个模块中的名字绑定到当前模块。
# 这里使用完整包路径，避免相对导入层级变化时产生歧义。
from aicare_agent_service.api.app import create_app

# ``__all__`` 声明本包希望公开的名字；执行 ``from ... import *`` 时只会导出这些名字。
# 项目通常不推荐星号导入，但该列表也能作为清晰的公共 API 清单。
__all__ = ["create_app"]
