"""开发调试入口包。

这个文件把 ``dev.graph_entry`` 中已经编译好的最小 LangGraph 图重新导出，
使 LangGraph 开发服务器可以通过稳定的包路径找到 ``graph``。这里不包含生产业务逻辑。
"""

# 从同一开发包的 graph_entry 模块导入已编译图；导入模块时会完成图的构建与编译。
from aicare_agent_service.dev.graph_entry import graph

# ``__all__`` 声明使用 ``from ... import *`` 时允许公开的名称，也用于明确本包公共接口。
__all__ = ["graph"]
