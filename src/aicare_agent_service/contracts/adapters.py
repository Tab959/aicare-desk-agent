"""把 Java/Python 线请求转换为 LangGraph 可执行的内部初始状态。

适配器是传输层与图状态之间的边界：先校验契约版本和完整请求，再建立不可伪造的身份、
LangChain 消息、业务上下文及安全历史。它不调用模型、工具、数据库，也不生成客服回复。
"""

# ``Mapping`` 表示只要求对象具备键值读取能力，比强制传入 dict 更通用。
from collections.abc import Mapping

# HumanMessage 是 LangChain 表示用户消息的标准消息对象。
from langchain_core.messages import HumanMessage

# 严格请求模型负责把普通映射校验为可信结构。
from aicare_agent_service.contracts.agent_run import AgentRunRequest

# 当前服务唯一支持的共享契约版本。
from aicare_agent_service.contracts.common import CONTRACT_HEADER_VERSION

# 安全历史需要固定角色枚举和冻结消息模型。
from aicare_agent_service.contracts.decisions import MessageRole, SafeConversationMessage

# 图状态中的身份与整体状态类型定义在 graph 层。
from aicare_agent_service.graph.state import AgentIdentity, CustomerServiceState
from aicare_agent_service.security.policy import assess_user_input


class UnsupportedContractVersionError(ValueError):
    """请求使用了当前服务未显式支持的共享契约版本。

    继承 ``ValueError`` 表示“值的类型是字符串，但内容不受支持”，调用层可据此返回稳定的契约错误。
    """


def adapt_run_request(
    contract_version: str,
    payload: Mapping[str, object],
    *,
    input_max_chars: int = 8000,
) -> CustomerServiceState:
    """把严格 v1 线请求转换为 LangGraph 初始状态。

    Args:
        contract_version: Java 从 ``X-Contract-Version`` 请求头传入的版本字符串。
        payload: 已解析 JSON 对象；键必须为 camelCase，值在进入图前会由 Pydantic 严格校验。

    Returns:
        包含身份、LangChain 消息、业务上下文和安全历史的 ``CustomerServiceState``。

    Raises:
        UnsupportedContractVersionError: 版本不是当前固定版本时抛出。
        pydantic.ValidationError: 请求缺字段、字段类型错误或含未知字段时抛出。
    """
    # ``!=`` 表示不等于；先检查版本可以避免用错误 Schema 解释请求正文。
    if contract_version != CONTRACT_HEADER_VERSION:
        # f-string 使用 ``{表达式}`` 把运行时值插入字符串，错误中只包含版本，不包含用户正文。
        raise UnsupportedContractVersionError(f"不支持的Agent契约版本：{contract_version}")

    # 1、校验共享契约版本和Java请求结构，身份字段只取自可信请求。
    request = AgentRunRequest.model_validate(payload)
    # 2、在创建任何LangGraph状态或消息对象前完成确定性安全判定和脱敏。
    assessment = assess_user_input(request.user_message, max_chars=input_max_chars)
    safe_message = assessment.sanitized_text
    # 3、只把安全文本写入消息、历史和可持久化安全状态。
    return CustomerServiceState(
        # 身份只能从已校验请求构造，不允许模型生成 tenant/customer/run 等字段。
        identity=AgentIdentity.from_request(request),
        # 方括号创建 list；首轮只放入触发本次 run 的用户消息。
        messages=[
            # LangChain HumanMessage 让后续模型节点识别这是用户角色消息。
            HumanMessage(
                # 复用 Java messageId，便于追踪且避免另造消息身份。
                id=request.trigger_message_id,
                # 消息正文进入本次图状态供模型理解，但不会写入 Redis run ledger。
                content=safe_message,
            )
        ],
        # 直接复用冻结的业务上下文对象，不复制 Java 业务逻辑。
        business_context=request.business_context,
        # safe_history 是内部可裁剪的历史窗口；首轮包含当前顾客消息。
        safe_history=[
            # 使用专门模型保证历史项只有允许的四个字段。
            SafeConversationMessage(
                # 历史项与原始 Java 消息保持相同 ID。
                message_id=request.trigger_message_id,
                # 序号用于恢复时保持 Java 会话顺序。
                sequence=request.trigger_sequence,
                # 枚举值明确标记消息来自顾客。
                role=MessageRole.CUSTOMER,
                # 保存当前图需要处理的文本。
                content=safe_message,
            ),
        ],
        sanitized_user_message=safe_message,
        input_safety_assessment=assessment,
        classification_failure=None,
    )
