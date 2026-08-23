"""提供可脚本化、可流式且线程安全的 LangChain Fake 聊天模型。

单元测试可预先放入 AIMessage 或 Exception；模型按顺序消费脚本，从而稳定模拟成功、
失败、同步调用、异步调用和 token 流式输出。该实现不会连接 DeepSeek，也不应在非测试
环境启用。``FakeModelProvider`` 为不同模型用途保存互相隔离的脚本。
"""

# 异步/同步迭代器描述 yield 返回方式；Callable、Mapping、Sequence 描述工具和脚本容器接口。
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence

# Lock 是互斥锁，防止多个线程同时消费同一个脚本游标。
from threading import Lock

# Any 表示此处需要兼容第三方 LangChain 方法签名中的多种参数类型。
from typing import Any

# 两类回调管理器分别服务异步和同步 LLM 运行，用于把流式 token 通知给 LangChain。
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)

# BaseChatModel 是自定义聊天模型需继承的抽象基类。
from langchain_core.language_models import BaseChatModel

# AIMessage 是完整回复，AIMessageChunk 是流式片段，BaseMessage 是所有输入消息的父类型。
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

# ChatResult/ChatGeneration 表示完整结果，ChatGenerationChunk 表示单个流式生成片段。
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

# Runnable 是 LangChain 可 invoke/ainvoke 对象的统一接口，bind_tools 需要返回它。
from langchain_core.runnables import Runnable

# BaseTool 是 LangChain 工具基类，用于保持 bind_tools 的官方兼容签名。
from langchain_core.tools import BaseTool

# PrivateAttr 声明不参与 Pydantic 字段校验和序列化的模型内部状态。
from pydantic import PrivateAttr

# Fake 模型复用统一的配置异常与模型用途枚举。
from aicare_agent_service.models.contracts import ModelConfigurationError, ModelPurpose

# 类型别名：一个脚本项只能是预制 AI 回复或准备在调用时抛出的异常。
ScriptedItem = AIMessage | Exception


# RuntimeError 表示配置本来有效，但运行时已经没有下一条脚本响应可消费。
class FakeModelScriptExhaustedError(RuntimeError):
    """确定性Fake模型没有剩余响应。"""


# 前导下划线表示该辅助函数仅供本模块内部使用。
def _normalize_script(script: Sequence[ScriptedItem]) -> tuple[ScriptedItem, ...]:
    """把只读脚本序列冻结为元组，并运行时校验每个脚本项。

    ``script`` 可为列表或其他 Sequence；返回不可变元组。非法元素会抛出
    ModelConfigurationError，防止测试运行到一半才出现难以理解的类型错误。
    """
    # tuple(...) 创建独立不可变快照，之后调用者修改原列表不会影响模型脚本。
    normalized = tuple(script)
    # 生成器表达式逐项检查；any 遇到第一个非法项就短路返回 True。
    if any(not isinstance(item, (AIMessage, Exception)) for item in normalized):
        # 第二个 isinstance 参数是类型元组，表示满足任一类型即可。
        raise ModelConfigurationError("Fake模型脚本项仅支持AIMessage或Exception")
    # 返回完成校验且可安全复用的元组。
    return normalized


# 继承 BaseChatModel 后，只需实现约定的底层方法即可自动获得 invoke/ainvoke 等公共 API。
class ScriptedFakeChatModel(BaseChatModel):
    """按顺序消费消息或异常的确定性LangChain聊天模型测试替身。"""

    # PrivateAttr() 表示脚本是内部状态，不应出现在模型公开字段或序列化结果中。
    _script: tuple[ScriptedItem, ...] = PrivateAttr()
    # default=0 让每个模型实例从脚本第一项开始消费。
    _cursor: int = PrivateAttr(default=0)
    # default_factory 为每个实例新建一把锁，避免所有模型错误共享同一把全局锁。
    _script_lock: Lock = PrivateAttr(default_factory=Lock)

    # **kwargs 收集并透传 BaseChatModel 可能需要的任意关键字参数。
    def __init__(self, script: Sequence[ScriptedItem], **kwargs: Any) -> None:
        """创建脚本模型。

        ``script`` 是按调用顺序消费的响应/异常序列；``**kwargs`` 是命名参数字典，
        原样交给 LangChain 基类。构造函数不返回业务值。
        """
        # 必须先初始化 Pydantic/LangChain 基类，再写入 PrivateAttr。
        super().__init__(**kwargs)
        # 校验并冻结脚本，防止外部后续修改原序列影响测试确定性。
        self._script = _normalize_script(script)

    # property 让方法可像只读属性一样访问，LangChain 用它识别模型实现类型。
    @property
    def _llm_type(self) -> str:
        """返回供 LangChain 追踪和序列化使用的稳定模型类型名。"""
        # 固定名称避免把密钥、脚本或动态数据写入追踪标识。
        return "aicare-scripted-fake-chat-model"

    # 该 property 返回区分模型实例配置的安全公开参数。
    @property
    def _identifying_params(self) -> dict[str, Any]:
        """返回用于追踪识别的非敏感参数字典。"""
        # Fake 模型没有真实厂商参数，只暴露固定模型名。
        return {"model_name": "aicare-scripted-fake-chat-model"}

    def _consume(self) -> AIMessage:
        """线程安全地消费下一项脚本，必要时抛出预制异常。

        返回深拷贝 AIMessage，确保调用者修改结果不会污染保存在 Provider 中的原脚本。
        """
        # ``with`` 自动获取并最终释放互斥锁，即使代码抛异常也不会忘记解锁。
        with self._script_lock:
            # 游标等于脚本长度时已经越过最后一个合法索引。
            if self._cursor >= len(self._script):
                # 使用专用异常让测试可精确断言“脚本耗尽”而非普通 IndexError。
                raise FakeModelScriptExhaustedError("Fake模型脚本已耗尽")
            # 先读取当前项，再移动游标，保证每项只被消费一次。
            item = self._script[self._cursor]
            self._cursor += 1

        # 锁外处理结果，缩短临界区；Exception 脚本项用于确定性模拟模型失败。
        if isinstance(item, Exception):
            # 抛出脚本保存的原异常，调用者可验证上层重试和错误处理。
            raise item
        # deep=True 连嵌套消息内容也复制，避免不同调用之间共享可变对象。
        return item.model_copy(deep=True)

    # staticmethod 不接收 self，因为结果包装逻辑不依赖模型实例状态。
    @staticmethod
    def _result(message: AIMessage) -> ChatResult:
        """把一条完整 AIMessage 包装成 LangChain 标准 ChatResult。"""
        # 一次调用只有一个候选生成结果，因此 generations 列表只含一项。
        return ChatResult(generations=[ChatGeneration(message=message)])

    # 同样是与实例无关的纯转换函数。
    @staticmethod
    def _message_chunks(message: AIMessage) -> tuple[AIMessageChunk, ...]:
        """把完整消息拆成可供同步/异步流式接口复用的不可变片段元组。"""
        # content 可能是普通字符串，也可能是 LangChain 的结构化内容块列表。
        content = message.content
        # 此局部变量标注说明每个 token 可为字符，或一个结构化内容块列表。
        tokens: list[str | list[str | dict[str, Any]]]
        # 字符串按字符拆分，以便测试能观察多次流式回调。
        if isinstance(content, str):
            # 空字符串会得到空列表，所以用 ``or [""]`` 确保仍产生一个最终 chunk。
            tokens = list(content) or [""]
        else:
            # 结构化内容保持整体，不尝试破坏其内容块边界。
            tokens = [content]

        # 先用可变列表累积片段，完成后再冻结为 tuple。
        chunks: list[AIMessageChunk] = []
        # 最终索引用于只在最后一个片段附带元数据和工具调用。
        final_index = len(tokens) - 1
        # enumerate 同时提供从零开始的索引和当前 token。
        for index, token in enumerate(tokens):
            # 布尔值标识当前片段是否为最后一项。
            is_final = index == final_index
            # append 把新构造的消息片段追加到结果列表末尾。
            chunks.append(
                AIMessageChunk(
                    # 当前字符或结构化内容块成为本次流式片段正文。
                    content=token,
                    # 所有片段沿用完整消息 ID，便于 LangChain 正确合并。
                    id=message.id,
                    # 保留可选消息名称。
                    name=message.name,
                    # 非正文元数据只放在最终片段，避免合并时重复。
                    additional_kwargs=message.additional_kwargs if is_final else {},
                    # 模型响应元数据也只附在最后片段。
                    response_metadata=message.response_metadata if is_final else {},
                    # 工具调用必须完整出现，因此只放最后片段。
                    tool_calls=message.tool_calls if is_final else [],
                    # 无效工具调用诊断信息同样只放最后片段。
                    invalid_tool_calls=message.invalid_tool_calls if is_final else [],
                    # token 用量统计若存在，只在流结束时输出一次。
                    usage_metadata=message.usage_metadata if is_final else None,
                    # LangChain 用 "last" 明确标记最终 chunk；其他片段不设置。
                    chunk_position="last" if is_final else None,
                )
            )
        # tuple 防止调用者修改已经构造好的片段序列。
        return tuple(chunks)

    def _generate(
        # messages 是调用者输入的历史消息；Fake 模型不读取内容，因为结果由脚本决定。
        self,
        messages: list[BaseMessage],
        # stop 是可选停止词列表；为兼容 BaseChatModel 签名而保留。
        stop: list[str] | None = None,
        # run_manager 是同步回调管理器；非流式脚本生成无需主动通知 token。
        run_manager: CallbackManagerForLLMRun | None = None,
        # kwargs 接收 LangChain 可能传入的额外模型参数。
        **kwargs: Any,
    ) -> ChatResult:
        """实现 BaseChatModel 的同步完整生成底层接口。"""
        # del 明确标记这些兼容参数有意不用，同时避免静态检查报告未使用变量。
        del messages, stop, run_manager, kwargs
        # 消费一条脚本消息并包装成标准完整结果。
        return self._result(self._consume())

    # async 方法满足 LangChain 异步调用路径；脚本本身在内存中，无需真实网络 await。
    async def _agenerate(
        # self 由实例方法调用自动传入。
        self,
        # 输入消息保留以匹配官方接口，但不会影响确定性脚本输出。
        messages: list[BaseMessage],
        # 可选停止词参数保持与真实聊天模型一致。
        stop: list[str] | None = None,
        # 异步回调管理器在非流式完整生成中无需使用。
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        # 收集其他关键字参数以兼容上层 Runnable 调用。
        **kwargs: Any,
    ) -> ChatResult:
        """实现 BaseChatModel 的异步完整生成底层接口。"""
        # 显式删除未使用的兼容参数。
        del messages, stop, run_manager, kwargs
        # Fake 脚本读取无需 I/O，可直接返回标准结果；async 函数仍会返回协程对象。
        return self._result(self._consume())

    def _stream(
        # self 指向当前拥有独立脚本游标的 Fake 模型实例。
        self,
        # 历史消息不参与脚本选择，但签名必须兼容 BaseChatModel。
        messages: list[BaseMessage],
        # 同步流式路径也接受可选停止词。
        stop: list[str] | None = None,
        # 非空时，每产生一个片段就通过它通知 LangChain 回调系统。
        run_manager: CallbackManagerForLLMRun | None = None,
        # 捕获其他调用参数，确保测试替身可代替真实模型。
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """同步逐片段产出脚本消息，并发送 token 回调。

        返回类型 Iterator 表示函数通过 ``yield`` 分多次交付 ChatGenerationChunk，
        而不是一次返回完整列表。
        """
        # 这些参数只是为了接口兼容，脚本生成无需读取它们。
        del messages, stop, kwargs
        # 先消费一条完整脚本项，再把它拆成消息片段逐个遍历。
        for message_chunk in self._message_chunks(self._consume()):
            # LangChain 的流式输出外层要求 ChatGenerationChunk，而不是裸 AIMessageChunk。
            chunk = ChatGenerationChunk(message=message_chunk)
            # 回调管理器是可选的，调用前必须排除 None。
            if run_manager is not None:
                # 把片段内容转成字符串通知观察者，并附上完整 chunk 对象。
                run_manager.on_llm_new_token(str(message_chunk.content), chunk=chunk)
            # yield 暂停函数并把当前片段交给调用者，下次迭代再从这里继续。
            yield chunk

    # async generator 同时使用 async def 和 yield，调用者需要 ``async for`` 消费。
    async def _astream(
        # self 指向当前模型实例。
        self,
        # 输入消息仅用于满足官方异步流签名。
        messages: list[BaseMessage],
        # 可选停止词参数。
        stop: list[str] | None = None,
        # 异步回调管理器的方法本身也需要 await。
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        # 其他兼容关键字参数。
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """异步逐片段产出脚本消息，并等待 token 回调完成。"""
        # 删除不参与脚本逻辑的参数。
        del messages, stop, kwargs
        # 片段拆分是内存同步操作，但对外暴露为异步迭代器以模拟真实模型。
        for message_chunk in self._message_chunks(self._consume()):
            # 包装为 LangChain 统一生成片段。
            chunk = ChatGenerationChunk(message=message_chunk)
            # 没有回调管理器时跳过通知，仍正常产出片段。
            if run_manager is not None:
                # await 确保异步观察者处理完当前 token 后再继续。
                await run_manager.on_llm_new_token(str(message_chunk.content), chunk=chunk)
            # 异步生成器的 yield 把片段交给 ``async for`` 调用者。
            yield chunk

    def bind_tools(
        # self 是当前 Fake 模型；方法最终返回自身以继续参与 Runnable 链。
        self,
        # tools 支持字典 schema、类、Callable 或 BaseTool，与 LangChain 官方签名一致。
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        # 单独的 ``*`` 表示后续参数必须用名称传入，不能按位置传入。
        *,
        # tool_choice 可指定工具选择策略；Fake 模型不据此动态生成调用。
        tool_choice: str | None = None,
        # 收集未来版本可能增加的绑定选项。
        **kwargs: Any,
    ) -> Runnable:
        """声明工具兼容性；工具调用内容仍完全来自测试脚本。

        返回当前模型本身，因为它已经是 Runnable。参数只用于保持接口兼容，
        工具调用的名称和参数必须预先写在脚本 AIMessage 中。
        """
        # Fake 不执行真实工具绑定，显式删除参数以表达这是有意的测试行为。
        del tools, tool_choice, kwargs
        # 返回 self 让上层继续调用 invoke、stream 等标准 Runnable API。
        return self


# Provider 负责按用途保存脚本，而具体模型实例负责独立消费脚本。
class FakeModelProvider:
    """按模型用途保存独立脚本，并为每次创建返回全新消费游标。"""

    def __init__(
        # scripts 的键是模型用途，值是该用途按顺序消费的脚本；整个参数可以省略。
        self,
        scripts: Mapping[ModelPurpose, Sequence[ScriptedItem]] | None = None,
    ) -> None:
        """创建 Fake Provider 并冻结每个用途的脚本。

        ``scripts=None`` 表示暂未配置任何用途；之后调用 create 会得到明确配置错误。
        每个用途的脚本都转换成元组，避免外部列表修改影响测试。
        """
        # 建立内部字典；键和值类型都被明确标注，前导下划线表示内部实现字段。
        self._scripts: dict[ModelPurpose, tuple[ScriptedItem, ...]] = {}
        # ``scripts or {}`` 把 None 视为空字典；items() 逐对取得用途和脚本。
        for purpose, script in (scripts or {}).items():
            # 运行时拒绝普通字符串等未经枚举约束的键。
            if not isinstance(purpose, ModelPurpose):
                # 统一使用模型配置异常，便于调用者捕获和展示。
                raise ModelConfigurationError("不支持的模型用途")
            # 校验脚本项并把脚本冻结为不可变元组。
            self._scripts[purpose] = _normalize_script(script)

    # 返回具体 ScriptedFakeChatModel，同时满足 ChatModelProvider 协议的 BaseChatModel 返回约束。
    def create(self, purpose: ModelPurpose) -> ScriptedFakeChatModel:
        """为指定用途创建不会与其他用途共享消费游标的 Fake 模型。

        ``purpose`` 必须是 ModelPurpose；返回一个游标从零开始的新模型实例。
        用途无脚本时抛出 ModelConfigurationError，不会伪造默认答案。
        """
        # 防御动态调用时绕过静态类型检查传入非法用途。
        if not isinstance(purpose, ModelPurpose):
            raise ModelConfigurationError("不支持的模型用途")
        # dict.get 在键不存在时返回 None，便于给出领域明确错误。
        script = self._scripts.get(purpose)
        # 空元组是合法但会在首次调用时报脚本耗尽；只有 None 表示完全未配置该用途。
        if script is None:
            # f-string 用花括号把枚举的字符串值嵌入错误消息。
            raise ModelConfigurationError(f"未配置Fake模型脚本：{purpose.value}")
        # 构造新模型会创建新的游标和锁，因此多次 create 互不影响。
        return ScriptedFakeChatModel(script)
