# AICare Agent Service

Stage 4的Python Agent服务。当前工程提供可锁定依赖、统一Settings、FastAPI健康探针、DeepSeek Provider、PostgreSQL Checkpointer、Redis RunStore、可恢复run生命周期，以及安全预处理、结构化分类、固定路由、输出门禁和终态封装组成的正式LangGraph根图；同时已提供Java只读工具生产客户端、20个固定LangChain工具和专业Agent最小权限能力包，专业子图与RAG将在后续Task接入。

## 边界

- Java `platform-api`是统一会话网关和业务事实来源，负责鉴权、WebSocket、消息持久化、会话状态与业务事务。
- `conversationId`由Java创建并传给Python；Python只把它映射为LangGraph `thread_id`。
- Python只负责路由、编排、受控工具调用、RAG检索和消息生成，不直接访问Java MySQL，也不修改订单、库存、余额、权益、工单或会话状态。
- LangGraph Studio、LangSmith和Agent Chat UI只用于开发调试与评测。C/B端生产浏览器始终连接Java WebSocket，不能直连Python或本地LangGraph Server。

## 安装

要求Python 3.11至3.13。以下命令在本目录执行：

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "uv==0.12.3"
.\.venv\Scripts\uv.exe sync --locked --group dev
```

`uv.lock`是唯一依赖锁文件。不要使用未锁定的`pip install -e ".[dev]"`替代上述同步流程。

### 根目录关键文件说明

- `pyproject.toml`：人工维护的Python项目元数据、直接依赖、开发依赖和pytest/Ruff配置；TOML支持`#`注释。
- `uv.lock`：`uv`根据`pyproject.toml`自动生成的完整传递依赖锁，包含不同平台的包、版本、来源与哈希。不要手工添加注释或修改内容；依赖变化后运行`python -m uv lock`重新生成，并用`python -m uv lock --check`校验。
- `langgraph.json`：LangGraph CLI读取的严格JSON配置。JSON标准不支持注释，因此不要在文件内加入`//`或`#`。当前`dependencies=["."]`表示从本项目安装依赖；`graphs`把两个Graph ID映射到Python文件中的已编译`graph`变量；`env=".env"`表示`langgraph dev`加载本地私有配置。
- `.env`：当前机器唯一的运行配置文件，被Git忽略。每个部署环境必须通过安全配置管理自行提供完整`.env`或等价进程环境变量。
- `src/aicare_agent_service/main.py`：Uvicorn加载的ASGI入口，只调用`create_app()`组装FastAPI。
- `src/aicare_agent_service/config.py`：配置事实入口，负责`.env`/环境变量读取、类型转换、范围约束和生产门禁。
- `src/aicare_agent_service/__init__.py`：只公开Python包版本，不启动应用。

## 配置

只通过Git忽略的`.env`或部署平台的环境变量提供配置。`.env`不得提交、截图或复制到公开材料；仓库不提供包含占位值的env模板。生产环境至少需要：

- `DEEPSEEK_API_KEY`
- `AICARE_AGENT_POSTGRES_DSN`：只保存Agent Checkpoint与可恢复运行状态，不是Java业务库。
- `AICARE_AGENT_JAVA_BASE_URL`
- `AICARE_AGENT_JAVA_SERVICE_TOKEN`

当前私有`.env`显式列出所有已知配置变量；生产环境不会依赖PostgreSQL、Redis、AES、Provider、超时和保留期等安全关键默认值。Elasticsearch与真实基础设施测试变量允许保持空：前者尚未部署且RAG适配尚未实现，后者仅用于显式pytest测试，不能指向生产数据。

默认Provider为`deepseek`，默认模型为`deepseek-v4-pro`；私有`.env`仍显式设置模型ID，避免生产行为依赖代码默认值：

```dotenv
AICARE_AGENT_DEEPSEEK_MODEL=deepseek-v4-pro
```

生产环境的DeepSeek Base URL必须使用HTTPS。普通模型调用默认超时30秒、最大输出2048 Token；专用Agent与最终答案分别允许使用独立的超时和最大输出Token配置。重试次数限制为0至5，单次最大输出限制为1至8192 Token。DeepSeek Provider只负责按用途构造模型，不会在应用启动或健康检查阶段发送模型请求。

当前Provider统一向DeepSeek请求发送`thinking.type=disabled`，因此Pro运行非思考模式，以兼容当前结构化输出/工具调用链。V4 Pro与Flash共用Base URL和调用协议，但能力、延迟、并发额度和费用并不相同。当前只配置一个全局模型ID，路由/摘要/审核、专业Agent和最终回答都会使用Pro；若后续要Flash处理轻任务、Pro处理复杂任务，需要增加按`ModelPurpose`分配模型ID的配置，而不是只改一个变量。

### 当前关键运行链路

当前已经实际接线的链路：

1. Uvicorn导入`aicare_agent_service.main:app`，`main.py`调用`create_app()`创建FastAPI应用。
2. `create_app()`调用`get_settings()`读取进程环境和`.env`，执行生产配置门禁，并注册健康路由与持久化lifespan。
3. lifespan创建LangGraph Checkpointer；生产环境通过三个Sentinel发现当前Redis master，再执行生产就绪审计并创建`RedisRunStore`和`AgentRunLifecycle`；随后创建唯一进程级HTTPX/Java工具客户端。退出时按Java HTTPX→Redis→Checkpointer的相反顺序关闭资源。
4. FastAPI目前只对外提供健康/就绪探针；尚没有Java调用的正式Agent生成Controller或流式端点。
5. Agent Server根据`langgraph.json`加载正式`customer_service`根图和独立`model_playground`；Server通过自定义资源注入同一套加密PostgreSQL Checkpointer。专业子图尚未交付时，正式根图对专业路由明确失败并交给run生命周期层处理，不生成伪业务回答。

已经实现并通过测试、但要等待正式Agent Gateway端点和根客服图接线的核心链路：

1. Java创建`conversationId`、`runId`和触发消息身份后调用Python；`conversationId`原样成为LangGraph `thread_id`，`runId`只标识本轮生成任务。
2. `AgentRunLifecycle.execute()`对请求计算规范摘要，由Redis Lua原子判定首次执行、过期恢复、正在执行、完成重放或请求冲突。
3. 首次执行把Java请求裁剪成安全状态；恢复执行调用`ainvoke(None)`沿同一thread最新checkpoint继续；完成重放只读取记录的checkpoint，不再次调用模型。
4. 图任务运行期间生命周期按周期检查取消、续租并发出`RUN_HEARTBEAT`；总超时提交固定`MODEL_TIMEOUT`失败。Redis故障或lease丢失不会伪装成模型失败。
5. 图完成后读取最终StateSnapshot，要求最终回答、转人工建议、工单升级建议三者恰好一个；然后取得checkpoint ID并计算忽略临时`eventIndex`的终态摘要。
6. Redis先原子提交COMPLETED/checkpoint/摘要并释放lease，成功后Python才把唯一终态事件交给Java。Java再次核对run和会话状态，持久化完整AI消息或执行结构化建议。
7. Redis只保留run协调元数据；PostgreSQL保存加密checkpoint；Java/MySQL仍是会话、订单、权益和工单的业务事实来源。

生产Redis必须专供RunStore使用，并满足：Redis 7+、已开启AOF、`appendfsync` 为 `everysec` 或 `always`、`maxmemory-policy=noeviction`、可执行 `ACL WHOAMI` 且使用非 `default` 的专用ACL用户，并且所连可写master具备至少一个副本。当前检测到Cluster会以稳定代码阻断启动，直到完成 `RedisCluster` 客户端适配。服务启动前会审计这些条件；生产不合规则拒绝启动，开发环境只记录不含连接地址、用户名或原始异常的稳定告警代码。审计读取各分区 `INFO`、`ROLE`、`ACL WHOAMI`，并仅执行精确的只读 `CONFIG GET appendfsync`；不会读取 `requirepass`。因此生产运行凭据如被授予 `CONFIG GET`，应将这一只读配置权限作为最小权限风险单独评估。

本仓库提供`deploy/redis/compose.yaml`作为Agent专用Redis Sentinel部署定义。数据节点监听`6380/6381`，三个Sentinel监听`26379/26380/26381`，master名称为`aicare-agent-master`且quorum为2。应用用户`aicare_agent`只允许访问固定RunStore key命名空间和门禁/脚本所需命令；复制、Sentinel管理Redis、Sentinel互联和应用发现分别使用独立ACL。所有密码从虚拟机只读secret注入，default用户关闭，不会修改或复用Java的`aicare-redis:6379`。部署前必须准备以下0600文件：

```text
/home/aicare/.config/aicare/secrets/agent-redis-password
/home/aicare/.config/aicare/secrets/agent-redis-replication-password
/home/aicare/.config/aicare/secrets/agent-redis-sentinel-client-password
/home/aicare/.config/aicare/secrets/agent-redis-sentinel-peer-password
/home/aicare/.config/aicare/secrets/agent-redis-sentinel-management-password
```

然后在目标机部署并检查状态：

```bash
cd /home/aicare/aicare-agent-redis
docker compose config --quiet
docker compose up -d
docker compose ps
```

生产环境必须设置`AICARE_AGENT_REDIS_MODE=sentinel`，并配置三个发现端点、master名称和Sentinel客户端ACL。`AICARE_AGENT_REDIS_URL`在Sentinel模式下只提供数据节点的用户名、密码和DB，URL中的主机不会作为固定连接地址。redis-py使用`Sentinel.master_for()`发现可写master，故障转移后关闭旧连接并重新发现。FastAPI与checkpoint清理CLI共用这套工厂，避免维护命令误连旧副本。

容器名中的`primary/replica`只表示首次部署身份，Sentinel切换后实时角色可能相反，应通过`ROLE`或`SENTINEL get-master-addr-by-name`判断。当前五个容器仍位于同一台虚拟机，只能处理Redis进程/容器故障，不能覆盖虚拟机、宿主机或网络整体故障。真正跨故障域生产部署应把Redis数据节点和至少三个Sentinel分散到不同主机。Redis Cluster仍不支持；检测到`cluster_enabled=1`会以`REDIS_CLUSTER_CLIENT_UNSUPPORTED`阻断启动。RunStore固定hash tag只为多key Lua同槽预留，不是Redlock。

支持ACL的图形客户端应选择“哨兵”，填写任一或全部`26379–26381`端点、master名称`aicare-agent-master`、用户名`aicare_sentinel`及Sentinel客户端密码。客户端随后还需使用数据节点用户名`aicare_agent`及数据密码；若客户端只有一组认证字段且不能分别配置Sentinel/data ACL，则该客户端不适合此安全拓扑。

run生命周期由`AgentRunLifecycle`统一编排。`AICARE_AGENT_RUN_HEARTBEAT_SECONDS`必须小于租约时长，心跳会原子续租并发出临时`RUN_HEARTBEAT`；`AICARE_AGENT_RUN_TIMEOUT_SECONDS`限制单次图执行。Java发起取消时，Redis先记录取消意图，持有当前lease的执行者停止图工作后提交`CANCELLED`，取消和超时都不会产生可持久化Final。Redis异常或lease丢失会原样返回稳定基础设施错误，不会伪装成模型失败。

相同`runId`完成后只从Redis记录的`checkpointId`读取PostgreSQL状态，校验请求身份与终态摘要后重建终态事件，不再次调用图或模型。checkpoint缺失、损坏或身份不一致时返回明确的`RUN_REPLAY_UNAVAILABLE`；Java必须创建新的`runId`，并可在新run请求中提供经过安全裁剪的历史和业务上下文，Python不会静默重跑已经完成的相同run。

Redis终态run按`AICARE_AGENT_RUN_RETENTION_SECONDS`自动过期；PostgreSQL checkpoint由显式维护命令按`AICARE_AGENT_CHECKPOINT_RETENTION_SECONDS`清理。RUNNING run使用无TTL的conversation active标记阻止清理，终态原子删除；若run ledger已过期，活跃检查会原子清除悬空标记。实际删除前还会获取短时cleanup guard，阻止同一会话启动新run，并保证单thread删除超时严格短于guard。

Elasticsearch和RabbitMQ配置将在RAG、知识事件Task启用，目前允许留空。

LangSmith本地追踪使用：

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-local-langsmith-key
LANGSMITH_PROJECT=aicare-agent-service-dev
```

追踪开启前必须确认Prompt、状态和元数据不包含CDK、账号密码、下载凭证、Bearer Token、Service Token或其他敏感权益。

## 启动FastAPI

```powershell
.\.venv\Scripts\python.exe -m uvicorn aicare_agent_service.main:app `
  --app-dir src `
  --host 127.0.0.1 `
  --port 8090
```

可用探针：

- `GET http://127.0.0.1:8090/health`
- `GET http://127.0.0.1:8090/api/v1/agent/health`
- `GET http://127.0.0.1:8090/health/ready`

`/health/ready`当前只表示本地配置校验通过，不代表DeepSeek、PostgreSQL或Java连通。

## 初始化PostgreSQL Checkpointer

首次部署或升级LangGraph Checkpointer表结构时，显式执行：

```powershell
.\.venv\Scripts\python.exe -m aicare_agent_service.persistence.init_db
```

该命令要求 `AICARE_AGENT_CHECKPOINT_BACKEND=postgres`、`AICARE_AGENT_POSTGRES_DSN` 和 `LANGGRAPH_AES_KEY` 均已配置。它只调用LangGraph `setup()`；普通FastAPI启动和请求路径都不会隐式建表。重复执行由LangGraph migrations安全处理。失败时命令只输出稳定中文诊断，不输出DSN、凭据或异常堆栈。

## 清理过期Checkpoint

清理命令默认只报告候选，不删除：

```powershell
.\.venv\Scripts\python.exe -m aicare_agent_service.persistence.cleanup_checkpoints
```

确认dry-run统计后，显式执行：

```powershell
.\.venv\Scripts\python.exe -m aicare_agent_service.persistence.cleanup_checkpoints --apply
```

候选判断只读取`thread_id`和最新`checkpoint_id`，兼容当前LangGraph生成的UUIDv6及UUIDv7时间ID；不读取、解密或输出checkpoint正文。无法解析的旧格式保守跳过。每个候选还会经过Redis active-run和lease核对；Redis不可用、会话活跃、guard冲突或删除超时都不会删除。单个thread失败不阻断其他候选，下次运行可继续重试。

### Windows PostgreSQL运行约束

在原生Windows上，显式初始化命令已自动使用 psycopg 兼容的 Selector 事件循环，仍按上一节命令执行。项目提供的`aicare-langgraph`包装器也会在官方CLI创建Uvicorn前注入可跨重载子进程导入的Selector loop工厂，并在GBK解释器下自动以当前进程环境启动UTF-8子进程。

若只启动FastAPI而不使用Agent Server，仍可用以下PowerShell命令显式创建Selector loop（不要把该策略设为机器级全局配置）：

```powershell
.\.venv\Scripts\python.exe -c "import asyncio; from selectors import SelectSelector; import uvicorn; config = uvicorn.Config('aicare_agent_service.main:app', host='127.0.0.1', port=8090); runner = asyncio.Runner(loop_factory=lambda: asyncio.SelectorEventLoop(SelectSelector())); runner.run(uvicorn.Server(config).serve()); runner.close()"
```

Windows包装器只用于本地开发和Agent Chat UI调试；生产部署仍使用Linux容器/托管Agent Server，并由部署平台提供进程生命周期、认证和网络隔离。

## 启动LangGraph dev

安装/同步项目后，使用项目包装器启动本地Agent Server。它会自动处理Windows GBK和psycopg事件循环兼容性，不需要设置机器级或终端级`PYTHONUTF8`：

```powershell
.\.venv\Scripts\aicare-langgraph.exe dev --config langgraph.json --no-browser
```

默认API地址为`http://127.0.0.1:2024`，当前注册两个开发图：

- `customer_service`：正式根图；执行Java身份上下文校验、安全预处理、DeepSeek结构化分类、确定性路由、输出门禁和终态封装。Task 7/8专业子图未装配前，进入这些分支会明确失败，不回退到固定或伪造业务回答。
- `model_playground`：使用`ModelPurpose.ANSWER`对应的DeepSeek模型，仅验证多轮消息、流式输出和LangSmith Trace。

`langgraph.json`为Agent Server注入应用统一的加密PostgreSQL Checkpointer；控制台中“in-memory runtime”仅表示本地API调度器版本，不代表会话checkpoint落在内存。`model_playground`仍不包含正式路由、Java业务工具、RAG、`conversationId`或人工接管状态。

## 连接Agent Chat UI

Agent Chat UI放在固定的仓库外目录，避免把Node依赖和调试页面混入Agent服务：

```powershell
git clone https://github.com/langchain-ai/agent-chat-ui.git D:\code\AICareDesk-agent-chat-ui
Set-Location D:\code\AICareDesk-agent-chat-ui
pnpm install
pnpm dev
```

打开`http://localhost:3000`并填写：

- Deployment URL：`http://localhost:2024`
- Assistant/Graph ID：`model_playground`
- LangSmith API Key：本地LangGraph Server不需要，可留空。
- Built with Agent Builder：关闭。

也可以在Agent Chat UI自己的`.env`中设置：

```dotenv
NEXT_PUBLIC_API_URL=http://localhost:2024
NEXT_PUBLIC_ASSISTANT_ID=model_playground
NEXT_PUBLIC_AUTH_SCHEME=
```

Agent Chat UI仅用于本地开发调试。DeepSeek与LangSmith Key只保存在Agent服务Git忽略的`.env`中，不得复制到UI目录。它不是商城C端，也不能替代Java会话网关。

## 测试

```powershell
.\.venv\Scripts\uv.exe sync --locked --group dev
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
git diff --check
```

单独验证开发图：

```powershell
.\.venv\Scripts\python.exe -m pytest tests\dev -q
```

真实基础设施测试默认跳过。仅在当前PowerShell进程注入专用、可删除的测试实例连接串后执行；不要使用Java业务库、生产数据库或生产Redis：

```powershell
$env:AICARE_AGENT_TEST_POSTGRES_DSN='postgresql://test-user:password@test-host/test_db'
$env:AICARE_AGENT_TEST_REDIS_URL='redis://:password@test-host:6379/1'
.\.venv\Scripts\python.exe -m pytest tests\persistence -q
```

原生Windows上的 pytest 会仅在测试进程内使用 Selector 事件循环，因此上面的普通 pytest 命令可用于 psycopg PostgreSQL 集成测试；它不会设置机器级或生产应用的事件循环策略。

真实DeepSeek/LangSmith Smoke默认跳过，不产生网络请求或模型费用。只有本地`.env`已经配置两个Key时，才在当前PowerShell进程显式开启：

```powershell
$env:AICARE_RUN_LIVE_MODEL_TESTS='1'
$env:LANGSMITH_TRACING='true'
.\.venv\Scripts\uv.exe run --env-file .env --no-sync python -m pytest tests\models\test_deepseek_live.py -q -s
```

该命令会读取DeepSeek模型列表、读取最多一个LangSmith Project，发送一次结构化输出和一次流式模型请求，并在`LANGSMITH_PROJECT`对应项目创建带`task2-live`标签的开发Trace。测试只使用虚构Prompt和非敏感元数据，但仍会产生少量模型费用；不要把真实用户、订单、权益或会话数据加入此测试。

## 安全

- 不提交`.env`、API Key、Service Token、生产DSN或脱敏前的调试Trace。
- Agent工具中的用户身份必须来自Java认证上下文，不能由模型传入`user_id`。
- 写操作必须由Java执行权限、状态、金额和幂等校验；支付、取消等高风险动作后续使用LangGraph中断确认。
- 敏感权益揭示由Java直接返回前端，不经过模型Prompt、Checkpoint、LangSmith或普通日志。
- 本地服务默认只绑定`127.0.0.1`；不要在不可信网络使用`0.0.0.0`暴露开发服务器。
