# LCode

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**LCode** 是一个基于 Python 和大语言模型 API（DeepSeek）的 **AI Agent Framework**，具备从单 Agent 对话到多 Agent 协作、从工具调用到 RAG 检索、从 CLI 到 Web UI 的完整能力栈。

### 技术选型

| 类别 | 技术栈 | 说明 |
|------|--------|------|
| **语言** | Python ≥ 3.10 | 类型提示 + 异步支持 |
| **LLM SDK** | `openai` + `httpx` | 兼容 OpenAI 格式 API（OpenAI / DeepSeek）|
| **Web** | FastAPI + Uvicorn + WebSocket | 内置原生 HTML/JS 前端 |
| **向量库** | ChromaDB + sentence-transformers | 本地 Embedding，零 API 成本 |
| **配置** | Pydantic Settings | `.env` 环境变量管理 |
| **日志** | structlog + loguru | 结构化 JSON 日志 |
| **CLI** | typer + rich | 现代终端交互体验 |
| **测试** | pytest + pytest-asyncio | 异步测试支持 |
| **部署** | Docker + docker-compose | CI/CD 通过 GitHub Actions |

## 架构概览
### 项目架构图：
![系统架构图](assets/SystemArchitectureDiagram.png)
![结合项目结构：](assets/ProjectStructureDiagram.png)

## 核心工作流图：
![核心工作流](assets/CoreWorkFlow.png)

### 项目基于课程设计选题：创建一个自己的Agent
#### 由基本实现到具备较为完整Agent功能的实现分为五个等级，现已完成至5th Level

### Level 1 — ChatAgent（单轮对话）

最基础的 Agent，维护对话历史，每轮将 `[system_prompt] + [历史消息] + [用户输入]` 拼装后调用 LLM，返回响应并存入历史。

```
用户输入 → 拼装消息列表 → 单次 LLM 调用 → 存入历史 → 返回响应
```

核心类：`ChatAgent`（`agents/chat_agent.py`），继承 `BaseAgent`，实现 `run(user_input) -> LLMResponse`。

### Level 2 — ReActAgent（工具调用循环）

在 ChatAgent 基础上增加工具调用能力。通过 `ToolRegistry` 注册工具，LLM 决定何时调用哪个工具，Agent 执行工具后将结果回传 LLM，循环直至得到最终答案。

```
用户输入 → 注入工具描述到 System Prompt
         → 循环（最多 10 次）:
             LLM 调用（传入 tools schema）
             ├─ 返回 tool_calls → ToolRegistry 执行工具
             │                  → 追加 role="tool" 消息 → 继续循环
             └─ 无 tool_calls  → 返回最终答案
```

核心类：`ReActAgent`（`agents/chat_agent.py`），持有 `ToolRegistry` 实例，每轮调用 `_call_llm(messages, tools=tool_registry.to_openai_schemas())`。

### Level 3 — RAGAgent（检索增强）

在 ChatAgent 基础上增加文档检索能力。用户输入先经过 `VectorStore` 检索相关文档片段，将检索结果拼接为上下文，增强 prompt 后送入 LLM。

```
用户输入 → VectorStore.query() 检索 Top-K 文档片段
         → 拼接上下文：[检索结果] + [用户问题]
         → LLM 调用 → 返回增强后的响应
         （无检索结果时回退为纯 LLM 对话）
```

核心类：`RAGAgent`（`agents/rag_agent.py`），持有 `VectorStore` 实例，提供 `ingest()` 和 `ingest_directory()` 方法加载文档。

### Level 4 — Multi-Agent（Plan-and-Solve 编排）

将复杂任务分解为子任务，指派给不同 Agent 执行，最后由 LLM 集成所有子结果。

```
复杂任务 → [Plan] LLM 分解为子任务 JSON
         → [Delegate & Solve] 逐个子任务:
             _select_agent(task) → agent.run(task.description)
             → event_bus.publish("task_completed", ...)
         → [Integrate] LLM 汇总所有子结果 → 返回最终答案
```

核心类：`AgentManager`（`orchestration/manager.py`），持有注册 Agent 字典，通过 `EventBus` 发布任务完成事件，`_select_agent()` 按名称匹配或回退到首个可用 Agent。

### Level 5 — 生产级（Web UI + Observability + MCP + Plugin + Docker）

在以上能力基础上叠加生产级组件：

| 组件 | 职责 |
|------|------|
| `ToolRegistry` | 装饰器 `@register` 注册工具，自动推断 JSON Schema，执行工具调用 |
| `VectorStore` | 基于 ChromaDB + sentence-transformers 的文档嵌入与余弦相似度检索 |
| `InMemoryMemory` / `SQLiteMemory` | 对话持久化后端（deque / SQLite，Redis 待实现） |
| `EventBus` | 发布/订阅模式解耦 Agent 间通信 |
| `MCPServer` | JSON-RPC 协议暴露工具给外部调用 |
| `PluginLoader` | 动态加载 `plugin_dir` 下的 Python 文件注入工具 |
| `Tracer` / `TraceSpan` | 基于 structlog + loguru 的链路追踪，`@trace_method()` 装饰器 |
| `SkillSystem` | 高层技能抽象，支持函数 / 异步函数 / 带 `.run()` 方法的对象 |

### 入口触发方式

| 入口 | 说明 |
|------|------|
| `lcode chat --agent react` | CLI REPL，创建 `OpenAIProvider` + 对应 Agent，循环收发消息，整个会话包裹在 `TraceSpan` 中 |
| `lcode server` | 启动 FastAPI + Uvicorn，WebSocket `/ws/{agent_type}` 每连接创建一个 Agent 实例 |
| `POST /mcp/` | JSON-RPC，由 `MCPServer` 处理工具发现与调用 |
| `POST /im/webhook` | IM 适配器，`WebhookAdapter` 通过 HTTP 接入即时通讯平台 |


## 快速开始

### 安装

### 建议启动虚拟环境，以隔离不同py版本，避免出现库版本冲突
目前我在开发环境所使用的是py自带的venv模块

```bash
python -m venv .venv
```

安装所需库：
```bash
pip install -e ".[all]"
```

### 配置

创建 `.env` 文件：样例参照.env.example文件.
如配置DeepSeek API：
`DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}`，建议以环境变量存储API_KEY避免泄露
`DEEPSEEK_BASE_URL=https://api.deepseek.com/v1`


### CLI 对话

```bash
lcode chat
```

### Web UI

```bash
lcode server
```

### Docker 部署

```bash
docker-compose up -d
```




## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 代码格式化
black lcode tests
ruff check --fix lcode tests

# 类型检查
mypy lcode

# 运行测试
pytest
```

## License

MIT
