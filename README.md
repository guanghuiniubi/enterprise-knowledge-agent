# Enterprise Knowledge Agent

一个基于 `langchain` + `langgraph` 的学习型面试助手项目骨架，目标是帮助你系统化练习 AI Agent 工程师面试能力。

## 当前能力

- 使用 `pydantic-settings` 从 `.env` 加载配置
- 支持使用 OpenAI-compatible client 接入 Xiaomi 大模型（默认 `mimo-v2-pro`）
- 基于统一抽象定义可扩展组件：
  - `BaseAgent`
  - `BaseTool`
  - `BaseMemory`
  - `BasePlanner`
  - `BaseRetriever`
  - `BaseSessionStore`
  - `BaseExecutor`
- 基于 `LangGraph` 编排标准 Agent 流程：
  - 加载会话历史
  - 生成计划摘要
  - 知识检索
  - LLM 决策与工具调用
  - 生成最终回答
  - 保存多轮记忆
- 支持展示安全的可观察执行过程：`plan summary`、检索结果、工具调用轨迹、阶段性 trace
- 提供最小可运行的命令行入口与 FastAPI 接口
- 提供测试样例，便于继续扩展

## 推荐目录结构

```text
src/eka/
  config/          # 配置与模型提供者
  core/            # 类型定义与抽象基类
  agents/          # Agent 实现
  tools/           # 工具实现
  memory/          # Memory 实现
  planner/         # Planner 实现
  retrievers/      # Retriever 实现
  session/         # Session Store ���现
  executor/        # LangGraph 执行器
  api/             # FastAPI 接口
```

## 核心设计原则

1. **配置与业务解耦**：统一通过 `Settings` 管理环境变量。
2. **抽象优先**：先定义 `Base*` 契约，再提供默认实现。
3. **可观察但不过度暴露推理**：展示 `plan summary`、检索结果、工具调用轨迹，而不是输出原始 chain-of-thought。
4. **可替换组件**：工具、检索器、记忆、会话存储、执行器都可单独替换。
5. **先可运行再扩展**：先提供最小可工作的面试助手，再逐步接入 RAG、数据库和更复杂的工具链。

## 快速开始

### 1. 安装依赖

```bash
uv sync
```

### 2. 准备环境变量

复制并编辑示例配置：

```bash
cp .env.example .env
```

当前项目会自动从工作区下的 `.env` 读取配置。

如果你使用的是 Xiaomi 大模型，可以直接填写：

```dotenv
LLM_PROVIDER=xiaomi
LLM_MODEL=mimo-v2-pro
LLM_BASE_URL=https://api.xiaomimimo.com/v1
LLM_API_KEY=your_xiaomi_api_key
LLM_TEMPERATURE=0.2
LLM_TIMEOUT=60
LLM_MAX_RETRIES=2
LLM_MAX_TOKENS=
```

当前实现会通过 `langchain-openai` 的 `ChatOpenAI` 作为 OpenAI-compatible client 来访问 Xiaomi 接口。

### 3. 运行命令行 Demo

```bash
uv run eka-demo --question "请帮我准备一段关于 LangGraph Agent 设计的面试回答"
```

交互式模式：

```bash
uv run eka-demo
```

### 4. 启动 API

```bash
uv run eka-api
```

默认地址：`http://127.0.0.1:8000`

### 5. 运行测试

```bash
uv run pytest
```

## 默认 Agent 流程

1. 读取历史会话消息
2. 使用 `Planner` 输出计划摘要：
   - 当前问题意图
   - 推荐检索词
   - 建议优先使用的工具
3. 使用 `Retriever` 从知识库目录中检索文档
4. 将计划摘要 + 检索结果作为上下文交给模型
5. 模型按需调用工具
6. 汇总最终回答并写入会话记忆

## Xiaomi / OpenAI-compatible 模型说明

- `LLM_PROVIDER=xiaomi` 时，会自动按 OpenAI-compatible 方式构建聊天模型
- 若未显式设置 `LLM_BASE_URL`，默认会使用 `https://api.xiaomimimo.com/v1`
- 当前 Provider 工厂位于 `src/eka/config/providers.py`
- 你也可以继续扩展更多 OpenAI-compatible 提供商，而不需要改动 Agent 主流程

## 下一步扩展建议

- 将 `InMemorySessionStore` 替换为 Redis / Postgres
- 将 `KeywordKnowledgeBaseRetriever` 升级为向量检索
- 增加简历解析、JD 对齐、模拟追问、答案打分等工具
- 增加 Web UI 或流式前端
- 接入 LangSmith trace 做链路观测

