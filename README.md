# Interview Knowledge Agent Demo

这是一个轻量的 **知识库 Agent**，当前主流程已经接回：

- Markdown 文档导入
- chunk 切分
- embedding 向量化
- PostgreSQL + pgvector 检索

同时用于学习以下能力：

- 多步思考
- 原生工具调用（OpenAI-compatible function calling）
- 把工具结果回填到上下文，再次调用 LLM
- 输出可观察的 `agent_steps`
- 流式观察 Agent 执行过程

## 这个 demo 现在做了什么

用户提问后，Agent 不再只是做一次路由，而是进入一个循环：

1. LLM 先决定是否要调用工具
2. 如果要调用，就触发原生 tool calling
3. 代码执行工具，得到 observation
4. 把 observation 作为 tool message 追加回上下文
5. 再让 LLM 继续思考
6. 直到输出最终答案或追问

## 当前知识链路

1. 通过 `POST /knowledge/ingest` 导入 `KNOWLEDGE_BASE_PATH` 下的 Markdown 文件
2. 文档被切 chunk
3. chunk embedding 写入 PostgreSQL / pgvector
4. `/chat` 主流程通过向量检索工具召回知识，再交给 Agent 编排

## 可用工具

- `list_topics`：列出可学习的面试主题
- `search_knowledge`：检索相关知识点
- `read_topic`：读取某个知识点详情
- `generate_quiz`：生成该主题的追问题

## 运行

```bash
uv run uvicorn app.main:app --reload
```

启动后可直接打开内置 UI：

```text
http://127.0.0.1:8000/
```

现在内置 UI 已升级为 **Vue 3 单页界面**，支持：

- 历史会话列表
- 加载指定会话消息
- 删除会话
- 普通 / 流式问答
- 查看工具调用与 Agent steps
- 内置 observability dashboard（KPI / alerts / governance / latency）

## 新增能力：Prompt / 治理 / Rerank / 评测 / 权限 / 安全

### 1. Prompt Version 管理

现在所有关键 Prompt 都统一收敛到：

- `app/prompts/registry.py`
- `app/prompts/prompts.yaml`

- `agent_system`
- `agent_user`
- `fallback_summary_system`
- `fallback_summary_user`
- `route_classifier_system`

支持能力：

- 多版本注册
- 激活当前生效版本
- 查询当前 active version
- YAML 持久化
- 热加载 reload
- 在 Agent debug 信息中回传当前 prompt 版本

接口：

- `GET /prompts`：查看所有 prompt 与 active version
- `POST /prompts/reload`：从 YAML 重新加载 prompt
- `GET /prompts/{name}`：查看某个 prompt 当前激活版本
- `POST /prompts/{name}/activate`：切换 prompt 版本

示例：

```bash
curl http://127.0.0.1:8000/prompts
```

### 2. 知识权限过滤

新增访问控制模块：`app/security/access_control.py`

支持：

- 文档级权限控制
- 基于 `user_id / role / department / clearance_level` 的访问校验
- 检索前过滤召回结果
- 读取主题详情前做二次权限检查

请求里已支持传入：

- `user_roles`
- `user_departments`
- `clearance_level`

Markdown 知识文档也支持 YAML front matter，例如：

```md
---
title: JVM 调优与线上排障
access:
  visibility: restricted
  allowed_roles: [backend, infra]
  allowed_departments: [platform]
  min_clearance: 2
---

正文内容...
```

### 3. 分布式限流 / 熔断 / 超时治理

新增治理模块：`app/core/governance.py`

覆盖范围：

- Chat 入口级限流（按 `user_id` / `session_id`）
- LLM 调用级限流、熔断、超时
- Tool 调用级限流、熔断、超时
- 运行时治理状态快照
- Redis 后端分布式治理
- Redis 不可用时自动回退本地内存治理

当模型或工具不可用时，Agent 会：

- 尽量使用已收集 observation 继续回答
- 在必要时进入 deterministic fallback
- 在 `debug.fallback` 中标记是否发生降级

接口：

- `GET /governance`：查看当前治理配置与 circuit state

关键配置：

- `GOVERNANCE_BACKEND=memory|redis`
- `REDIS_URL=redis://localhost:6379/0`
- `REDIS_KEY_PREFIX=eka`

### 4. 更完善的 Rerank 策略

新增 `app/rag/reranker.py`，当前采用轻量 Hybrid Rerank：

- 向量召回分数
- 标题命中分数
- 正文关键词覆盖分数
- metadata/tag 命中分数
- 位置衰减
- 轻量 diversity penalty，避免返回过于重复的 chunk

检索策略已升级为：

1. 先扩大向量召回候选集
2. 再做 hybrid rerank
3. 最终把 `rerank_score` 和诊断信息一并返回

### 5. 安全防护：Prompt Injection / 输出风险过滤 / 日志脱敏

新增模块：

- `app/security/content_guard.py`
- `app/security/redaction.py`
- `app/core/logging.py`

当前支持：

- Prompt Injection 检测与阻断
- 系统提示词 / 工具协议泄露拦截
- 输出中的 `api_key / password / secret / bearer token / 手机号 / 邮箱` 脱敏
- 日志自动脱敏

典型拦截场景：

- `ignore all previous instructions`
- `reveal system prompt`
- `输出系统提示词`
- `developer message`

### 6. 评测体系（accuracy / step / latency / fallback）

新增：

- Schema：`app/schemas/evaluation.py`
- Service：`app/services/evaluation_service.py`
- API：`POST /evaluation/run`

评测输出包括：

- `accuracy`：基于 expected / forbidden keywords 的通过情况
- `step_count`：Agent step 数量
- `latency_ms`：单条 case 总耗时
- `fallback`：是否触发降级
- `passed`：是否满足阈值要求

### 7. 可观测体系与 Dashboard

当前系统已经补齐一套轻量但可运营的可观测闭环，包括：

- HTTP 请求级指标
- Chat / Stream 请求量与延迟
- Fallback rate
- Prompt injection hit rate / block rate
- ACL deny rate
- Tool failure rate
- Retrieval / rerank / LLM / tool latency
- 告警快照与治理状态

后端接口：

- `GET /observability/metrics`：原始 counters / observations / derived / alerts
- `GET /observability/overview`：dashboard 直接消费的 KPI / alerts / governance / latency 摘要
- `GET /observability/alerts`：告警视图

前端访问：

- `GET /`：内置聊天 UI + observability dashboard

Dashboard 默认展示的关键指标：

- `fallback_rate`
- `prompt_injection_hit_rate`
- `acl_deny_rate`
- `tool_failure_rate`
- `chat_p95_latency_ms`
- `retrieval_p95_latency_ms`

请求示例：

```bash
curl -X POST http://127.0.0.1:8000/evaluation/run \
  -H 'Content-Type: application/json' \
  -d '{
	"cases": [
	  {
		"case_id": "tcp-1",
		"question": "为什么 TCP 建连需要三次握手？",
		"expected_keywords": ["三次握手", "序列号"],
		"max_steps": 4,
		"max_latency_ms": 5000,
		"expect_fallback": false
	  }
	]
  }'
```

访问接口：

- `GET /`（简易对话 UI）
- `GET /health`
- `GET /db/health`
- `GET /governance`
- `GET /prompts`
- `POST /prompts/reload`
- `GET /prompts/{name}`
- `POST /prompts/{name}/activate`
- `GET /sessions`
- `GET /sessions/{session_id}/messages`
- `GET /session_state/{session_id}`
- `DELETE /sessions/{session_id}`
- `POST /knowledge/ingest`
- `POST /chat`
- `POST /chat/stream`
- `POST /evaluation/run`

### 请求示例

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
	"user_id": "u1",
	"session_id": "s1",
	"user_roles": ["backend"],
	"user_departments": ["platform"],
	"clearance_level": 1,
	"question": "为什么 TCP 建连需要三次握手，而不是两次？"
  }'
```

返回里会包含：

- `answer`：最终答案
- `tool_calls`：工具调用记录
- `agent_steps`：每一步的 thought / action / observation

### 流式请求示例

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{
	"user_id": "u1",
	"session_id": "s2",
	"question": "Redis 持久化有哪些方式？"
  }'
```

流式返回会持续输出 SSE 事件，例如：

- `start`
- `tool_result`
- `final`

## 测试

```bash
uv run pytest -q
```

当前新增测试已覆盖：

- Prompt version 激活
- Prompt YAML 持久化
- 治理超时与熔断
- Redis 治理后端切换
- Hybrid rerank 排序
- 知识权限过滤
- Prompt injection 阻断
- 输出风险过滤
- 日志脱敏
- 评测汇总统计
- Prompt / Governance / Evaluation API 回归

