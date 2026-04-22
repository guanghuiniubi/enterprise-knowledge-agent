# 基于当前项目的 Agent 核心知识点梳理

> 这份文档不是泛泛讲 Agent，而是**完全基于当前项目实现**来总结：这个项目里真正体现了哪些 Agent 能力、当前知识数据到底走哪条链路、面试时可以怎么讲。

---

## 一、先说结论：当前项目的数据是不是 embedding 到数据库？

### 结论
**当前在线问答主链路，没有走数据库向量检索。**

也就是说：
- 你之前初始化过的一批 Markdown 文件，**确实有一套 ingestion + embedding + PostgreSQL/pgvector 的代码链路**
- 但是**当前运行中的 `/chat` Agent 主流程，并没有使用这条链路**
- 当前主流程实际使用的是：**本地 JSON 文件 + 轻量关键词检索**

---

## 二、当前项目里存在两条知识链路

## 1）旧链路：Markdown -> chunk -> embedding -> DB

这条链路是存在的，代码如下：

- `app/ingestion/ingest_service.py`
- `app/rag/vector_retriever.py`
- `app/repositories/kb_chunk_repo.py`
- `app/repositories/kb_document_repo.py`
- `app/models/kb_chunk.py`
- `app/models/kb_document.py`

### 这条链路怎么工作
1. 从 `settings.knowledge_base_path` 读取 Markdown 目录
2. 把文档切 chunk
3. 用 embedding 模型向量化
4. 写入 `kb_document` / `kb_chunk`
5. 查询时通过 `pgvector` 做相似度搜索

### 关键代码
在 `app/ingestion/ingest_service.py`：
- `markdown_loader.load_directory(...)`
- `markdown_chunker.split(...)`
- `local_embedding_service.embed_texts(...)`
- `kb_chunk_repo.bulk_insert(rows)`

这说明：
**“你之前那批 md 文件被 embedding 到数据库”这件事，在旧知识链路里是成立的。**

---

## 2）当前在线主链路：JSON 文件 -> `KnowledgeRetriever`

当前 `/chat` 实际调用的是：

- `app/agent/orchestrator.py`
- `app/tools/interview_tools.py`
- `app/rag/retriever.py`

### 当前真实调用链
在 `app/agent/orchestrator.py` 中：
- orchestrator 只依赖 `interview_toolkit`
- `interview_toolkit` 内部依赖 `KnowledgeRetriever()`
- `KnowledgeRetriever()` 读取的是 JSON 文件，而不是向量库

在 `app/rag/retriever.py` 中：
- `_load_docs()` 用 `json.load(...)` 直接读文件
- `search()` 是基于 token 命中做打分
- 没有查数据库
- 没有走 `vector_retriever`

### 为什么说当前没走数据库
因为当前主入口 `app/main.py` 里只挂了：
- `chat_router`

没有挂：
- `knowledge_router`
- 数据导入接口也没接到主应用上

所以现在的系统虽然**保留了 DB 向量链路代码**，但**活跃路径已经切到轻量 JSON 检索**。

---

## 三、这正是一个很好的面试点：同一个项目里有“演进中的双检索架构”

面试时你可以这样讲：

> 这个项目里我实际上保留了两条知识检索链路：
> 一条是面向生产形态的 Markdown -> chunk -> embedding -> pgvector 检索；
> 另一条是为了快速演示 Agent 能力而采用的轻量 JSON 检索。
> 当前在线问答主链路走的是轻量检索，因为它启动成本低、依赖少、便于调试 Agent 循环；但项目中仍保留向量化入库和向量召回能力，后续可以把当前 Agent 的工具层切回 DB 检索。

这句话很有价值，因为它体现了：
- 你理解工程权衡
- 你不是只会“堆技术名词”
- 你知道 demo 和生产方案的取舍

---

# 四、基于当前项目，你最应该掌握的 Agent 核心知识点

下面这些点，是**这个项目本身已经体现出来**，并且非常适合在面试里讲的。

---

## 1. 什么是 Agent，不只是“调一次 LLM”

### 当前项目里的体现
在 `app/agent/orchestrator.py` 中，当前实现不是普通的“用户问题 -> 一次 LLM -> 返回答案”。
而是：

1. LLM 先看问题和上下文
2. 判断要不要调用工具
3. 如果要，返回 `tool_calls`
4. 代码执行工具
5. 把工具结果作为 `tool` message 回填给 LLM
6. LLM 再继续思考
7. 直到产出最终答案

### 面试怎么讲
> Agent 的核心不在于“用了几个 prompt”，而在于它具备了**基于上下文持续决策**的能力。这个项目里，LLM 不直接回答，而是先判断是否需要检索工具，再根据工具返回的 observation 继续推理，直到形成最终答案。这比单轮问答更接近真实 Agent。

### 关键词
- tool calling
- observe-act-think loop
- iterative reasoning
- context accumulation

---

## 2. Agent 编排的本质是一个循环，而不是 if/else

### 当前项目里的体现
在 `app/agent/orchestrator.py` 的 `_run_core()` 里：
- `for step_index in range(1, self.MAX_STEPS + 1)`
- 每一轮都可能：
  - 调工具
  - 收 observation
  - 再继续调用 LLM

这就是编排循环。

### 面试怎么讲
> 传统路由器是 if/else 分发：识别意图后直接走固定分支；而 Agent 编排是循环式的，它允许模型在每一步根据中间结果动态选择下一步动作。这个项目里就是通过 `_run_core()` 把这种多步决策体现出来的。

### 你可以强调
- Router 是静态分发
- Agent 是动态决策
- 两者最大的区别是是否有“中间状态驱动的下一步选择”

---

## 3. 原生 Tool Calling 和“伪 JSON Action”有什么区别

### 当前项目里的体现
你现在已经升级成了原生工具调用：

- `app/llm/client.py` 中新增了 `chat_with_tools(...)`
- `app/tools/interview_tools.py` 中定义了 `tool_schemas()`
- `app/agent/orchestrator.py` 中消费 `tool_calls`

### 为什么这比“让 LLM 输出 JSON 再手工解析”更好
因为原生 tool calling：
- 协议更稳定
- 输出结构更规范
- 更接近生产系统
- 更适合后续接 LangGraph / OpenAI tool ecosystem

### 面试怎么讲
> 项目一开始是用 JSON action 模拟 Agent，后来升级成原生 tool calling。这样做的好处是工具协议更稳定，LLM 的输出更结构化，也更利于扩展多工具场景。

---

## 4. 上下文回填是 Agent 能“思考下去”的关键

### 当前项目里的体现
在 `app/agent/orchestrator.py` 中：
- 工具执行完成后，会把 observation 追加到 `messages`
- 且角色是 `tool`
- 然后再调一次 `llm_client.chat_with_tools(...)`

这一步非常关键，因为没有它，工具调用就只是“查完立刻返回”，不是 Agent。

### 面试怎么讲
> 真正的 Agent 不是调用一次工具就结束，而是要把工具结果作为新的上下文回填给模型。这个项目里就是把 observation 作为 tool message 追加回去，让模型基于观察结果继续规划下一步。

### 这句话很重要
> **没有 observation 回填，就没有真正的多步 Agent。**

---

## 5. 为什么要限制 `MAX_STEPS`

### 当前项目里的体现
在 `app/agent/orchestrator.py`：
- `MAX_STEPS = 6`

### 原因
为了防止：
- 死循环
- 工具反复调用
- token 失控
- LLM 一直不收敛

### 面试怎么讲
> 多步 Agent 必须加 step budget。否则模型可能在某些 prompt 分布下反复调用工具，导致推理不收敛。这个项目里用 `MAX_STEPS` 做了最基本的执行保护。

### 可以补充
更成熟的系统还会加：
- token budget
- timeout
- tool whitelist / blacklist
- 同工具重复调用次数限制

---

## 6. 工具层要和 Agent 解耦

### 当前项目里的体现
在 `app/tools/interview_tools.py` 中：
- `list_topics`
- `search_knowledge`
- `read_topic`
- `generate_quiz`

orchestrator 并不直接读文件，而是只调用 toolkit。

### 这意味着什么
你后面可以把工具底层替换成：
- JSON 检索
- DB 向量检索
- Elasticsearch
- 远程知识服务

而 Agent 不需要大改。

### 面试怎么讲
> 我把 Agent 的决策层和工具执行层做了解耦。Orchestrator 只负责多步编排，不直接耦合具体存储或检索细节，底层知识源可以从 JSON 换成 pgvector，而不会破坏上层 Agent 流程。

---

## 7. Agent 不是只有最终答案，可观测性同样重要

### 当前项目里的体现
你现在返回里已经有：
- `tool_calls`
- `agent_steps`
- 流式 SSE 事件：`start` / `tool_result` / `final`
- `@traceable` tracing

### 这说明什么
这个项目不只是“能答”，还具备：
- 可解释性
- 可调试性
- 可观测性

### 面试怎么讲
> Agent 相比普通聊天更需要可观测性，因为错误可能出现在路由、工具选择、工具执行、上下文拼装、最终总结多个阶段。这个项目里我把中间步骤通过 `agent_steps` 和流式事件暴露出来，方便观察每一步怎么决策。

---

## 8. 流式输出不只是 UX，更是调试手段

### 当前项目里的体现
在 `app/api/chat.py`：
- `POST /chat/stream`

在 `app/services/chat_service.py`：
- `chat_stream(...)`
- 逐步输出 SSE event

### 面试怎么讲
> 流式接口除了提升用户体验，更重要的价值是把 Agent 的执行过程暴露出来。尤其在学习和调试阶段，看到 tool_result 比只看 final answer 更有帮助。

---

## 9. 当前项目的知识检索其实是“Agent 工具化的 RAG”

### 当前项目里的体现
不是传统“先召回再统一生成”，而是：
- 检索被封装成工具
- 由 LLM 自己决定何时检索、检索几次、是否继续读详情

### 这和普通 RAG 的区别
普通 RAG：
- 固定流程：retrieve -> stuff prompt -> answer

你这个 Agent 式 RAG：
- 可以先检索主题
- 再读主题详情
- 再生成 quiz
- 再最终总结

### 面试怎么讲
> 这个项目更接近 Agentic RAG，而不是传统 RAG。检索不是固定前置步骤，而是被封装成工具，由模型在推理过程中动态决定是否调用。

---

## 10. Demo 与生产系统的权衡，是一个非常值得讲的点

### 当前项目里的现实状态
你现在项目里其实有两套能力：
- 轻量 JSON 检索：便于 demo 和学习
- 向量入库 + pgvector：更接近生产方案

### 面试怎么讲
> 为了快速验证 Agent 循环和可观测性，我在线链路先用了低依赖的 JSON 检索；同时保留了 Markdown + embedding + pgvector 的生产化路径。这个选择体现的是工程权衡：先验证 Agent 编排本身，再逐步替换成更强的检索后端。

这句话非常像真实工程师的表达。

---

# 五、如果面试官问：“这个项目里最核心的能力是什么？”

你可以直接这样回答：

> 这个项目最核心的不是知识库本身，而是一个最小可运行的 Agent 编排闭环：
> 1）LLM 基于当前上下文决定是否调用工具；
> 2）工具返回 observation；
> 3）observation 被回填进上下文；
> 4）LLM 再次推理，直到输出最终答案。
> 同时我把工具层、检索层和编排层做了解耦，并增加了流式步骤输出和 tracing，便于调试和演示。

这段回答你可以直接背。

---

# 六、如果你想把“你之前导入的 md 数据”重新接回当前 Agent，需要做什么

## 当前缺口
当前主流程没有使用：
- `app/rag/vector_retriever.py`
- `app/api/knowledge.py`
- ingestion 接口也没在 `app/main.py` 挂载

## 要恢复到 DB 向量检索，需要做的事

### 最小改法
1. 在 `app/tools/interview_tools.py` 里把 `KnowledgeRetriever()` 换成 `vector_retriever`
2. 适配 `search_knowledge()` / `read_topic()` 的返回结构
3. 如果需要重新 ingest，就把 `knowledge_router` 挂回 `app/main.py`
4. 确保 PostgreSQL + pgvector 数据表和 embeddings 已准备好

## 这意味着什么
**你的 md 数据不是没用，而是现在这版主流程暂时没接上它。**

---

# 七、你接下来最值得继续学的知识点（完全基于当前项目）

如果你是为了面试，我建议优先把下面 7 个点讲熟：

1. **什么是真正的 Agent loop**
2. **Tool Calling 和普通 Prompt QA 的区别**
3. **Observation 回填上下文的重要性**
4. **Agent 和 Router 的区别**
5. **Agentic RAG 和传统 RAG 的区别**
6. **为什么要做 step limit / 可观测性 / 流式输出**
7. **为什么工程上会同时保留 demo 检索链路和生产向量链路**

---

# 八、最适合你面试时直接说的“项目总结版”

> 这个项目是一个面向知识问答的最小 Agent Demo。它的核心不只是调用一次大模型，而是实现了一个带工具调用的多步推理闭环：模型会根据问题先决定是否调用检索工具，工具结果会作为 observation 回填给模型，模型再继续决定下一步，直到产出最终答案。为了更容易调试和学习，我给系统加入了 `agent_steps`、SSE 流式输出和 tracing。同时项目还保留了 Markdown -> chunk -> embedding -> pgvector 的向量检索链路，只是当前在线主链路为了轻量演示，暂时走的是 JSON 检索。这体现了一个从 demo 到生产逐步演进的 Agent 工程思路。

---

# 九、建议的下一步

如果你接下来希望“完全基于当前项目继续准备面试”，最值得做的是：

1. 把当前 Agent 再切回你之前的 Markdown + pgvector 数据
2. 再新增两个工具：
   - `summarize_project_architecture`
   - `generate_project_interview_questions`
3. 这样这个 Agent 就不只是回答通用知识点，而是能直接围绕**你这个项目本身**帮你准备项目面试

---

> 文档生成时间：2026-04-21
> 生成依据：当前 workspace 中的 `orchestrator` / `toolkit` / `retriever` / `vector retriever` / `ingestion` / `main` 实现

