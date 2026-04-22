import json

from app.agent.orchestrator import AgentOrchestrator
from app.llm.client import llm_client
from app.tools.interview_tools import interview_toolkit
from app.services.chat_service import chat_service
from app.schemas.chat import ChatRequest


def _parse_sse_chunks(chunks: list[str]) -> list[dict]:
    events: list[dict] = []
    for chunk in chunks:
        line = next((item for item in chunk.splitlines() if item.startswith("data: ")), None)
        if line:
            events.append(json.loads(line[6:]))
    return events


def test_agent_runs_react_loop(monkeypatch):
    monkeypatch.setattr(
        interview_toolkit,
        "search_knowledge",
        lambda query, top_k=3: {
            "query": query,
            "results": [{
                "id": "101",
                "chunk_id": "1001",
                "title": "TCP 三次握手与四次挥手",
                "summary": "TCP 通过三次握手建立可靠连接。",
                "tags": ["TCP"],
                "score": 0.99,
            }]
        }
    )
    monkeypatch.setattr(
        interview_toolkit,
        "read_topic",
        lambda doc_id="", topic="": {
            "found": True,
            "doc": {
                "id": "101",
                "title": "TCP 三次握手与四次挥手",
                "summary": "TCP 通过三次握手建立可靠连接。",
                "tags": ["TCP"],
                "key_points": ["三次握手", "四次挥手"],
                "interview_questions": [],
                "content": "TCP 需要三次握手确认双方的收发能力和初始序列号。",
                "source_file": "tcp.md",
            }
        }
    )

    responses = iter([
        {
            "content": "先搜索最相关的知识点",
            "tool_calls": [{
                "id": "call_1",
                "name": "search_knowledge",
                "arguments": json.dumps({"query": "TCP三次握手", "top_k": 2}, ensure_ascii=False)
            }]
        },
        {
            "content": "读取命中的主题详情",
            "tool_calls": [{
                "id": "call_2",
                "name": "read_topic",
                "arguments": json.dumps({"doc_id": "101"}, ensure_ascii=False)
            }]
        },
        {
            "content": "核心结论：在《TCP 三次握手与四次挥手》这个主题里，TCP 需要三次握手来确认双方的收发能力和初始序列号。面试回答思路：先说明三次握手的三个报文，再解释两次握手的风险，最后补充历史失效连接和四次挥手。易错点：不要只说“建立连接”，要强调确认双方都具备收发能力。可追问：为什么挥手通常是四次？",
            "tool_calls": []
        }
    ])

    monkeypatch.setattr(llm_client, "chat_with_tools", lambda messages, tools: next(responses))

    agent = AgentOrchestrator()
    result = agent.run("为什么 TCP 建连需要三次握手？")

    assert result.route == "agent_answer"
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0]["tool_name"] == "search_knowledge"
    assert result.tool_calls[1]["tool_name"] == "read_topic"
    assert len(result.agent_steps) == 3
    assert "三次握手" in result.answer
    assert result.citations[0]["doc_id"] == "101"


def test_agent_can_ask_for_clarification(monkeypatch):
    monkeypatch.setattr(
        llm_client,
        "chat_with_tools",
        lambda messages, tools: {
            "content": "CLARIFICATION: 你想练哪一类面试知识点？比如操作系统、网络、数据库或 Python。",
            "tool_calls": []
        }
    )

    agent = AgentOrchestrator()
    result = agent.run("帮我准备面试")

    assert result.route == "clarification"
    assert result.need_clarification is True
    assert "哪一类" in result.answer


def test_chat_stream_emits_tool_and_final_events(monkeypatch):
    monkeypatch.setattr(
        interview_toolkit,
        "search_knowledge",
        lambda query, top_k=3: {
            "query": query,
            "results": [{
                "id": "202",
                "chunk_id": "2001",
                "title": "Redis 缓存设计与持久化",
                "summary": "Redis 持久化包括 RDB 和 AOF。",
                "tags": ["Redis"],
                "score": 0.95,
            }]
        }
    )

    responses = iter([
        {
            "content": "先检索 Redis 主题",
            "tool_calls": [{
                "id": "call_redis_1",
                "name": "search_knowledge",
                "arguments": json.dumps({"query": "Redis 持久化", "top_k": 1}, ensure_ascii=False)
            }]
        },
        {
            "content": "资料已足够，整理成最终回答",
            "tool_calls": []
        }
    ])
    monkeypatch.setattr(llm_client, "chat_with_tools", lambda messages, tools: next(responses))
    monkeypatch.setattr(
        llm_client,
        "chat_messages_stream",
        lambda messages: iter([
            "核心结论：Redis 持久化包括 RDB 和 AOF。",
            "回答思路：先说明两者区别，再补充混合持久化。",
        ]),
    )

    req = ChatRequest(user_id="u1", session_id="s-stream", question="Redis 持久化有哪些方式？")
    chunks = list(chat_service.chat_stream(req))
    events = _parse_sse_chunks(chunks)
    event_types = [event["type"] for event in events]
    answer_deltas = [event["delta"] for event in events if event["type"] == "answer_delta"]
    final_event = next(event for event in events if event["type"] == "final")

    assert event_types[0] == "start"
    assert "tool_result" in event_types
    assert "answer_delta" in event_types
    assert event_types.index("answer_delta") < event_types.index("final")
    assert answer_deltas == [
        "核心结论：Redis 持久化包括 RDB 和 AOF。",
        "回答思路：先说明两者区别，再补充混合持久化。",
    ]
    assert "".join(answer_deltas) == final_event["result"]["answer"]
    assert final_event["result"]["route"] == "agent_answer"
    assert final_event["result"]["tool_calls"][0]["tool_name"] == "search_knowledge"
    assert final_event["result"]["agent_steps"]


def test_agent_retries_tool_then_recovers(monkeypatch):
    attempts = {"count": 0}

    def flaky_search(query, top_k=3):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary vector failure")
        return {
            "query": query,
            "results": [{
                "id": "303",
                "chunk_id": "3001",
                "title": "调优检索",
                "summary": "调优检索需要关注召回、排序和上下文构造。",
                "tags": ["RAG"],
                "score": 0.93,
            }]
        }

    monkeypatch.setattr(interview_toolkit, "search_knowledge", flaky_search)
    monkeypatch.setattr(
        llm_client,
        "chat_with_tools",
        lambda messages, tools: {
            "content": "先尝试检索，再整理答案" if len(messages) == 2 else "最终结论：检索系统要做好召回、重排和上下文拼装。",
            "tool_calls": [{
                "id": "call_retry_1",
                "name": "search_knowledge",
                "arguments": json.dumps({"query": "调优检索", "top_k": 1}, ensure_ascii=False)
            }] if len(messages) == 2 else []
        }
    )

    agent = AgentOrchestrator()
    result = agent.run("explain retrieval tuning")

    assert attempts["count"] == 2
    assert result.tool_calls[0]["tool_output"]["retry_recovered"] is True
    assert result.route == "agent_answer"


