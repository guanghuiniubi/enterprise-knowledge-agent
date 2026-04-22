from __future__ import annotations

from eka.retrievers import KeywordKnowledgeBaseRetriever


def test_keyword_retriever_returns_ranked_documents(tmp_path):
    (tmp_path / "agent.md").write_text("LangGraph agent planning and tool calling interview notes", encoding="utf-8")
    (tmp_path / "other.txt").write_text("unrelated content", encoding="utf-8")

    retriever = KeywordKnowledgeBaseRetriever(tmp_path)
    docs = retriever.retrieve("LangGraph tool calling", limit=2)

    assert len(docs) == 1
    assert docs[0].metadata["filename"] == "agent.md"
    assert docs[0].score > 0

