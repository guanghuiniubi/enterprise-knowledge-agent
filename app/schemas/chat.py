from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    question: str = Field(..., description="用户问题")


class Citation(BaseModel):
    doc_id: str
    title: str
    snippet: str


class ToolCall(BaseModel):
    tool_name: str
    tool_input: dict
    tool_output: dict


class ChatResponse(BaseModel):
    answer: str
    route: str
    citations: List[Citation] = []
    tool_calls: List[ToolCall] = []
    session_id: str
    debug: Optional[dict] = None
