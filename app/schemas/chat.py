from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    question: str = Field(..., description="用户问题")
    user_roles: List[str] = Field(default_factory=list, description="用户角色")
    user_departments: List[str] = Field(default_factory=list, description="用户部门")
    clearance_level: int = Field(default=0, description="用户密级等级")


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
    citations: List[Citation] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    agent_steps: List[dict] = Field(default_factory=list)
    session_id: str
    need_clarification: bool = False
    clarification_question: Optional[str] = None
    debug: Optional[dict] = None
