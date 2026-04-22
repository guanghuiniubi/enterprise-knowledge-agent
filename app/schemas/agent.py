from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RouteResult(BaseModel):
    route: str
    reason: str
    missing_slots: List[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    answer: str
    route: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    agent_steps: List[Dict[str, Any]] = Field(default_factory=list)
    debug: Dict[str, Any] = Field(default_factory=dict)
    need_clarification: bool = False
    clarification_question: Optional[str] = None
