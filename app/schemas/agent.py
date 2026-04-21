from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class RouteResult(BaseModel):
    route: str
    reason: str
    missing_slots: List[str] = []


class AgentResult(BaseModel):
    answer: str
    route: str
    citations: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {}
    need_clarification: bool = False
    clarification_question: Optional[str] = None
