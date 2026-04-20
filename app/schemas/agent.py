from typing import Any, Dict, List
from pydantic import BaseModel


class RouteResult(BaseModel):
    route: str
    reason: str


class AgentResult(BaseModel):
    answer: str
    route: str
    citations: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {}
