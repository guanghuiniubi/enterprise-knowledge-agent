from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel


class BaseTool(ABC):
    """Base abstraction for project tools that can be exposed to LangChain."""

    name: str
    description: str
    args_schema: type[BaseModel] | None = None

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        raise NotImplementedError

    async def arun(self, **kwargs: Any) -> str:
        return self.run(**kwargs)

    def as_langchain_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.run,
            coroutine=self.arun,
            name=self.name,
            description=self.description,
            args_schema=self.args_schema,
        )

