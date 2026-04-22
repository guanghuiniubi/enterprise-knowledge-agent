from abc import ABC, abstractmethod
from typing import Iterator


class BaseLLM(ABC):
    @abstractmethod
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def chat_messages(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError

    @abstractmethod
    def chat_messages_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        raise NotImplementedError

    @abstractmethod
    def chat_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        raise NotImplementedError

