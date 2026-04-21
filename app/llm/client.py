from openai import OpenAI
from langsmith import traceable
from app.core.config import settings
from app.llm.base import BaseLLM


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url
        )
        self.model = settings.llm_model

    @traceable(name="llm_chat", run_type="llm")
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2
        )
        return resp.choices[0].message.content or ""


llm_client = OpenAICompatibleLLM()
