import os
from app.core.config import settings


def setup_langsmith():
    if settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key or ""
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
