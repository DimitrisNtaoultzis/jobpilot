"""Chat model factory: the one place that knows which LLM provider we use.

The rest of the service depends only on LangChain's generic BaseChatModel, so the
provider can change here, and tests can swap in a fake model.
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq

from .config import get_settings


def get_chat_model() -> BaseChatModel:
    """Return the configured chat model (Groq)."""
    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model,
        # Passed explicitly: pydantic-settings reads .env without putting it in os.environ.
        api_key=settings.groq_api_key,
        temperature=0,  # stable output: the same input should give the same extraction
        reasoning_effort="low",  # fewer reasoning tokens (we have an 8K tokens/minute limit)
        timeout=60,  # seconds; every network call needs a timeout
        max_retries=2,  # retried on transient errors such as 429 or 5xx
    )
