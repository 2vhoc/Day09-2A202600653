from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import Settings


def build_custom_model(settings: Settings) -> ChatOpenAI:
    if not settings.custom_llm_base_url:
        raise ValueError("CUSTOM_LLM_BASE_URL is required for provider=custom")
    if not settings.custom_llm_api_key:
        raise ValueError("CUSTOM_LLM_API_KEY is required for provider=custom")
    model_name = settings.custom_llm_model or settings.model
    return ChatOpenAI(
        model=model_name,
        api_key=settings.custom_llm_api_key,
        base_url=settings.custom_llm_base_url,
        temperature=settings.temperature,
    )
