from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import Settings


def build_openai_model(settings: Settings) -> ChatOpenAI:
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for provider=openai")
    return ChatOpenAI(
        model=settings.model,
        api_key=settings.openai_api_key,
        temperature=settings.temperature,
    )
