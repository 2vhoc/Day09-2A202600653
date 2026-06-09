from __future__ import annotations

from langchain_ollama import ChatOllama

from app.config import Settings


def build_ollama_model(settings: Settings) -> ChatOllama:
    return ChatOllama(
        model=settings.model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
    )
