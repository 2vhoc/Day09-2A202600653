from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings
from provider.custom import build_custom_model
from provider.gemini import build_gemini_model
from provider.ollama import build_ollama_model
from provider.openai import build_openai_model
from provider.openrouter import build_openrouter_model


def get_chat_model(settings: Settings) -> BaseChatModel:
    builders = {
        "gemini": build_gemini_model,
        "openai": build_openai_model,
        "openrouter": build_openrouter_model,
        "ollama": build_ollama_model,
        "custom": build_custom_model,
    }
    try:
        builder = builders[settings.provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {settings.provider}") from exc
    return builder(settings)
