from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import Settings


def build_openrouter_model(settings: Settings) -> ChatOpenAI:
    if not settings.openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY is required for provider=openrouter")
    extra_headers = {}
    if settings.openrouter_site_url:
        extra_headers["HTTP-Referer"] = settings.openrouter_site_url
    if settings.openrouter_app_name:
        extra_headers["X-Title"] = settings.openrouter_app_name
    return ChatOpenAI(
        model=settings.model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=settings.temperature,
        default_headers=extra_headers or None,
    )
