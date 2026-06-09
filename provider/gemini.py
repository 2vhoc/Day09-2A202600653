from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import Settings


def build_gemini_model(settings: Settings) -> ChatGoogleGenerativeAI:
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is required for provider=gemini")
    return ChatGoogleGenerativeAI(
        model=settings.model,
        google_api_key=settings.google_api_key,
        temperature=settings.temperature,
    )
