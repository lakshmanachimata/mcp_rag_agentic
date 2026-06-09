"""Client for Google Gemini models via the Gemini API."""

from __future__ import annotations

from google import genai
from google.genai import types


def _supports_generate_content(model) -> bool:
    actions = (
        getattr(model, "supported_actions", None)
        or getattr(model, "supported_generation_methods", None)
        or []
    )
    if not actions:
        name = (model.name or "").lower()
        return "gemini" in name and "embed" not in name
    return "generateContent" in actions


def list_models(api_key: str) -> list[str]:
    """Return Gemini model names that support generateContent."""
    if not api_key.strip():
        return []

    names: list[str] = []
    with genai.Client(api_key=api_key) as client:
        for model in client.models.list():
            if not _supports_generate_content(model):
                continue
            name = model.name or ""
            if name.startswith("models/"):
                name = name[len("models/") :]
            if name:
                names.append(name)
    return sorted(names)


def chat(
    api_key: str,
    model: str,
    user_prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    top_k: int = 5,
) -> str:
    """Send a chat request to Gemini and return the assistant reply."""
    config_kwargs: dict = {
        "temperature": temperature,
        "top_k": top_k,
    }
    if system_prompt.strip():
        config_kwargs["system_instruction"] = system_prompt.strip()

    with genai.Client(api_key=api_key) as client:
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )
    return response.text or ""
