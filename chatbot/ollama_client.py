"""Client for loading and chatting with local Ollama models."""

from __future__ import annotations

from ollama import Client

DEFAULT_HOST = "http://localhost:11434"


def get_client(host: str = DEFAULT_HOST) -> Client:
    return Client(host=host)


def list_models(host: str = DEFAULT_HOST) -> list[str]:
    """Return names of models available in the local Ollama instance."""
    try:
        response = get_client(host).list()
        models = getattr(response, "models", None) or response.get("models", [])
        names: list[str] = []
        for model in models:
            name = getattr(model, "model", None) or model.get("name") or model.get("model")
            if name:
                names.append(name)
        return sorted(names)
    except Exception:
        return []


def chat(
    model: str,
    user_prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    top_k: int = 40,
    host: str = DEFAULT_HOST,
) -> str:
    """Send a chat request to Ollama and return the assistant reply."""
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt})

    response = get_client(host).chat(
        model=model,
        messages=messages,
        options={
            "temperature": temperature,
            "top_k": top_k,
        },
    )
    message = getattr(response, "message", None) or response.get("message", {})
    content = getattr(message, "content", None) or message.get("content", "")
    return content
