"""Client for loading and chatting with local Ollama models."""

from __future__ import annotations

from ollama import Client

DEFAULT_HOST = "http://localhost:11434"


def get_client(host: str = DEFAULT_HOST) -> Client:
    return Client(host=host)


def _as_dict(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def get_model_info(model: str, host: str = DEFAULT_HOST) -> dict:
    """Return capabilities and details for an Ollama model."""
    try:
        response = get_client(host).show(model=model)
        raw = _as_dict(response)
        return {
            "capabilities": list(raw.get("capabilities") or []),
            "details": _as_dict(raw.get("details")),
        }
    except Exception:
        return {"capabilities": [], "details": {}}


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


def generate_image(
    model: str,
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = 0,
    host: str = DEFAULT_HOST,
) -> str:
    """Generate an image from a text prompt. Returns base64-encoded PNG data."""
    kwargs: dict = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "width": width,
        "height": height,
    }
    if steps > 0:
        kwargs["steps"] = steps

    response = get_client(host).generate(**kwargs)
    raw = _as_dict(response)
    image = raw.get("image") or getattr(response, "image", None) or ""
    return image


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
