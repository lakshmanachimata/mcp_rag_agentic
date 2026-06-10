"""Client for loading and chatting with local Ollama models."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable

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


def list_models_by_capability(capability: str, host: str = DEFAULT_HOST) -> list[str]:
    """Return model names that report a given Ollama capability."""
    try:
        request = urllib.request.Request(
            f"{host.rstrip('/')}/api/tags",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return []

    names: list[str] = []
    for model in payload.get("models", []):
        name = model.get("model") or model.get("name")
        capabilities = model.get("capabilities") or []
        if name and capability in capabilities:
            names.append(name)
    return sorted(names)


def list_embedding_models(host: str = DEFAULT_HOST) -> list[str]:
    return list_models_by_capability("embedding", host=host)


def embed_texts(
    model: str,
    texts: list[str],
    host: str = DEFAULT_HOST,
) -> list[list[float]]:
    """Return embedding vectors for a batch of texts."""
    if not texts:
        return []

    response = get_client(host).embed(model=model, input=texts)
    raw = _as_dict(response)
    embeddings = raw.get("embeddings") or getattr(response, "embeddings", None) or []
    return [list(vector) for vector in embeddings]


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


def _message_to_dict(message: object) -> dict:
    if isinstance(message, dict):
        return message
    return _as_dict(message)


def chat(
    model: str,
    user_prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    top_k: int = 40,
    host: str = DEFAULT_HOST,
) -> str:
    """Send a chat request to Ollama and return the assistant reply."""
    messages: list[dict] = []
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


def chat_with_tools(
    model: str,
    user_prompt: str,
    tools: list[dict],
    tool_handler: Callable[[str, dict], str],
    system_prompt: str = "",
    temperature: float = 0.7,
    top_k: int = 40,
    host: str = DEFAULT_HOST,
    max_rounds: int = 5,
) -> str:
    """Run a tool-calling chat loop and return the final assistant reply."""
    messages: list[dict] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.append({"role": "user", "content": user_prompt})

    for _ in range(max_rounds):
        response = get_client(host).chat(
            model=model,
            messages=messages,
            tools=tools,
            options={
                "temperature": temperature,
                "top_k": top_k,
            },
        )
        message = _message_to_dict(getattr(response, "message", None) or response.get("message", {}))
        messages.append(message)

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            return message.get("content", "")

        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            tool_name = function.get("name", "")
            raw_arguments = function.get("arguments", "{}")
            arguments = (
                json.loads(raw_arguments)
                if isinstance(raw_arguments, str)
                else dict(raw_arguments)
            )
            tool_result = tool_handler(tool_name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "content": tool_result,
                    "name": tool_name,
                }
            )

    return messages[-1].get("content", "Tool loop limit reached.")
