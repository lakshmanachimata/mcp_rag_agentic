"""Streamlit chat UI backed by Ollama or Google Gemini."""

from __future__ import annotations

import base64

import streamlit as st

from gemini_client import chat as gemini_chat
from gemini_client import list_models as list_gemini_models
from ollama_client import chat as ollama_chat
from ollama_client import generate_image as ollama_generate_image
from ollama_client import get_model_info as get_ollama_model_info
from ollama_client import list_models as list_ollama_models

st.set_page_config(page_title="LLM Chat", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { min-width: 320px; max-width: 360px; }
    .block-container { padding-top: 1.5rem; }
    .model-capabilities {
        background: #f0f4f8;
        border: 1px solid #d8dee9;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
    }
    .model-capabilities-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .capability-badges { display: flex; flex-wrap: wrap; gap: 0.4rem; }
    .capability-badge {
        background: #e2e8f0;
        border-radius: 999px;
        color: #2d3748;
        font-size: 0.8rem;
        font-weight: 500;
        padding: 0.2rem 0.65rem;
    }
    .model-meta {
        color: #718096;
        font-size: 0.8rem;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "gemini_models" not in st.session_state:
    st.session_state.gemini_models = []


def clear_chat() -> None:
    st.session_state.messages = []


CAPABILITY_LABELS = {
    "completion": "Text completion",
    "vision": "Vision",
    "image": "Text to image",
    "tools": "Tool calling",
    "embedding": "Embeddings",
    "thinking": "Thinking",
}


@st.cache_data(ttl=300, show_spinner=False)
def cached_ollama_model_info(model: str) -> dict:
    return get_ollama_model_info(model)


def model_supports_image_generation(model: str) -> bool:
    info = cached_ollama_model_info(model)
    return "image" in (info.get("capabilities") or [])


def render_message(message: dict) -> None:
    if message.get("content"):
        st.markdown(message["content"])
    if message.get("image_b64"):
        st.image(base64.b64decode(message["image_b64"]), use_container_width=True)


def render_ollama_capabilities(model: str) -> None:
    info = cached_ollama_model_info(model)
    capabilities = info.get("capabilities") or []
    details = info.get("details") or {}

    if not capabilities and not details:
        return

    badges = "".join(
        f'<span class="capability-badge">{CAPABILITY_LABELS.get(cap, cap.replace("_", " ").title())}</span>'
        for cap in capabilities
    )

    meta_parts: list[str] = []
    for key, label in (
        ("family", "Family"),
        ("parameter_size", "Size"),
        ("quantization_level", "Quantization"),
    ):
        value = details.get(key)
        if value:
            meta_parts.append(f"{label}: {value}")
    meta_html = (
        f'<div class="model-meta">{" · ".join(meta_parts)}</div>' if meta_parts else ""
    )

    st.markdown(
        f"""
        <div class="model-capabilities">
            <div class="model-capabilities-title">Model capabilities · {model}</div>
            <div class="capability-badges">{badges or '<span class="capability-badge">Unknown</span>'}</div>
            {meta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_gemini_models(api_key: str) -> None:
    if not api_key.strip():
        st.session_state.gemini_models = []
        st.session_state.gemini_load_error = "Enter an API key first."
        return
    try:
        st.session_state.gemini_models = list_gemini_models(api_key)
        if not st.session_state.gemini_models:
            st.session_state.gemini_load_error = "No chat models returned. Check your API key."
        else:
            st.session_state.pop("gemini_load_error", None)
    except Exception as exc:
        st.session_state.gemini_models = []
        st.session_state.gemini_load_error = str(exc)


with st.sidebar:
    st.title("Settings")

    provider = st.radio(
        "Provider",
        ["Ollama", "Google Gemini"],
        key="provider",
        horizontal=True,
    )

    selected_model: str | None = None

    if provider == "Ollama":
        models = list_ollama_models()
        if not models:
            st.error("No Ollama models found. Start Ollama and pull a model first.")
            st.stop()
        selected_model = st.selectbox("Model", models, key="ollama_model")
    else:
        gemini_api_key = st.text_input(
            "Gemini API key",
            type="password",
            placeholder="Enter your Google AI API key",
            key="gemini_api_key",
        )
        if st.button("Load models", use_container_width=True):
            load_gemini_models(gemini_api_key)

        load_error = st.session_state.get("gemini_load_error")
        if load_error:
            st.error(f"Failed to load models: {load_error}")

        gemini_models = st.session_state.gemini_models
        if not gemini_models:
            st.info("Enter your API key and click **Load models**.")
            st.stop()

        selected_model = st.selectbox("Model", gemini_models, key="gemini_model")

    is_image_mode = (
        provider == "Ollama"
        and selected_model is not None
        and model_supports_image_generation(selected_model)
    )

    if is_image_mode:
        image_width = st.slider(
            "Width",
            min_value=256,
            max_value=1024,
            value=1024,
            step=64,
            key="image_width",
        )
        image_height = st.slider(
            "Height",
            min_value=256,
            max_value=1024,
            value=1024,
            step=64,
            key="image_height",
        )
        image_steps = st.slider(
            "Steps",
            min_value=0,
            max_value=50,
            value=0,
            step=1,
            help="0 uses the model's recommended step count.",
            key="image_steps",
        )
    else:
        system_prompt = st.text_area(
            "System prompt",
            height=160,
            placeholder="You are a helpful assistant.",
            key="system_prompt",
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            key="temperature",
        )

        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="top_k",
        )

    st.button("Clear chat", on_click=clear_chat, use_container_width=True)

if provider == "Ollama" and selected_model:
    render_ollama_capabilities(selected_model)

st.title("Image" if is_image_mode else "Chat")
st.caption(f"**{provider}** · **{selected_model}**")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_message(message)

input_placeholder = (
    "Describe the image you want to generate..."
    if is_image_mode
    else "Enter your prompt..."
)
if prompt := st.chat_input(input_placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if is_image_mode:
            with st.spinner("Generating image..."):
                try:
                    image_b64 = ollama_generate_image(
                        model=selected_model,
                        prompt=prompt,
                        width=image_width,
                        height=image_height,
                        steps=image_steps,
                    )
                    if not image_b64:
                        raise RuntimeError("No image returned by the model.")
                    assistant_message = {"role": "assistant", "image_b64": image_b64}
                except Exception as exc:
                    assistant_message = {
                        "role": "assistant",
                        "content": f"Error: {exc}",
                    }
            render_message(assistant_message)
        else:
            with st.spinner("Thinking..."):
                try:
                    if provider == "Ollama":
                        reply = ollama_chat(
                            model=selected_model,
                            user_prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=temperature,
                            top_k=top_k,
                        )
                    else:
                        reply = gemini_chat(
                            api_key=st.session_state.gemini_api_key,
                            model=selected_model,
                            user_prompt=prompt,
                            system_prompt=system_prompt,
                            temperature=temperature,
                            top_k=top_k,
                        )
                except Exception as exc:
                    reply = f"Error: {exc}"
            st.markdown(reply)
            assistant_message = {"role": "assistant", "content": reply}

    st.session_state.messages.append(assistant_message)
