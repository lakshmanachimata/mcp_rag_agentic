"""Streamlit chat UI backed by Ollama or Google Gemini."""

from __future__ import annotations

import streamlit as st

from gemini_client import chat as gemini_chat
from gemini_client import list_models as list_gemini_models
from ollama_client import chat as ollama_chat
from ollama_client import list_models as list_ollama_models

st.set_page_config(page_title="LLM Chat", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { min-width: 320px; max-width: 360px; }
    .block-container { padding-top: 1.5rem; }
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

st.title("Chat")
st.caption(f"**{provider}** · **{selected_model}**")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your prompt..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
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

    st.session_state.messages.append({"role": "assistant", "content": reply})
