"""Streamlit chat UI backed by Ollama or Google Gemini."""

from __future__ import annotations

import base64

import streamlit as st

from folder_picker import pick_folder
from localhost import is_localhost_request
from gemini_client import chat as gemini_chat
from gemini_client import list_models as list_gemini_models
from mcp_client import call_tool, get_server_details
from mcp_config import list_mcp_server_names
from ollama_client import chat as ollama_chat
from ollama_client import chat_with_tools as ollama_chat_with_tools
from ollama_client import generate_image as ollama_generate_image
from ollama_client import get_model_info as get_ollama_model_info
from ollama_client import list_embedding_models
from ollama_client import list_models as list_ollama_models
from rag_store import RagStore
from web_search import extract_web_search_query, is_web_search_only_request, search_web

show_sidebar = is_localhost_request()

st.set_page_config(
    page_title="LLM Chat",
    layout="wide",
    initial_sidebar_state="expanded" if show_sidebar else "collapsed",
)

_sidebar_css = (
    """
    [data-testid="stSidebar"] { min-width: 320px; max-width: 360px; }
    """
    if show_sidebar
    else """
    [data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    section[data-testid="stMain"] > div {
        max-width: 100% !important;
    }
    """
)

st.markdown(
    """
    <style>
    """
    + _sidebar_css
    + """
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
    .web-source {
        border-left: 3px solid #cbd5e0;
        margin: 0.35rem 0;
        padding-left: 0.75rem;
    }
    .web-source-title {
        font-size: 0.9rem;
        font-weight: 600;
    }
    .web-source-snippet {
        color: #4a5568;
        font-size: 0.82rem;
        margin-top: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "gemini_models" not in st.session_state:
    st.session_state.gemini_models = []
if "mcp_details" not in st.session_state:
    st.session_state.mcp_details = None
if "rag_index_info" not in st.session_state:
    st.session_state.rag_index_info = None


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


def model_supports_tool_calling(provider: str, model: str) -> bool:
    if provider == "Google Gemini":
        return True
    info = cached_ollama_model_info(model)
    return "tools" in (info.get("capabilities") or [])


def model_supports_rag(provider: str, model: str, *, is_image_mode: bool) -> bool:
    if is_image_mode:
        return False
    if provider == "Google Gemini":
        return True
    info = cached_ollama_model_info(model)
    return "completion" in (info.get("capabilities") or [])


def build_rag_context(query: str, index_info: dict, embedding_model: str, top_k: int) -> str:
    store = RagStore()
    matches = store.query(
        index_info["collection_name"],
        query,
        embedding_model,
        top_k=top_k,
    )
    if not matches:
        return ""

    sections: list[str] = []
    for match in matches:
        sections.append(
            f"Source: {match['source']} (chunk {match['chunk_index']})\n{match['text']}"
        )
    return "\n\n---\n\n".join(sections)


def augment_prompt_with_rag(user_prompt: str, rag_context: str) -> str:
    if not rag_context.strip():
        return user_prompt
    return (
        "Use the following retrieved context when answering. "
        "If the context is not relevant, say so.\n\n"
        f"{rag_context}\n\n"
        f"Question: {user_prompt}"
    )


def render_mcp_details_panel(details: dict) -> None:
    if not details:
        return

    st.subheader("MCP server")
    if details.get("error"):
        st.error(details["error"])
        return

    st.caption(
        f"**{details['name']}** · {len(details.get('tools', []))} tools · "
        f"{len(details.get('resources', []))} resources · "
        f"{len(details.get('prompts', []))} prompts"
    )

    if details.get("description"):
        st.write(details["description"])

    with st.expander("Tools", expanded=True):
        if details.get("tools"):
            for tool in details["tools"]:
                st.markdown(f"**{tool['name']}** — {tool.get('description', '')}")
        else:
            st.write("No tools exposed by this server.")

    with st.expander("Resources"):
        if details.get("resources"):
            for resource in details["resources"]:
                label = resource.get("name") or resource.get("uri")
                st.markdown(f"**{label}** — {resource.get('description', '')}")
        else:
            st.write("No resources exposed by this server.")

    with st.expander("Prompts"):
        if details.get("prompts"):
            for prompt in details["prompts"]:
                st.markdown(f"**{prompt['name']}** — {prompt.get('description', '')}")
        else:
            st.write("No prompts exposed by this server.")


def render_rag_details_panel(index_info: dict) -> None:
    if not index_info:
        return

    st.subheader("Vector DB")
    st.caption(
        f"**{index_info.get('file_count', 0)} files** · "
        f"**{index_info.get('chunk_count', 0)} chunks** · "
        f"model `{index_info.get('embedding_model', '')}`"
    )
    st.write(f"Folder: `{index_info.get('folder', '')}`")
    st.write(f"Collection: `{index_info.get('collection_name', '')}`")
    st.write(f"Persisted at: `{index_info.get('persist_dir', '')}`")

    with st.expander("Indexed files", expanded=False):
        for source in index_info.get("sources", []):
            st.write(f"- `{source}`")


def render_web_sources(sources: list[dict]) -> None:
    if not sources:
        st.caption("No web results found for this query.")
        return

    for index, source in enumerate(sources, start=1):
        title = source.get("title") or f"Source {index}"
        url = source.get("url") or ""
        snippet = source.get("snippet") or ""
        st.markdown(
            f"""
            <div class="web-source">
                <div class="web-source-title">[{index}] <a href="{url}" target="_blank">{title}</a></div>
                <div class="web-source-snippet">{snippet}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_message(message: dict) -> None:
    if message.get("content"):
        st.markdown("**LLM**")
        st.markdown(message["content"])
    if message.get("web_sources") is not None:
        st.markdown("**Web search**")
        render_web_sources(message["web_sources"])
    if message.get("image_b64"):
        st.image(base64.b64decode(message["image_b64"]), width="stretch")


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


def default_ollama_model() -> str:
    models = list_ollama_models()
    if not models:
        st.error("No Ollama models found. Start Ollama and pull a model first.")
        st.stop()
    for model in models:
        if not model_supports_image_generation(model):
            return model
    return models[0]


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


if show_sidebar:
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
            if st.button("Load models", width="stretch"):
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

            st.divider()
            web_search_enabled = st.toggle(
                "Web search",
                value=True,
                key="web_search_enabled",
                help="Search the web and show results in a separate section.",
            )
            if web_search_enabled:
                web_search_max_results = st.slider(
                    "Web results",
                    min_value=3,
                    max_value=10,
                    value=5,
                    step=1,
                    key="web_search_max_results",
                )
            else:
                web_search_max_results = 5

        supports_tools = (
            selected_model is not None
            and model_supports_tool_calling(provider, selected_model)
            and not is_image_mode
        )
        supports_rag = (
            selected_model is not None
            and model_supports_rag(provider, selected_model, is_image_mode=is_image_mode)
        )

        selected_mcp_server: str | None = None
        rag_folder_path = ""
        rag_embedding_model: str | None = None

        if supports_tools:
            st.divider()
            st.subheader("MCP server")
            mcp_servers = list_mcp_server_names()
            if not mcp_servers:
                st.info("Add servers in `chatbot/mcp_servers.json` or `~/.cursor/mcp.json`.")
            else:
                selected_mcp_server = st.selectbox(
                    "MCP server",
                    mcp_servers,
                    key="mcp_server",
                )
                if st.button("Load MCP details", width="stretch"):
                    with st.spinner("Connecting to MCP server..."):
                        st.session_state.mcp_details = get_server_details(selected_mcp_server)

        if supports_rag:
            st.divider()
            st.subheader("RAG / Vector DB")
            browse_col, folder_col = st.columns([1, 5])
            with browse_col:
                st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
                if st.button("📁", key="rag_folder_browse", width="stretch", help="Choose folder"):
                    selected = pick_folder(st.session_state.get("rag_folder_path", ""))
                    if selected:
                        st.session_state.rag_folder_path = selected
                        st.rerun()
            with folder_col:
                rag_folder_path = st.text_input(
                    "Folder path",
                    placeholder="Click 📁 or enter a path",
                    key="rag_folder_path",
                    help="Click the folder icon to open the native folder picker.",
                )
            embedding_models = list_embedding_models()
            if not embedding_models:
                st.warning("Pull an embedding model first, e.g. `ollama pull nomic-embed-text`.")
            else:
                rag_embedding_model = st.selectbox(
                    "Embedding model",
                    embedding_models,
                    key="rag_embedding_model",
                )
                if st.button("Build vector DB", width="stretch"):
                    if not rag_folder_path.strip():
                        st.error("Enter a folder path to index.")
                    else:
                        with st.spinner("Indexing documents..."):
                            try:
                                store = RagStore()
                                st.session_state.rag_index_info = store.build_from_folder(
                                    rag_folder_path.strip(),
                                    rag_embedding_model,
                                )
                            except Exception as exc:
                                st.error(str(exc))

            if st.session_state.rag_index_info:
                info = st.session_state.rag_index_info
                st.caption(
                    f"Indexed **{info.get('file_count', 0)}** files · "
                    f"**{info.get('chunk_count', 0)}** chunks"
                )

        st.button("Clear chat", on_click=clear_chat, width="stretch")
else:
    provider = "Ollama"
    selected_model = default_ollama_model()
    is_image_mode = model_supports_image_generation(selected_model)
    image_width = 1024
    image_height = 1024
    image_steps = 0
    system_prompt = ""
    temperature = 0.7
    top_k = 5
    web_search_enabled = True
    web_search_max_results = 5
    supports_tools = False
    supports_rag = False
    selected_mcp_server = None
    rag_folder_path = ""
    rag_embedding_model = None

if provider == "Ollama" and selected_model:
    render_ollama_capabilities(selected_model)

detail_cols = st.columns(2)
with detail_cols[0]:
    if supports_tools and st.session_state.mcp_details:
        render_mcp_details_panel(st.session_state.mcp_details)
with detail_cols[1]:
    if supports_rag and st.session_state.rag_index_info:
        render_rag_details_panel(st.session_state.rag_index_info)

st.title("Image" if is_image_mode else "Chat")
st.caption(f"**{provider}** · **{selected_model}**")
if not show_sidebar:
    st.caption("Settings are available when you open the app at `http://localhost:8501`.")

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
            web_sources: list[dict] | None = None
            reply: str | None = None
            web_only = is_web_search_only_request(prompt)
            try:
                if web_only:
                    search_query = extract_web_search_query(prompt) or prompt
                    with st.spinner("Searching the web..."):
                        web_sources = search_web(
                            search_query,
                            max_results=web_search_max_results,
                        )
                else:
                    chat_prompt = prompt
                    index_info = st.session_state.rag_index_info
                    embedding_model = (
                        rag_embedding_model
                        or (index_info or {}).get("embedding_model")
                    )
                    if supports_rag and index_info and embedding_model:
                        with st.spinner("Searching knowledge base..."):
                            rag_context = build_rag_context(
                                prompt,
                                index_info,
                                embedding_model,
                                top_k=top_k,
                            )
                        chat_prompt = augment_prompt_with_rag(prompt, rag_context)

                    if web_search_enabled:
                        with st.spinner("Searching the web..."):
                            web_sources = search_web(
                                prompt,
                                max_results=web_search_max_results,
                            )

                    with st.spinner("Thinking..."):
                        mcp_details = st.session_state.mcp_details
                        use_mcp_tools = (
                            provider == "Ollama"
                            and supports_tools
                            and selected_mcp_server
                            and mcp_details
                            and mcp_details.get("ollama_tools")
                            and not mcp_details.get("error")
                        )

                        if use_mcp_tools:
                            server_name = selected_mcp_server

                            def handle_tool(tool_name: str, arguments: dict) -> str:
                                return call_tool(server_name, tool_name, arguments)

                            reply = ollama_chat_with_tools(
                                model=selected_model,
                                user_prompt=chat_prompt,
                                tools=mcp_details["ollama_tools"],
                                tool_handler=handle_tool,
                                system_prompt=system_prompt,
                                temperature=temperature,
                                top_k=top_k,
                            )
                        elif provider == "Ollama":
                            reply = ollama_chat(
                                model=selected_model,
                                user_prompt=chat_prompt,
                                system_prompt=system_prompt,
                                temperature=temperature,
                                top_k=top_k,
                            )
                        else:
                            reply = gemini_chat(
                                api_key=st.session_state.gemini_api_key,
                                model=selected_model,
                                user_prompt=chat_prompt,
                                system_prompt=system_prompt,
                                temperature=temperature,
                                top_k=top_k,
                            )
            except Exception as exc:
                reply = f"Error: {exc}"
                web_sources = web_sources or []

            assistant_message = {
                "role": "assistant",
                "content": reply,
                "web_sources": web_sources if (web_only or web_search_enabled) else None,
            }
            render_message(assistant_message)

    st.session_state.messages.append(assistant_message)
