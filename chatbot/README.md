# Chatbot

Streamlit chat UI with local **Ollama** or cloud **Google Gemini**, plus MCP tools, RAG over local files, and web search.

## Prerequisites

- Python 3.11+ (3.13 works with the project venv)
- [Ollama](https://ollama.com/) (optional, for local models)
- [Google AI API key](https://aistudio.google.com/apikey) (optional, for Gemini)

## Install

From the repo root:

```bash
cd chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Ollama, install the app and pull at least one model:

```bash
ollama pull llama3.2:3b
```

## Launch

Activate the virtual environment, then start Streamlit:

```bash
cd chatbot
source .venv/bin/activate
streamlit run app.py
```

The app opens in your browser (default: `http://localhost:8501`).

### Using the app

**Sidebar (localhost only)**

- **Provider** — choose Ollama or Google Gemini
- **Ollama** — models are loaded automatically from your local instance
- **Gemini** — enter your API key, click **Load models**, then pick a model
- **System prompt**, **Temperature** (0–1), **Top K** (1–10)
- MCP servers, RAG folder indexing, and related settings

**Main area**

- Chat history appears above
- Enter prompts in the input at the bottom

## Debug

VS Code / Cursor debug configs live in the repo root at `.vscode/launch.json`.

1. Open the workspace root (`mcp_rag_agentic`), not only the `chatbot` folder
2. Ensure the interpreter is `chatbot/.venv/bin/python` (set in `.vscode/settings.json`)
3. Open **Run and Debug** (Cmd+Shift+D)
4. Select **Chatbot: Streamlit App**
5. Press **F5**

The debugger starts Streamlit on port **8502** at `http://localhost:8502`.

Set breakpoints in any module, for example:

- `gemini_client.py` — Gemini model listing and chat
- `ollama_client.py` — Ollama model listing and chat
- `app.py` — UI flow and provider selection

Then use the app in the browser (load Gemini models, send a message, etc.) to hit your breakpoints.

### Manual debug without VS Code

```bash
cd chatbot
source .venv/bin/activate
python -m debugpy --listen 5678 --wait-for-client -m streamlit run app.py --server.port=8502
```

Attach your debugger to port `5678`.

## Project layout

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI |
| `ollama_client.py` | Ollama API client |
| `gemini_client.py` | Google Gemini API client |
| `mcp_client.py` / `mcp_config.py` | MCP server integration |
| `rag_store.py` | Local document RAG |
| `requirements.txt` | Python dependencies |
