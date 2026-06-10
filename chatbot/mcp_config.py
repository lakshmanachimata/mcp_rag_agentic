"""Load MCP server definitions from local config files."""

from __future__ import annotations

import json
from pathlib import Path

CHATBOT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CHATBOT_DIR / "mcp_servers.json"
CURSOR_CONFIG_PATH = Path.home() / ".cursor" / "mcp.json"


def _normalize_server(name: str, config: dict) -> dict | None:
    command = config.get("command")
    if not command:
        return None
    return {
        "name": name,
        "command": command,
        "args": list(config.get("args") or []),
        "env": dict(config.get("env") or {}),
        "description": config.get("description", ""),
    }


def _parse_config_file(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if "mcpServers" in data:
        raw_servers = data["mcpServers"]
    else:
        raw_servers = data

    servers: dict[str, dict] = {}
    for name, config in raw_servers.items():
        if not isinstance(config, dict):
            continue
        normalized = _normalize_server(name, config)
        if normalized:
            servers[name] = normalized
    return servers


def load_mcp_servers() -> dict[str, dict]:
    """Merge MCP servers from project config and Cursor's mcp.json."""
    servers: dict[str, dict] = {}
    for path in (DEFAULT_CONFIG_PATH, CURSOR_CONFIG_PATH):
        servers.update(_parse_config_file(path))
    return servers


def list_mcp_server_names() -> list[str]:
    return sorted(load_mcp_servers().keys())
