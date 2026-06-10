"""MCP client helpers for discovering and calling server tools."""

from __future__ import annotations

import asyncio
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_config import load_mcp_servers


def _tool_to_dict(tool: Any) -> dict:
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": getattr(tool, "inputSchema", None) or {},
    }


def _resource_to_dict(resource: Any) -> dict:
    return {
        "name": getattr(resource, "name", "") or "",
        "uri": str(getattr(resource, "uri", "") or ""),
        "description": getattr(resource, "description", "") or "",
    }


def _prompt_to_dict(prompt: Any) -> dict:
    return {
        "name": prompt.name,
        "description": prompt.description or "",
    }


def _mcp_tool_to_ollama(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"] or {"type": "object", "properties": {}},
        },
    }


async def _fetch_server_details_async(server_config: dict) -> dict:
    params = StdioServerParameters(
        command=server_config["command"],
        args=server_config.get("args") or [],
        env=server_config.get("env") or None,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            resources_result = await session.list_resources()
            prompts_result = await session.list_prompts()

            tools = [_tool_to_dict(tool) for tool in tools_result.tools]
            resources = [_resource_to_dict(resource) for resource in resources_result.resources]
            prompts = [_prompt_to_dict(prompt) for prompt in prompts_result.prompts]

            return {
                "name": server_config["name"],
                "description": server_config.get("description", ""),
                "command": server_config["command"],
                "args": server_config.get("args") or [],
                "tools": tools,
                "resources": resources,
                "prompts": prompts,
                "ollama_tools": [_mcp_tool_to_ollama(tool) for tool in tools],
            }


async def _call_tool_async(server_config: dict, tool_name: str, arguments: dict) -> str:
    params = StdioServerParameters(
        command=server_config["command"],
        args=server_config.get("args") or [],
        env=server_config.get("env") or None,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)

    parts: list[str] = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts) if parts else str(result.content)


def get_server_details(server_name: str) -> dict:
    """Connect to an MCP server and return tools, resources, and prompts."""
    servers = load_mcp_servers()
    if server_name not in servers:
        raise ValueError(f"MCP server '{server_name}' is not configured.")

    try:
        return asyncio.run(_fetch_server_details_async(servers[server_name]))
    except Exception as exc:
        return {
            "name": server_name,
            "error": str(exc),
            "tools": [],
            "resources": [],
            "prompts": [],
            "ollama_tools": [],
        }


def call_tool(server_name: str, tool_name: str, arguments: dict) -> str:
    servers = load_mcp_servers()
    if server_name not in servers:
        raise ValueError(f"MCP server '{server_name}' is not configured.")
    return asyncio.run(_call_tool_async(servers[server_name], tool_name, arguments))
