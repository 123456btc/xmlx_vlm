# SPDX-License-Identifier: Apache-2.0
"""
MCP Tool Manager.

Aggregates built-in tools and external MCP servers, exposing them as OpenAI
function schemas for injection into chat completions.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from .client import MCPClient
from .config import load_mcp_config
from .executor import ToolExecutor
from .tools import TOOL_SCHEMAS
from .types import MCPConfig, MCPToolResult

logger = logging.getLogger("xmlx_vlm.mcp.manager")


class MCPManager:
    """Manages MCP tools for a server session."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.executor = ToolExecutor()
        self._external_schemas: List[Dict[str, Any]] = []
        self._external_clients: Dict[str, MCPClient] = {}
        self._external_config: Optional[MCPConfig] = None

    @property
    def schemas(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        return list(TOOL_SCHEMAS) + list(self._external_schemas)

    def add_external_tools(self, schemas: List[Dict[str, Any]]) -> None:
        self._external_schemas.extend(schemas)

    async def connect_external_servers(self, config_path: Optional[str] = None) -> None:
        """Load MCP config and connect to all enabled external servers."""
        if not self.enabled:
            return

        try:
            config = load_mcp_config(config_path)
        except FileNotFoundError:
            logger.info("No MCP config found, skipping external server connection")
            return
        except Exception as e:
            logger.warning(f"Failed to load MCP config: {e}")
            return

        self._external_config = config
        self._external_clients = {}
        self._external_schemas = []

        for name, server_config in config.servers.items():
            client = MCPClient(server_config)
            success = await client.connect()
            if success:
                self._external_clients[name] = client
                for tool in client.tools:
                    self._external_schemas.append(tool.to_openai_format())
            else:
                logger.warning(f"MCP server '{name}' failed to connect")

        logger.info(
            f"MCP external servers: {len(self._external_clients)} connected, "
            f"{len(self._external_schemas)} tools available"
        )

    async def refresh_external_tools(self) -> None:
        """Refresh tool lists from connected external servers."""
        self._external_schemas = []
        for client in self._external_clients.values():
            await client.refresh_tools()
            for tool in client.tools:
                self._external_schemas.append(tool.to_openai_format())

    async def disconnect_all(self) -> None:
        """Disconnect from all external MCP servers."""
        for client in list(self._external_clients.values()):
            await client.disconnect()
        self._external_clients.clear()
        self._external_schemas.clear()

    def get_status(self) -> List[Dict[str, Any]]:
        """Return status of all external MCP servers."""
        statuses = []
        for client in self._external_clients.values():
            statuses.append(client.get_status().to_dict())
        return statuses

    async def execute(self, calls: list[dict]) -> list[dict]:
        if not self.enabled:
            return []

        # Separate built-in calls from external calls
        builtin_calls = []
        external_calls = []

        for call in calls:
            tool_name = call.get("function", {}).get("name", "")
            if "__" in tool_name and tool_name in {
                schema.get("function", {}).get("name", "")
                for schema in self._external_schemas
            }:
                external_calls.append(call)
            else:
                builtin_calls.append(call)

        results = []

        # Execute built-in tools (synchronous)
        if builtin_calls:
            results.extend(self.executor.handle_calls(builtin_calls))

        # Execute external tools (asynchronous)
        for call in external_calls:
            tool_name = call.get("function", {}).get("name", "")
            # Parse server__tool namespace
            if "__" in tool_name:
                server_name, actual_tool = tool_name.split("__", 1)
            else:
                server_name, actual_tool = "", tool_name

            client = self._external_clients.get(server_name)
            if client is None:
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": f"Error: MCP server '{server_name}' not connected",
                    }
                )
                continue

            arguments = call.get("function", {}).get("arguments", {})
            if isinstance(arguments, str):
                try:
                    import json

                    arguments = json.loads(arguments)
                except Exception:
                    arguments = {}

            result = await client.call_tool(actual_tool, arguments)
            results.append(result.to_message(tool_call_id=call.get("id", "")))

        return results

    def format_tools_for_prompt(self) -> str:
        """Return a plain-text tool description for models without native function calling."""
        lines = ["You have access to the following tools:"]
        for schema in self.schemas:
            fn = schema.get("function", {})
            name = fn.get("name", "unknown")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            props = params.get("properties", {})
            req = params.get("required", [])
            args_str = ", ".join(
                f"{k}: {v.get('type', 'any')}{' (required)' if k in req else ''}"
                for k, v in props.items()
            )
            lines.append(f"- {name}({args_str}): {desc}")
        lines.append(
            "When you need to use a tool, output a JSON object like:"
            ' {"tool": "tool_name", "arguments": {"arg1": "value1"}}'
        )
        return "\n".join(lines)


# Global singleton for the server process
_global_manager: Optional[MCPManager] = None


def get_manager() -> MCPManager:
    global _global_manager
    if _global_manager is None:
        _global_manager = MCPManager(
            enabled=os.environ.get("MLX_MCP_ENABLE", "true").lower() == "true"
        )
    return _global_manager


def set_manager(manager: MCPManager) -> None:
    global _global_manager
    _global_manager = manager
