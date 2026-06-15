"""工具注册表 —— 统一管理 AI Trader 可调用的工具，包含内置工具和 MCP 服务."""

from __future__ import annotations

import json
import logging
import asyncio
from typing import Any, Dict, List
from pathlib import Path

from xmlx_vlm.ai_trader.tools.market import MarketDataTool
# ChartTool removed: model uses structured kline+L2 feature data, not visual charts.
# Re-add if visual chart analysis is needed in future.
from xmlx_vlm.ai_trader.tools.trading import TradingTool
from xmlx_vlm.ai_trader.tools.web_search import WebSearchTool, WebExtractTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册表：注册、查询、执行工具."""

    def __init__(
        self,
        live: bool = False,
        exchange: str = "paper",
        risk_profile: str = "conservative",
        dry_run: bool = False,
    ):
        self._tools: Dict[str, Any] = {}
        self.register(MarketDataTool())
        # ChartTool deliberately excluded: model uses structured feature data.
        self.register(WebSearchTool())
        self.register(WebExtractTool())
        self.register(
            TradingTool(
                oms=None,  # 懒加载
            )
        )
        self.mcp_manager = None
        self.loop = None

    def register(self, tool: Any):
        """注册一个工具实例."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Any:
        return self._tools.get(name)

    async def connect_mcp_servers(self):
        """Connect to MCP servers configured in ~/.hermes/config.yaml."""
        import yaml
        from xmlx_vlm.mcp.manager import MCPManager
        from xmlx_vlm.mcp.config import validate_config

        # 1. Load hermes config
        config_path = Path("~/.hermes/config.yaml").expanduser()
        if not config_path.exists():
            logger.info("No hermes config found at ~/.hermes/config.yaml, skipping MCP setup.")
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load hermes config: {e}")
            return

        mcp_servers = data.get("mcp_servers", {})
        if not mcp_servers:
            logger.info("No mcp_servers found in ~/.hermes/config.yaml.")
            return

        # Transform to validate_config structure expected by xmlx_vlm.mcp
        formatted_data = {
            "servers": mcp_servers,
            "max_tool_calls": data.get("max_tool_calls", 10),
            "default_timeout": data.get("default_timeout", 120.0),
            "allowed_high_risk_tools": data.get("allowed_high_risk_tools", [])
        }

        try:
            mcp_config = validate_config(formatted_data)
        except Exception as e:
            logger.warning(f"Failed to validate MCP config: {e}")
            return

        # 2. Initialize MCPManager and connect
        self.mcp_manager = MCPManager(enabled=True)
        self.mcp_manager._external_config = mcp_config
        self.loop = asyncio.get_running_loop()

        for sname, server_config in mcp_config.servers.items():
            if not server_config.enabled:
                continue
            from xmlx_vlm.mcp.client import MCPClient
            client = MCPClient(server_config)
            logger.info(f"Connecting to MCP server '{sname}'...")
            success = await client.connect()
            if success:
                self.mcp_manager._external_clients[sname] = client
                for tool in client.tools:
                    self.mcp_manager._external_schemas.append(tool.to_openai_format())
                logger.info(f"Successfully connected to MCP server '{sname}' with {len(client.tools)} tools.")
            else:
                logger.warning(f"Failed to connect to MCP server '{sname}'")

    async def disconnect_mcp_servers(self):
        """Disconnect from all MCP servers."""
        if self.mcp_manager:
            logger.info("Disconnecting from all external MCP servers...")
            await self.mcp_manager.disconnect_all()

    def list_tools(self) -> List[Dict[str, Any]]:
        """返回 OpenAI function calling 格式的工具列表."""
        tools = []
        for tool in self._tools.values():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
            )
        # Append MCP tools dynamically
        if self.mcp_manager and self.mcp_manager.enabled:
            for schema in self.mcp_manager._external_schemas:
                tools.append(schema)
        return tools

    def execute(self, name: str, arguments: Any) -> str:
        """执行指定工具."""
        # 1. Route to MCP server if it's an MCP tool
        if "__" in name and self.mcp_manager:
            server_name, actual_tool = name.split("__", 1)
            client = self.mcp_manager._external_clients.get(server_name)
            if client is None:
                return f"错误：未连接 MCP 服务 {server_name}"

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return f"错误：参数不是合法 JSON"

            coro = client.call_tool(actual_tool, arguments)
            if self.loop and self.loop.is_running():
                future = asyncio.run_coroutine_threadsafe(coro, self.loop)
                try:
                    result = future.result(timeout=120)
                    if result.is_error:
                        return f"Error: {result.error_message}"
                    if isinstance(result.content, str):
                        return result.content
                    return json.dumps(result.content, ensure_ascii=False, indent=2)
                except Exception as e:
                    return f"Error executing MCP tool: {e}"
            else:
                try:
                    result = asyncio.run(coro)
                    if result.is_error:
                        return f"Error: {result.error_message}"
                    if isinstance(result.content, str):
                        return result.content
                    return json.dumps(result.content, ensure_ascii=False, indent=2)
                except Exception as e:
                    return f"Error executing MCP tool: {e}"

        # 2. Otherwise route to built-in tools
        tool = self._tools.get(name)
        if tool is None:
            return f"错误：未找到工具 {name}"

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"错误：工具 {name} 的参数不是合法 JSON"
        if not isinstance(arguments, dict):
            return f"错误：工具 {name} 的参数必须是对象"

        logger.info("Executing tool %s with args %s", name, arguments)
        return tool.run(**arguments)

    def execute_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量执行工具调用，返回结果列表."""
        results = []
        for call in tool_calls:
            name = call.get("name") or call.get("function", {}).get("name")
            args = call.get("arguments") or call.get("function", {}).get("arguments", "{}")
            output = self.execute(name, args)
            results.append(
                {
                    "tool_call_id": call.get("id", ""),
                    "name": name,
                    "output": output,
                }
            )
        return results
