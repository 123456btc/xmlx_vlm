"""工具注册表 —— 统一管理 AI Trader 可调用的工具."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from xmlx_vlm.ai_trader.tools.market import MarketDataTool
from xmlx_vlm.ai_trader.tools.chart import ChartTool
from xmlx_vlm.ai_trader.tools.trading import TradingTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册表：注册、查询、执行工具."""

    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self.register(MarketDataTool())
        self.register(ChartTool())
        self.register(TradingTool())

    def register(self, tool: Any):
        """注册一个工具实例."""
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Any:
        return self._tools.get(name)

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
        return tools

    def execute(self, name: str, arguments: Any) -> str:
        """执行指定工具."""
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
