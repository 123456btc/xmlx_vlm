# SPDX-License-Identifier: Apache-2.0
"""
MCP tool executor.

Invokes built-in tools with argument validation and security checks.
"""

import json
import logging
from typing import Any, Dict

from . import security
from .tools import BUILTIN_TOOLS

logger = logging.getLogger("xmlx_vlm.mcp.executor")


class ToolExecutor:
    """Executes MCP tool calls safely."""

    def __init__(self, policy: security.SecurityPolicy | None = None):
        self.policy = policy or security.default_policy()
        self.tools = dict(BUILTIN_TOOLS)

    def register_tool(self, name: str, fn) -> None:
        self.tools[name] = fn

    def call(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call and return a string result."""
        if name not in self.tools:
            return f"Error: unknown tool '{name}'"

        fn = self.tools[name]

        # Security checks for file operations
        if name in ("read_file", "list_dir", "search_files", "git_diff"):
            path = arguments.get("path", ".")
            if not self.policy.is_path_allowed(path):
                return f"Error: path not allowed by security policy: {path}"

        if name in ("write_file",):
            path = arguments.get("path", "")
            try:
                self.policy.check_write(path)
            except PermissionError as e:
                return f"Error: {e}"

        if name == "shell":
            # Additional guard: shell tool already filters commands,
            # but we double-check here for policy overrides if we add them later.
            pass

        try:
            result = fn(**arguments)
        except Exception as e:
            logger.exception("Tool %s failed", name)
            return f"Error executing {name}: {e}"

        # Coerce result to string
        if not isinstance(result, str):
            try:
                result = json.dumps(result, ensure_ascii=False, indent=2)
            except Exception:
                result = str(result)
        return result

    def handle_calls(self, calls: list[dict]) -> list[dict]:
        """Process a batch of tool calls and return tool messages."""
        outputs = []
        for call in calls:
            name = call.get("name") or call.get("function", {}).get("name")
            args = call.get("arguments") or call.get("function", {}).get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result = self.call(name, args)
            outputs.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", "unknown"),
                    "name": name,
                    "content": result,
                }
            )
        return outputs
