# SPDX-License-Identifier: Apache-2.0
"""
MCP tool executor.

Invokes built-in tools with argument validation, security checks,
audit logging, and tool-whitelist enforcement.
"""

import asyncio
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
        self._audit = None

    def _get_audit(self):
        if self._audit is None:
            from ..audit import get_audit_logger

            self._audit = get_audit_logger()
        return self._audit

    def register_tool(self, name: str, fn) -> None:
        self.tools[name] = fn

    def call(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call and return a string result."""
        audit = self._get_audit()

        # Tool whitelist check
        if not security.is_tool_allowed(name):
            msg = f"Error: tool '{name}' is not in the allowed tools whitelist"
            if audit:
                audit.log_security_event("blocked_tool", f"tool={name}")
            logger.warning(msg)
            return msg

        if name not in self.tools:
            return f"Error: unknown tool '{name}'"

        fn = self.tools[name]

        # Security checks for file operations
        if name in ("read_file", "list_dir", "search_files", "git_diff"):
            path = arguments.get("path", ".")
            if not self.policy.is_path_allowed(path):
                msg = f"Error: path not allowed by security policy: {path}"
                if audit:
                    audit.log_security_event("blocked_path", f"tool={name} path={path}")
                logger.warning(msg)
                return msg

        if name in ("write_file",):
            path = arguments.get("path", "")
            try:
                self.policy.check_write(path)
            except PermissionError as e:
                msg = f"Error: {e}"
                if audit:
                    audit.log_security_event("blocked_write", f"tool={name} path={path}")
                logger.warning(msg)
                return msg

        try:
            result = fn(**arguments)
        except Exception as e:
            logger.exception("Tool %s failed", name)
            result = f"Error executing {name}: {e}"

        # Audit log
        if audit:
            audit.log_tool_call(
                tool_name=name,
                arguments=arguments,
                result=result,
            )

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

    async def handle_calls_async(self, calls: list[dict]) -> list[dict]:
        """Process a batch of tool calls in parallel using thread pool."""
        if not calls:
            return []

        async def _run_one(call: dict) -> dict:
            name = call.get("name") or call.get("function", {}).get("name")
            args = call.get("arguments") or call.get("function", {}).get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            result = await asyncio.to_thread(self.call, name, args)
            return {
                "role": "tool",
                "tool_call_id": call.get("id", "unknown"),
                "name": name,
                "content": result,
            }

        return await asyncio.gather(*[_run_one(c) for c in calls])
