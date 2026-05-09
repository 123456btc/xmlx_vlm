# SPDX-License-Identifier: Apache-2.0
"""Tests for parallel MCP tool execution."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from xmlx_vlm.mcp.executor import ToolExecutor
from xmlx_vlm.mcp.manager import MCPManager
from xmlx_vlm.mcp.types import MCPToolResult


class TestToolExecutorParallel(unittest.TestCase):
    def test_handle_calls_async_runs_in_parallel(self):
        """Built-in tools execute in parallel via asyncio.to_thread."""
        executor = ToolExecutor()
        call_order = []
        call_times = {}

        def slow_tool(delay: float):
            import time

            t0 = time.perf_counter()
            time.sleep(delay)
            call_order.append("slow")
            call_times["slow"] = time.perf_counter() - t0
            return f"slept {delay}"

        def fast_tool():
            import time

            t0 = time.perf_counter()
            time.sleep(0.05)
            call_order.append("fast")
            call_times["fast"] = time.perf_counter() - t0
            return "fast result"

        executor.register_tool("slow_tool", slow_tool)
        executor.register_tool("fast_tool", fast_tool)

        calls = [
            {"id": "1", "function": {"name": "slow_tool", "arguments": {"delay": 0.2}}},
            {"id": "2", "function": {"name": "fast_tool", "arguments": {}}},
        ]

        async def _run():
            return await executor.handle_calls_async(calls)

        results = asyncio.run(_run())

        # Both should complete; parallel execution means total time < 0.25s
        self.assertEqual(len(results), 2)
        # fast_tool should finish first even though it was called second
        self.assertEqual(call_order[0], "fast")


class TestMCPManagerParallel(unittest.IsolatedAsyncioTestCase):
    async def test_execute_runs_builtin_and_external_in_parallel(self):
        """Manager runs built-in and external tool calls concurrently."""
        manager = MCPManager(enabled=True)

        # Register a built-in tool
        manager.executor.register_tool("builtin_echo", lambda x: f"echo {x}")

        # Mock an external client
        mock_client = MagicMock()
        mock_client.call_tool = AsyncMock(
            return_value=MCPToolResult(
                tool_name="ext_tool", content="ext_result", is_error=False
            )
        )
        manager._external_clients["test_server"] = mock_client
        manager._external_schemas = [
            {
                "type": "function",
                "function": {"name": "test_server__ext_tool"},
            }
        ]

        calls = [
            {"id": "b1", "function": {"name": "builtin_echo", "arguments": {"x": "hello"}}},
            {"id": "e1", "function": {"name": "test_server__ext_tool", "arguments": {}}},
        ]

        results = await manager.execute(calls)

        self.assertEqual(len(results), 2)
        # External client should have been called
        mock_client.call_tool.assert_awaited_once_with("ext_tool", {})

    async def test_execute_parallel_external_calls(self):
        """Multiple external calls to the same server run in parallel."""
        manager = MCPManager(enabled=True)

        call_order = []

        async def slow_call(tool_name, arguments):
            await asyncio.sleep(0.1)
            call_order.append(tool_name)
            return MCPToolResult(tool_name=tool_name, content="ok")

        mock_client = MagicMock()
        mock_client.call_tool = slow_call
        manager._external_clients["srv"] = mock_client
        manager._external_schemas = [
            {"type": "function", "function": {"name": "srv__t1"}},
            {"type": "function", "function": {"name": "srv__t2"}},
        ]

        calls = [
            {"id": "1", "function": {"name": "srv__t1", "arguments": {}}},
            {"id": "2", "function": {"name": "srv__t2", "arguments": {}}},
        ]

        t0 = asyncio.get_event_loop().time()
        results = await manager.execute(calls)
        elapsed = asyncio.get_event_loop().time() - t0

        self.assertEqual(len(results), 2)
        # Parallel execution should finish in < 0.18s (two 0.1s sleeps in parallel)
        self.assertLess(elapsed, 0.18)


if __name__ == "__main__":
    unittest.main()
