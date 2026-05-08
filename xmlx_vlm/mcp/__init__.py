# SPDX-License-Identifier: Apache-2.0
"""
Model Context Protocol (MCP) integration for mlx_vlm.

Provides filesystem, shell, and Git tools to the model through a unified
executor and security policy. Also supports external MCP servers via stdio
and SSE transports.

Usage in server.py:
    from .mcp import get_manager
    mcp = get_manager()
    # Inject mcp.schemas into the chat request tools
    # Execute tool calls via mcp.execute(calls)
    # Connect external servers via await mcp.connect_external_servers(path)
"""

from .client import MCPClient
from .config import create_example_config, load_mcp_config
from .executor import ToolExecutor
from .manager import MCPManager, get_manager, set_manager
from .security import MCPSecurityError, SecurityPolicy, default_policy
from .tools import BUILTIN_TOOLS, TOOL_SCHEMAS
from .types import (
    MCPConfig,
    MCPServerConfig,
    MCPServerState,
    MCPServerStatus,
    MCPTool,
    MCPToolResult,
    MCPTransport,
)

__all__ = [
    "BUILTIN_TOOLS",
    "MCPClient",
    "MCPConfig",
    "MCPSecurityError",
    "MCPManager",
    "MCPServerConfig",
    "MCPServerState",
    "MCPServerStatus",
    "MCPTool",
    "MCPToolResult",
    "MCPTransport",
    "SecurityPolicy",
    "TOOL_SCHEMAS",
    "ToolExecutor",
    "create_example_config",
    "default_policy",
    "get_manager",
    "load_mcp_config",
    "set_manager",
]
