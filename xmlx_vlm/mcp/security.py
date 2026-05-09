# SPDX-License-Identifier: Apache-2.0
"""
Security policies for MCP tool execution.

- Path sandboxing: restricts file operations to allowed roots
- Command filtering: only allow-listed shell commands
- Read-only mode: prevents write/delete operations
- MCP server command validation: whitelist + blocked patterns
- Finance mode: strict defaults for financial-sector deployments
- Tool whitelist: per-deployment allowed tools
- Config integrity: SHA256 verification of MCP config files
"""

import hashlib
import os
import re
from pathlib import Path
from typing import List, Optional, Set


_FINANCE_MODE = os.environ.get("XMLX_VLM_FINANCE_MODE", "0").lower() in ("1", "true", "yes")

# ─── Finance mode overrides ────────────────────────────────────────────────

if _FINANCE_MODE:
    # Force disable unsafe bypass
    os.environ["MLX_MCP_ALLOW_UNSAFE"] = "0"
    # Default to read-only unless explicitly overridden
    if "MLX_MCP_READONLY" not in os.environ:
        os.environ["MLX_MCP_READONLY"] = "true"


# ─── Tool whitelist ────────────────────────────────────────────────────────

_TOOL_WHITELIST_RAW = os.environ.get("XMLX_VLM_ALLOWED_TOOLS", "")
if _TOOL_WHITELIST_RAW and _TOOL_WHITELIST_RAW.strip():
    _TOOL_WHITELIST: Optional[Set[str]] = set(t.strip() for t in _TOOL_WHITELIST_RAW.split(",") if t.strip())
else:
    _TOOL_WHITELIST = None  # None means "all allowed" (legacy behavior)


# Finance mode default: only safe read-only tools
if _FINANCE_MODE and _TOOL_WHITELIST is None:
    _TOOL_WHITELIST = {"read_file", "list_dir", "search_files"}


def is_tool_allowed(name: str) -> bool:
    """Check if a tool is in the deployment whitelist."""
    if _TOOL_WHITELIST is None:
        return True
    return name in _TOOL_WHITELIST


def get_allowed_tools() -> Optional[Set[str]]:
    """Return the current tool whitelist, or None if all allowed."""
    return _TOOL_WHITELIST.copy() if _TOOL_WHITELIST else None


# ─── Config integrity ──────────────────────────────────────────────────────

_CONFIG_HASH_EXPECTED = os.environ.get("XMLX_VLM_MCP_CONFIG_HASH", "").strip()


def verify_config_integrity(config_path: str) -> bool:
    """Verify MCP config file SHA256 against expected hash (if set)."""
    if not _CONFIG_HASH_EXPECTED:
        return True
    try:
        with open(config_path, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        return actual.lower() == _CONFIG_HASH_EXPECTED.lower()
    except Exception:
        return False


# ─── SecurityPolicy ────────────────────────────────────────────────────────

class SecurityPolicy:
    """Defines what an MCP session is allowed to do."""

    def __init__(
        self,
        allowed_roots: Optional[List[str]] = None,
        read_only: bool = False,
        blocked_paths: Optional[List[str]] = None,
    ):
        self.read_only = read_only
        self.blocked_paths: Set[Path] = set()
        if blocked_paths:
            for p in blocked_paths:
                self.blocked_paths.add(Path(p).resolve())

        # Default allowed roots: current working dir + home
        roots = allowed_roots or [os.getcwd(), os.path.expanduser("~")]
        self.allowed_roots: List[Path] = [Path(r).resolve() for r in roots]

    def _normalize(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = Path(os.getcwd()) / p
        return p.resolve()

    def is_path_allowed(self, path: str) -> bool:
        """Check if a path is inside any allowed root and not blocked."""
        p = self._normalize(path)
        for blocked in self.blocked_paths:
            try:
                p.relative_to(blocked)
                return False
            except ValueError:
                pass
        for root in self.allowed_roots:
            try:
                p.relative_to(root)
                return True
            except ValueError:
                pass
        return False

    def check_write(self, path: str) -> None:
        if self.read_only:
            raise PermissionError("MCP is in read-only mode")
        if not self.is_path_allowed(path):
            raise PermissionError(f"Path not allowed: {path}")

    def check_read(self, path: str) -> None:
        if not self.is_path_allowed(path):
            raise PermissionError(f"Path not allowed: {path}")


class MCPSecurityError(Exception):
    """Raised when an MCP server configuration fails security validation."""

    pass


# Default command whitelist for MCP stdio servers
DEFAULT_COMMAND_WHITELIST = {
    # Node.js ecosystem
    "npx",
    "npm",
    "node",
    # Python ecosystem
    "uvx",
    "uv",
    "python",
    "python3",
    "pip",
    "pipx",
    # Docker
    "docker",
    # Official MCP server prefixes
    "mcp-server-filesystem",
    "mcp-server-github",
    "mcp-server-postgres",
    "mcp-server-brave-search",
    "mcp-server-sqlite",
    "mcp-server-puppeteer",
    "mcp-server-slack",
    "mcp-server-gitlab",
}

# Patterns that indicate command injection attempts
BLOCKED_ARG_PATTERNS = [
    re.compile(r"[;|&`$()]"),  # shell metacharacters
    re.compile(r"\.\./"),  # path traversal
    re.compile(r"\$\("),  # command substitution
    re.compile(r"`"),  # backtick substitution
]

# Dangerous environment variable keys
BLOCKED_ENV_KEYS = {
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "PATH",
    "PYTHONPATH",
    "DYLD_INSERT_LIBRARIES",
}


def validate_mcp_server_config(
    server_name: str,
    command: Optional[str] = None,
    args: Optional[List[str]] = None,
    env: Optional[dict] = None,
    url: Optional[str] = None,
    custom_whitelist: Optional[Set[str]] = None,
) -> None:
    """
    Validate an MCP server configuration for security.

    Args:
        server_name: Name of the server (for error messages)
        command: Command to execute (stdio transport)
        args: Command arguments (stdio transport)
        env: Environment variables (stdio transport)
        url: URL for SSE transport
        custom_whitelist: Additional commands to allow

    Raises:
        MCPSecurityError: If the configuration violates security policy
    """
    # Finance mode: never allow bypass
    if not _FINANCE_MODE and os.environ.get("MLX_MCP_ALLOW_UNSAFE", "0") == "1":
        return

    # SSE transport has fewer risks; only basic URL validation
    if url is not None:
        if not url.startswith(("http://", "https://")):
            raise MCPSecurityError(
                f"MCP server '{server_name}': SSE URL must use http:// or https://"
            )
        return

    # Stdio transport validation
    if not command:
        raise MCPSecurityError(
            f"MCP server '{server_name}': stdio transport requires a command"
        )

    # Check command whitelist
    whitelist = set(DEFAULT_COMMAND_WHITELIST)
    if custom_whitelist:
        whitelist.update(custom_whitelist)

    # Allow commands that start with whitelisted prefixes
    command_base = Path(command).name
    is_allowed = command_base in whitelist
    if not is_allowed:
        for prefix in ("mcp-server-", "mcp-"):
            if command_base.startswith(prefix):
                is_allowed = True
                break

    if not is_allowed:
        raise MCPSecurityError(
            f"MCP server '{server_name}': Command '{command_base}' is not in the "
            f"allowed commands whitelist. Allowed: {sorted(whitelist)}. "
            f"Set MLX_MCP_ALLOW_UNSAFE=1 to bypass for local development."
        )

    # Check arguments for injection patterns
    if args:
        for arg in args:
            for pattern in BLOCKED_ARG_PATTERNS:
                if pattern.search(arg):
                    raise MCPSecurityError(
                        f"MCP server '{server_name}': Argument '{arg}' contains "
                        f"blocked characters (shell metacharacters or path traversal)."
                    )

    # Check environment variables
    if env:
        for key in env.keys():
            if key.upper() in BLOCKED_ENV_KEYS:
                raise MCPSecurityError(
                    f"MCP server '{server_name}': Environment variable '{key}' is blocked."
                )


def default_policy() -> SecurityPolicy:
    """Return a sensible default policy for local coding."""
    blocked = [
        "/usr/bin",
        "/bin",
        "/sbin",
        "/usr/sbin",
        "/etc",
        "/var",
        os.path.expanduser(".ssh"),
        os.path.expanduser(".gnupg"),
    ]
    read_only = os.environ.get("MLX_MCP_READONLY", "false").lower() == "true"
    return SecurityPolicy(read_only=read_only, blocked_paths=blocked)
