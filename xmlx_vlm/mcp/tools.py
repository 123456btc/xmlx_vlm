# SPDX-License-Identifier: Apache-2.0
"""
Built-in MCP tools for mlx_vlm.

Provides filesystem, shell, and Git operations scoped to the working directory.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List


def _cwd() -> Path:
    return Path(os.getcwd()).resolve()


def _resolve(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _cwd() / p
    return p.resolve()


# ---------------------------------------------------------------------------
# Filesystem tools
# ---------------------------------------------------------------------------

def read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    """Read text from a file."""
    p = _resolve(path)
    if not p.exists():
        return f"Error: file not found: {p}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"
    lines = text.splitlines()
    if offset:
        lines = lines[offset:]
    if limit and limit < len(lines):
        lines = lines[:limit]
    return "\n".join(lines)


def write_file(path: str, content: str) -> str:
    """Write text to a file (creates parent directories)."""
    p = _resolve(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} chars to {p}"
    except Exception as e:
        return f"Error writing file: {e}"


def list_dir(path: str = ".") -> str:
    """List directory contents."""
    p = _resolve(path)
    if not p.is_dir():
        return f"Error: not a directory: {p}"
    entries = []
    for e in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        marker = "📁" if e.is_dir() else "📄"
        entries.append(f"{marker} {e.name}")
    return "\n".join(entries) if entries else "(empty directory)"


def search_files(query: str, path: str = ".") -> str:
    """Recursively search file names matching a query string."""
    p = _resolve(path)
    matches = []
    for root, _, files in os.walk(p):
        for f in files:
            if query.lower() in f.lower():
                matches.append(os.path.join(root, f))
    if not matches:
        return "No matches found."
    return "\n".join(matches[:50])


# ---------------------------------------------------------------------------
# Shell tools
# ---------------------------------------------------------------------------

SAFE_SHELL_COMMANDS = {
    "ls", "cat", "head", "tail", "wc", "find", "grep", "rg", "pwd", "echo",
    "git", "python", "python3", "pip", "pytest", "make", "cargo", "npm", "yarn",
    "mkdir", "touch", "rm", "cp", "mv", "chmod", "chown", "du", "df",
    "swift", "swiftc", "xcodebuild",
}


def shell(command: str, timeout: int = 30) -> str:
    """Run a shell command with safety restrictions."""
    cmd = command.strip()
    if not cmd:
        return "Error: empty command"
    first_word = cmd.split()[0]
    if first_word not in SAFE_SHELL_COMMANDS:
        return f"Error: command '{first_word}' is not in the allow-list. Allowed: {', '.join(sorted(SAFE_SHELL_COMMANDS))}"
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_cwd()),
        )
        out = result.stdout
        if result.stderr:
            out += "\n[stderr]\n" + result.stderr
        if result.returncode != 0:
            out += f"\n[exit code {result.returncode}]"
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Git tools
# ---------------------------------------------------------------------------

def git_status() -> str:
    return shell("git status --short")


def git_diff(path: str = "") -> str:
    cmd = "git diff"
    if path:
        cmd += f" -- {path}"
    return shell(cmd)


def git_log(n: int = 5) -> str:
    return shell(f"git log --oneline -{n}")


def git_branch() -> str:
    return shell("git branch -a")


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

BUILTIN_TOOLS: Dict[str, Callable[..., Any]] = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "search_files": search_files,
    "shell": shell,
    "git_status": git_status,
    "git_diff": git_diff,
    "git_log": git_log,
    "git_branch": git_branch,
}


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative or absolute file path."},
                    "offset": {"type": "integer", "description": "Start line offset (0-based).", "default": 0},
                    "limit": {"type": "integer", "description": "Max lines to read (0 = unlimited).", "default": 0},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text to a file. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative or absolute file path."},
                    "content": {"type": "string", "description": "Text content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List the contents of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path.", "default": "."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Recursively search for files by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Substring to match in file names."},
                    "path": {"type": "string", "description": "Root directory to search.", "default": "."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command from an allow-list (ls, cat, git, python, make, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute."},
                    "timeout": {"type": "integer", "description": "Timeout in seconds.", "default": 30},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show git working tree status (short format).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show git diff for the working tree or a specific path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Specific file or directory.", "default": ""},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_log",
            "description": "Show recent git commits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "Number of commits.", "default": 5},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_branch",
            "description": "List local and remote branches.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
