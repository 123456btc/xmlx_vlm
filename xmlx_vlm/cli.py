#!/usr/bin/env python
"""
Unified CLI for xmlx_vlm.

Usage:
    xmlx_vlm serve          Start the OpenAI-compatible server
    xmlx_vlm bench          Run performance benchmarks
    xmlx_vlm generate       Generate text from a prompt
    xmlx_vlm chat           Interactive chat in the terminal
    xmlx_vlm chat_ui        Launch Gradio chat UI (local model)
    xmlx_vlm chat_server    Launch Gradio chat UI (connect to server)
    xmlx_vlm convert        Convert / quantize a model
    xmlx_vlm video_generate Generate text from video
    xmlx_vlm agent          Launch the browser agent
    xmlx_vlm model list     List locally cached models
"""

import argparse
import importlib
import sys
from typing import List, Optional


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add args that are shared across many subcommands."""
    parser.add_argument("--model", type=str, help="Path or HF repo of the model")
    parser.add_argument("--adapter", type=str, default=None, help="LoRA adapter path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xmlx_vlm",
        description="Unified CLI for MLX Vision-Language Models",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # serve
    serve_p = subparsers.add_parser("serve", help="Start the OpenAI-compatible server")
    serve_p.add_argument("--model", type=str, default=None, help="Default model to load")
    serve_p.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    serve_p.add_argument("--port", type=int, default=5118, help="Server port")
    serve_p.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    serve_p.add_argument("--mcp-config", type=str, default=None, help="Path to MCP config file for external MCP servers")

    # bench
    bench_p = subparsers.add_parser("bench", help="Run performance benchmarks")
    _add_common_args(bench_p)
    bench_p.add_argument("--prompts", type=int, default=10, help="Number of prompts to run")
    bench_p.add_argument("--max-tokens", type=int, default=256, help="Max tokens per prompt")

    # generate
    gen_p = subparsers.add_parser("generate", help="Generate text from a prompt")
    _add_common_args(gen_p)
    gen_p.add_argument("--prompt", type=str, required=True, help="Input prompt")
    gen_p.add_argument("--max-tokens", type=int, default=8192, help="Max tokens to generate")
    gen_p.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    gen_p.add_argument("--image", type=str, default=None, help="Image file path")

    # chat
    chat_p = subparsers.add_parser("chat", help="Interactive chat in the terminal")
    _add_common_args(chat_p)
    chat_p.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")

    # chat_ui
    chat_ui_p = subparsers.add_parser("chat_ui", help="Launch Gradio chat UI (local model)")
    _add_common_args(chat_ui_p)

    # chat_server
    chat_server_p = subparsers.add_parser("chat_server", help="Launch Gradio chat UI (connect to server)")
    chat_server_p.add_argument("--server-url", type=str, default="http://localhost:5118", help="Server URL")
    chat_server_p.add_argument("--api-key", type=str, default=None, help="API key")
    chat_server_p.add_argument("--model", type=str, default=None, help="Model name")
    chat_server_p.add_argument("--port", type=int, default=7860, help="Gradio port")
    chat_server_p.add_argument("--share", action="store_true", help="Create public share link")
    chat_server_p.add_argument("--max-tokens", type=int, default=2048, help="Max tokens")
    chat_server_p.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    chat_server_p.add_argument("--enable-thinking", action="store_true", help="Enable thinking")
    chat_server_p.add_argument("--thinking-budget", type=int, default=None, help="Thinking budget")
    chat_server_p.add_argument("--text-only", action="store_true", help="Text-only mode")

    # convert
    conv_p = subparsers.add_parser("convert", help="Convert / quantize a model")
    conv_p.add_argument("--hf-path", type=str, required=True, help="HuggingFace model path")
    conv_p.add_argument("--mlx-path", type=str, required=True, help="Output MLX model path")
    conv_p.add_argument("--q-bits", type=int, default=4, help="Quantization bits")

    # video_generate
    vgen_p = subparsers.add_parser("video_generate", help="Generate text from video")
    _add_common_args(vgen_p)
    vgen_p.add_argument("--video", type=str, required=True, help="Video file path")
    vgen_p.add_argument("--prompt", type=str, required=True, help="Input prompt")

    # agent
    agent_p = subparsers.add_parser("agent", help="Launch the browser agent")
    agent_p.add_argument("--task", type=str, default=None, help="Task description")
    agent_p.add_argument("--url", type=str, default=None, help="Start URL")

    # doctor
    doctor_p = subparsers.add_parser("doctor", help="Run system health diagnostics")

    # mcp subcommand
    mcp_p = subparsers.add_parser("mcp", help="MCP server management")
    mcp_sub = mcp_p.add_subparsers(dest="mcp_command", help="MCP subcommands")

    mcp_install_p = mcp_sub.add_parser("install", help="Install MCP server into Codex/Cursor config")
    mcp_install_p.add_argument("--global", action="store_true", dest="global_install",
                               help="Install into ~/.codex/config.toml (default)")
    mcp_install_p.add_argument("--project", action="store_true",
                               help="Install into ./.codex/config.toml")
    mcp_install_p.add_argument("--base-url", type=str, default="http://127.0.0.1:5118/v1",
                               help="Server base URL")
    mcp_install_p.add_argument("--api-key", type=str, default="x123456",
                               help="API key for the server")
    mcp_install_p.add_argument("--model", type=str, default="mlx-community/diffusiongemma-26B-A4B-it-4bit",
                               help="Default model name")

    # model subcommand
    model_p = subparsers.add_parser("model", help="Model management")
    model_sub = model_p.add_subparsers(dest="model_command", help="Model subcommands")

    model_list_p = model_sub.add_parser("list", help="List locally cached models")
    model_list_p.add_argument("--limit", type=int, default=20, help="Max models to show")

    return parser


def _run_module_main(command: str, argv: List[str]) -> None:
    """Delegate to the existing module main() while preserving sys.argv."""
    module_map = {
        "serve": "xmlx_vlm.server",
        "bench": "xmlx_vlm.bench",
        "generate": "xmlx_vlm.generate",
        "chat": "xmlx_vlm.chat",
        "chat_ui": "xmlx_vlm.chat_ui",
        "chat_server": "xmlx_vlm.chat_server",
        "convert": "xmlx_vlm.convert",
        "video_generate": "xmlx_vlm.video_generate",
        "agent": "xmlx_vlm.agent",
    }
    mod_name = module_map.get(command)
    if not mod_name:
        raise ValueError(f"Unknown command: {command}")
    module = importlib.import_module(mod_name)
    # Most mains parse sys.argv themselves; ensure argv[0] is consistent
    old_argv = sys.argv
    sys.argv = [f"mlx_vlm {command}"] + argv
    try:
        module.main()
    finally:
        sys.argv = old_argv


def _mcp_install(args: argparse.Namespace) -> None:
    """Install xmlx-vlm MCP server into Claude Code / Cursor config."""
    base_url = args.base_url.rstrip("/")
    api_key = args.api_key
    model = args.model
    project = args.project
    global_install = getattr(args, "global_install", False) or not project

    if project:
        config_dir = Path(".codex")
        config_file = config_dir / "config.toml"
    else:
        config_dir = Path.home() / ".codex"
        config_file = config_dir / "config.toml"

    config_dir.mkdir(parents=True, exist_ok=True)

    # Read existing config if present (best-effort TOML parsing without extra deps)
    existing = ""
    if config_file.exists():
        existing = config_file.read_text(encoding="utf-8")

    server_block = f"""[mcp_servers.xmlx_vlm]
command = "python"
args = ["-m", "xmlx_vlm.mcp.server"]
startup_timeout_sec = 10
tool_timeout_sec = 120

[mcp_servers.xmlx_vlm.env]
XMLX_VLM_BASE_URL = "{base_url}"
XMLX_VLM_API_KEY = "{api_key}"
XMLX_VLM_MODEL = "{model}"
"""

    # Simple append strategy: if [mcp_servers.xmlx_vlm] already exists, replace it
    marker = "[mcp_servers.xmlx_vlm]"
    if marker in existing:
        start = existing.find(marker)
        end = existing.find("\n[", start + len(marker))
        if end == -1:
            end = len(existing)
        new_content = existing[:start] + server_block.rstrip() + "\n" + existing[end:]
    else:
        sep = "\n" if existing and not existing.endswith("\n") else ""
        new_content = existing + sep + server_block + "\n"

    config_file.write_text(new_content, encoding="utf-8")

    scope = "global (~/.codex/config.toml)" if global_install else "project (.codex/config.toml)"
    print(f"✅ xmlx-vlm MCP server installed ({scope})")
    print(f"   base_url: {base_url}")
    print(f"   model:    {model}")
    print(f"\nTo use it, tell Claude Code:")
    print('   Use xmlx_vlm for local vision-language inference.')


def _model_list(args: argparse.Namespace) -> None:
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        print("huggingface_hub is required for model list")
        sys.exit(1)

    cache = scan_cache_dir()
    repos = sorted(cache.repos, key=lambda r: r.repo_id or "")
    print(f"{'Repo ID':<50} {'Size':>12} {'Refs':>8}")
    print("-" * 72)
    total = 0
    for repo in repos[: args.limit]:
        size = repo.size_on_disk
        total += size
        size_str = f"{size / 1e9:.2f} GB" if size > 1e9 else f"{size / 1e6:.2f} MB"
        refs = ",".join(repo.refs.keys()) if hasattr(repo.refs, "keys") else ",".join(repo.refs) or "main"
        print(f"{repo.repo_id or 'unknown':<50} {size_str:>12} {refs:>8}")
    print("-" * 72)
    print(f"Total: {total / 1e9:.2f} GB  (showing {min(len(repos), args.limit)} of {len(repos)} repos)")


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "model":
        if args.model_command == "list":
            _model_list(args)
        else:
            model_p = [a for a in parser._subparsers._actions if isinstance(a, argparse._SubParsersAction)][0]
            model_p.choices["model"].print_help()
        return

    if args.command == "doctor":
        from .doctor import run_diagnostics
        ok = run_diagnostics()
        sys.exit(0 if ok else 1)

    if args.command == "mcp":
        if args.mcp_command == "install":
            _mcp_install(args)
        else:
            mcp_p = [a for a in parser._subparsers._actions if isinstance(a, argparse._SubParsersAction)][0]
            mcp_p.choices["mcp"].print_help()
        return

    # For all other commands, delegate to the existing module main() functions.
    # We reconstruct the raw argv so that existing argparsers in each module still work.
    raw = sys.argv[2:] if argv is None else argv
    _run_module_main(args.command, raw)


if __name__ == "__main__":
    main()
