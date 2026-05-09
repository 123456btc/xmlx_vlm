#!/usr/bin/env python
"""
System health diagnostic for xmlx_vlm.

Inspired by CodexSaver's `doctor` command — quick checks that tell the user
whether the stack is ready for inference.
"""

import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple


def _ok(msg: str) -> str:
    return f"  ✅ {msg}"


def _warn(msg: str) -> str:
    return f"  ⚠️  {msg}"


def _err(msg: str) -> str:
    return f"  ❌ {msg}"


def check_platform() -> Tuple[bool, List[str]]:
    """Check OS and hardware."""
    lines: List[str] = []
    ok = True
    system = platform.system()
    machine = platform.machine()
    lines.append(f"Platform: {system} {machine}")

    if system != "Darwin":
        lines.append(_warn("macOS recommended for MLX GPU acceleration"))
    else:
        lines.append(_ok("macOS detected"))

    if machine not in ("arm64", "aarch64"):
        lines.append(_warn("Apple Silicon (arm64) recommended for best performance"))
    else:
        lines.append(_ok("Apple Silicon (arm64) detected"))

    return ok, lines


def check_python() -> Tuple[bool, List[str]]:
    """Check Python version."""
    lines: List[str] = []
    ok = True
    version = sys.version_info
    lines.append(f"Python: {sys.version.split()[0]}")
    if version < (3, 10):
        lines.append(_err("Python 3.10+ required"))
        ok = False
    else:
        lines.append(_ok(f"Python {version.major}.{version.minor}.{version.micro}"))
    return ok, lines


def check_mlx() -> Tuple[bool, List[str]]:
    """Check MLX runtime and GPU memory."""
    lines: List[str] = []
    ok = True
    try:
        import mlx.core as mx
        lines.append(_ok(f"MLX {mx.__version__}"))
    except ImportError as e:
        lines.append(_err(f"MLX not installed: {e}"))
        return False, lines

    try:
        device = mx.default_device()
        lines.append(_ok(f"Default device: {device}"))
    except Exception as e:
        lines.append(_warn(f"Could not detect default device: {e}"))

    try:
        mem = mx.get_active_memory() / 1e9
        peak = mx.get_peak_memory() / 1e9
        lines.append(_ok(f"GPU memory: active={mem:.2f}GB peak={peak:.2f}GB"))
    except Exception as e:
        lines.append(_warn(f"Could not query GPU memory: {e}"))

    return ok, lines


def check_models() -> Tuple[bool, List[str]]:
    """Check whether default / cached models exist."""
    lines: List[str] = []
    ok = True
    try:
        from huggingface_hub import scan_cache_dir
        cache = scan_cache_dir()
        total_gb = sum(r.size_on_disk for r in cache.repos) / 1e9
        lines.append(_ok(f"HF cache: {len(cache.repos)} repos, {total_gb:.1f}GB total"))
    except ImportError:
        lines.append(_warn("huggingface_hub not installed — model cache check skipped"))
    except Exception as e:
        lines.append(_warn(f"Could not scan model cache: {e}"))
    return ok, lines


def check_apc() -> Tuple[bool, List[str]]:
    """Check APC disk cache configuration."""
    lines: List[str] = []
    ok = True
    apc_enabled = os.environ.get("APC_ENABLED", "")
    apc_path = os.environ.get("APC_DISK_PATH", os.path.expanduser("~/.cache/xmlx_vlm/apc"))
    if apc_enabled in ("1", "true", "True", "yes"):
        lines.append(_ok(f"APC enabled (disk={apc_path})"))
    else:
        lines.append(_warn("APC not enabled (set APC_ENABLED=1 to enable prefix caching)"))
    return ok, lines


def check_ports() -> Tuple[bool, List[str]]:
    """Check whether the default server port is free."""
    lines: List[str] = []
    ok = True
    port = int(os.environ.get("XMLX_VLM_PORT", "8080"))
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(("127.0.0.1", port))
            if result == 0:
                lines.append(_warn(f"Port {port} is already in use (server may be running)"))
            else:
                lines.append(_ok(f"Port {port} is free"))
    except Exception as e:
        lines.append(_warn(f"Could not check port {port}: {e}"))
    return ok, lines


def check_service() -> Tuple[bool, List[str]]:
    """Check whether service.sh reports a running server."""
    lines: List[str] = []
    ok = True
    script = Path(__file__).with_name("service.sh")
    if not script.exists():
        lines.append(_warn("service.sh not found — service status check skipped"))
        return ok, lines
    try:
        result = subprocess.run(
            ["bash", str(script), "status"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and ("running" in result.stdout.lower() or "pid" in result.stdout.lower()):
            lines.append(_ok("Server appears to be running (service.sh status)"))
        else:
            lines.append(_warn("Server not running"))
    except Exception as e:
        lines.append(_warn(f"Could not run service.sh status: {e}"))
    return ok, lines


def run_diagnostics() -> bool:
    """Run all diagnostic checks and print a report."""
    print("=" * 60)
    print(" XMLX-VLM Doctor")
    print("=" * 60)
    overall = True
    checks = [
        ("Platform", check_platform),
        ("Python", check_python),
        ("MLX Runtime", check_mlx),
        ("Model Cache", check_models),
        ("APC Cache", check_apc),
        ("Network", check_ports),
        ("Service", check_service),
    ]
    for title, fn in checks:
        print(f"\n{title}:")
        ok, lines = fn()
        overall = overall and ok
        for line in lines:
            print(line)

    print("\n" + "=" * 60)
    if overall:
        print(" Result: Ready for inference 🚀")
    else:
        print(" Result: Some checks failed — review above ⚠️")
    print("=" * 60)
    return overall


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="XMLX-VLM system diagnostics")
    parser.parse_args()
    ok = run_diagnostics()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
