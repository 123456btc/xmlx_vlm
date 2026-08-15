#!/usr/bin/env python3
"""
XMLX-VLM Dynamic Model Selector & Universal Hardware Compatibility Engine
Scans arbitrary user directories, HuggingFace Hub, custom model paths,
and matches them against the machine's real unified memory and Apple Silicon architecture.
Zero hardcoding: fully dynamic discovery, config parsing, and memory sizing.
"""

from __future__ import annotations

import os
import sys
import json
import glob
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Universal community model reference catalog (for auto-recommendations based on user RAM)
COMMUNITY_REFERENCE_CATALOG = [
    {
        "id": "mlx-community/Qwen3.8-27B-4bit",
        "name": "Qwen 3.8 27B (4-bit)",
        "params": 27,
        "ram_gb": 18,
        "desc": "最新 Qwen 3.8 旗舰视觉语言模型 (高阶推理 / 全模态 / 代码)",
        "min_system_ram": 24,
    },
    {
        "id": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "name": "Qwen 3.6 35B A3B (4-bit)",
        "params": 35,
        "ram_gb": 22,
        "desc": "35B 高性能 MoE 架构模型 (极速推理与复杂量化决策)",
        "min_system_ram": 32,
    },
    {
        "id": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        "name": "Qwen 2.5 VL 7B (4-bit)",
        "params": 7,
        "ram_gb": 6,
        "desc": "7B 极速视觉多模态语言模型 (图表分析 / 轻量高效)",
        "min_system_ram": 8,
    },
    {
        "id": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
        "name": "Qwen 2.5 Coder 32B (4-bit)",
        "params": 32,
        "ram_gb": 20,
        "desc": "32B 编程与量化代码生成专家",
        "min_system_ram": 24,
    },
    {
        "id": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
        "name": "DeepSeek R1 Distill Qwen 32B (4-bit)",
        "params": 32,
        "ram_gb": 20,
        "desc": "32B 深度思维链推理模型 (DeepSeek R1 蒸馏版)",
        "min_system_ram": 24,
    },
    {
        "id": "mlx-community/Qwen2.5-72B-Instruct-4bit",
        "name": "Qwen 2.5 72B (4-bit)",
        "params": 72,
        "ram_gb": 44,
        "desc": "72B 超大容量旗舰语言模型 (大内存 Mac 优选)",
        "min_system_ram": 64,
    },
    {
        "id": "mlx-community/Llama-3.3-70B-Instruct-4bit",
        "name": "Llama 3.3 70B (4-bit)",
        "params": 70,
        "ram_gb": 42,
        "desc": "Meta Llama 3.3 70B 旗舰指令模型",
        "min_system_ram": 64,
    },
    {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "name": "TinyLlama 1.1B Chat",
        "params": 1.1,
        "ram_gb": 2,
        "desc": "1.1B 极速冒烟测试与超轻量调试模型",
        "min_system_ram": 4,
    },
]

# Patterns of auxiliary non-chat models to exclude from primary list
IGNORED_SUBSTRINGS = ("timesfm", "dflash", "assistant", "clip-vit", "whisper", "bert")


def get_hardware_info() -> Dict[str, Any]:
    """Dynamically probe host system memory and CPU architecture."""
    ram_gb = 16.0
    chip_name = "Apple Silicon"
    os_name = sys.platform

    if os_name == "darwin":
        # macOS sysctl
        try:
            res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
            ram_bytes = int(res.stdout.strip())
            ram_gb = round(ram_bytes / (1024 ** 3), 1)
        except Exception:
            pass

        try:
            res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True)
            brand = res.stdout.strip()
            if brand:
                chip_name = brand
        except Exception:
            pass
    elif os_name.startswith("linux"):
        # Linux /proc/meminfo
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        ram_gb = round(kb / (1024 ** 2), 1)
                        break
            res = subprocess.run(["uname", "-m"], capture_output=True, text=True)
            chip_name = f"Linux ({res.stdout.strip()})"
        except Exception:
            pass

    return {
        "chip": chip_name,
        "ram_gb": ram_gb,
        "os": os_name,
    }


def parse_model_directory(dir_path: Path, repo_id_hint: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Inspect an on-disk folder, parse config.json, determine quantization, and calculate disk size."""
    if not dir_path.is_dir():
        return None

    config_file = dir_path / "config.json"
    cfg = {}
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            pass

    # Determine Model Architecture & Type
    arch_list = cfg.get("architectures", [])
    arch = arch_list[0] if arch_list else cfg.get("model_type", "Unknown")

    # Filter out pure drafters / auxiliary models
    check_name = (repo_id_hint or dir_path.name).lower()
    if any(ign in check_name for ign in IGNORED_SUBSTRINGS) or "dflash" in arch.lower():
        return None

    # Calculate actual disk size
    try:
        files = [f for f in dir_path.glob("*") if f.is_file()]
        # Follow symlinks if any
        size_bytes = sum(f.stat().st_size for f in files)
        # Check subfolders (like safetensors shards)
        if size_bytes == 0:
            size_bytes = sum(f.stat().st_size for f in dir_path.glob("**/*") if f.is_file())
        size_gb = round(size_bytes / (1024 ** 3), 1)
    except Exception:
        size_gb = 0.0

    # Determine Quantization
    quant_cfg = cfg.get("quantization") or cfg.get("quantization_config") or {}
    quant_bits = None
    if isinstance(quant_cfg, dict):
        quant_bits = quant_cfg.get("bits") or quant_cfg.get("group_size")
    if not quant_bits:
        # Infer from name
        if "4bit" in check_name or "4-bit" in check_name or "q4" in check_name:
            quant_bits = 4
        elif "8bit" in check_name or "8-bit" in check_name or "q8" in check_name:
            quant_bits = 8

    quant_str = f"{quant_bits}-bit" if quant_bits else "fp16/bf16"

    # Compute estimated RAM required (Weights + KV Cache + Run headroom)
    if size_gb > 0:
        needed_ram = round(size_gb * 1.15 + 2.0, 1)
    else:
        # Fallback based on name or defaults
        needed_ram = 16.0

    model_id = repo_id_hint or dir_path.name

    return {
        "id": model_id,
        "local_path": str(dir_path.resolve()),
        "disk_size_gb": size_gb,
        "needed_ram_gb": needed_ram,
        "quant": quant_str,
        "arch": arch,
        "cached": True,
    }


def discover_all_local_models() -> List[Dict[str, Any]]:
    """
    Universally discover all downloaded models across all possible user locations:
    - $HF_HOME, $HUGGINGFACE_HUB_CACHE, $TRANSFORMERS_CACHE
    - $XMLX_VLM_MODELS_DIR, $MODEL_DIR, $MODELS_DIR
    - ~/.cache/huggingface/hub/
    - ./models, ../models, ~/models, ~/.models
    - ~/.lmstudio/models, ~/.cache/mlx.models, ~/.ollama/models
    """
    search_dirs: List[Path] = []

    # 1. Environment variables
    env_vars = [
        "XMLX_VLM_MODELS_DIR",
        "MODEL_DIR",
        "MODELS_DIR",
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
    ]
    for ev in env_vars:
        val = os.getenv(ev)
        if val:
            p = Path(val).expanduser().resolve()
            if p.exists():
                search_dirs.append(p)
                if (p / "hub").exists():
                    search_dirs.append(p / "hub")

    # 2. Standard user directories
    standard_paths = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.cwd() / "models",
        Path.cwd().parent / "models",
        Path.home() / "models",
        Path.home() / ".models",
        Path.home() / ".lmstudio" / "models",
        Path.home() / ".cache" / "mlx.models",
    ]
    for sp in standard_paths:
        if sp.exists() and sp not in search_dirs:
            search_dirs.append(sp)

    discovered_map: Dict[str, Dict[str, Any]] = {}

    for base_dir in search_dirs:
        if not base_dir.exists():
            continue

        # Strategy A: Hugging Face Hub structure (`models--org--name/snapshots/<hash>/`)
        for model_folder in base_dir.glob("models--*"):
            parts = model_folder.name.replace("models--", "").split("--")
            repo_id = f"{parts[0]}/{'--'.join(parts[1:])}" if len(parts) >= 2 else parts[0]
            
            snapshots_dir = model_folder / "snapshots"
            if snapshots_dir.exists():
                # Pick the snapshot with the largest size / latest files
                best_snapshot = None
                max_size = -1
                for snap in snapshots_dir.iterdir():
                    if snap.is_dir():
                        parsed = parse_model_directory(snap, repo_id_hint=repo_id)
                        if parsed and parsed["disk_size_gb"] > max_size:
                            max_size = parsed["disk_size_gb"]
                            best_snapshot = parsed

                if best_snapshot:
                    discovered_map[repo_id] = best_snapshot

        # Strategy B: Direct local subdirectories containing config.json or weights
        for sub in base_dir.iterdir():
            if sub.is_dir() and not sub.name.startswith("."):
                # Avoid re-parsing HF Hub models-- folders
                if sub.name.startswith("models--"):
                    continue

                if (sub / "config.json").exists():
                    parsed = parse_model_directory(sub, repo_id_hint=sub.name)
                    if parsed:
                        discovered_map[sub.name] = parsed
                else:
                    # Check 1 level deeper (e.g. models/org/model_name)
                    for nested in sub.iterdir():
                        if nested.is_dir() and (nested / "config.json").exists():
                            repo_name = f"{sub.name}/{nested.name}"
                            parsed = parse_model_directory(nested, repo_id_hint=repo_name)
                            if parsed:
                                discovered_map[repo_name] = parsed

    # Return sorted by disk size (descending)
    return sorted(discovered_map.values(), key=lambda x: x["disk_size_gb"], reverse=True)


def evaluate_memory_compatibility(needed_ram_gb: float, system_ram_gb: float) -> Tuple[bool, str, str]:
    """
    Dynamically evaluate whether a model fits in system RAM and return status tag & badge color.
    """
    if needed_ram_gb <= system_ram_gb * 0.70:
        return True, "极佳 (流畅运行)", "green"
    elif needed_ram_gb <= system_ram_gb * 0.90:
        return True, "良好 (内存充裕)", "green"
    elif needed_ram_gb <= system_ram_gb:
        return True, "可运行 (接近上限)", "yellow"
    else:
        return False, f"内存不足 (需 ~{int(needed_ram_gb)}G / 本机 {int(system_ram_gb)}G)", "red"


def get_full_model_menu() -> Dict[str, Any]:
    """Combine local discovered models with tailored community recommendations."""
    hw = get_hardware_info()
    sys_ram = hw["ram_gb"]

    local_models = discover_all_local_models()
    local_ids = {m["id"] for m in local_models}

    # Format local models
    local_list = []
    for lm in local_models:
        fits, status_text, color = evaluate_memory_compatibility(lm["needed_ram_gb"], sys_ram)
        lm["fits_memory"] = fits
        lm["status_text"] = status_text
        lm["color"] = color
        local_list.append(lm)

    # Filter community recommendations (only show models not already cached locally)
    community_list = []
    for ref in COMMUNITY_REFERENCE_CATALOG:
        ref_id = ref["id"]
        if ref_id in local_ids:
            continue

        needed_ram = float(ref["ram_gb"])
        fits, status_text, color = evaluate_memory_compatibility(needed_ram, sys_ram)

        community_list.append({
            "id": ref_id,
            "name": ref["name"],
            "needed_ram_gb": needed_ram,
            "desc": ref["desc"],
            "fits_memory": fits,
            "status_text": status_text,
            "color": color,
            "cached": False,
        })

    # Sort community recommendations so models that fit this user's RAM appear first
    community_list.sort(key=lambda x: (not x["fits_memory"], x["needed_ram_gb"]))

    return {
        "hardware": hw,
        "local_models": local_list,
        "community_models": community_list,
    }


def render_interactive_menu(default_model: Optional[str] = None) -> str:
    """Display dynamic, fully responsive interactive CLI menu."""
    menu_data = get_full_model_menu()
    hw = menu_data["hardware"]
    local_models = menu_data["local_models"]
    community_models = menu_data["community_models"]

    # ANSI Colors
    C_RESET = "\033[0m"
    C_BOLD = "\033[1m"
    C_CYAN = "\033[36m"
    C_GREEN = "\033[32m"
    C_YELLOW = "\033[33m"
    C_RED = "\033[31m"
    C_GRAY = "\033[90m"
    C_MAGENTA = "\033[35m"

    color_map = {
        "green": C_GREEN,
        "yellow": C_YELLOW,
        "red": C_RED,
    }

    print(f"\n{C_BOLD}{C_CYAN}================================================================================{C_RESET}")
    print(f"{C_BOLD}  🚀 XMLX-VLM 动态模型选择器{C_RESET} {C_GRAY}(设备: {C_GREEN}{hw['chip']}{C_GRAY} | 统一内存: {C_GREEN}{hw['ram_gb']} GB{C_GRAY}){C_RESET}")
    print(f"{C_BOLD}{C_CYAN}================================================================================{C_RESET}")

    choices_map: Dict[int, Dict[str, Any]] = {}
    idx = 1

    # Section 1: Locally Discovered Models
    print(f"\n{C_BOLD}{C_GREEN}📦 本机已发现并可运行的模型 ({len(local_models)} 个已就绪，无需下载):{C_RESET}")
    if not local_models:
        print(f"  {C_GRAY}(未在本地缓存目录中找到已下载的模型){C_RESET}")
    else:
        for m in local_models:
            choices_map[idx] = m
            size_str = f" [{m['disk_size_gb']}G | {m['quant']}]"
            c_code = color_map.get(m["color"], C_GREEN)
            compat = f" - {c_code}{m['status_text']}{C_RESET}"
            print(f"  {C_BOLD}{C_CYAN}[{idx}]{C_RESET} {C_BOLD}{m['id']}{C_RESET}{C_GRAY}{size_str}{C_RESET}{compat}")
            idx += 1

    # Section 2: Tailored Community Recommendations for this machine
    print(f"\n{C_BOLD}{C_YELLOW}🌐 社区精选模型 (适合本机内存，初次选择将自动极速下载):{C_RESET}")
    for m in community_models:
        choices_map[idx] = m
        c_code = color_map.get(m["color"], C_YELLOW)
        compat = f" [{c_code}{m['status_text']}{C_RESET}]"
        desc = f" - {C_GRAY}{m['desc']}{C_RESET}"
        print(f"  {C_BOLD}{C_YELLOW}[{idx}]{C_RESET} {m['id']}{compat}{desc}")
        idx += 1

    print(f"\n  {C_BOLD}[0]{C_RESET} {C_MAGENTA}输入自定义 HuggingFace Repo ID 或本地磁盘绝对路径...{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}================================================================================{C_RESET}")

    # Determine default recommendation
    default_idx = 1
    if default_model:
        for i, m in choices_map.items():
            if m["id"] == default_model:
                default_idx = i
                break
    else:
        # Default to first local model if available, else first compatible community model
        if local_models:
            default_idx = 1
        else:
            for i, m in choices_map.items():
                if m.get("fits_memory", False):
                    default_idx = i
                    break

    default_name = choices_map.get(default_idx, {}).get("id", "mlx-community/Qwen3.8-27B-4bit")

    # If non-interactive stdin, return default
    if not sys.stdin.isatty():
        return default_name

    try:
        sys.stdout.write(f"\n{C_BOLD}请选择要运行的模型编号 [默认: {default_idx} ({default_name})]: {C_RESET}")
        sys.stdout.flush()
        user_input = sys.stdin.readline().strip()

        if not user_input:
            return default_name

        if user_input == "0":
            sys.stdout.write(f"{C_BOLD}请输入自定义模型名称或本地路径: {C_RESET}")
            sys.stdout.flush()
            custom_model = sys.stdin.readline().strip()
            return custom_model if custom_model else default_name

        try:
            choice_num = int(user_input)
            if choice_num in choices_map:
                return choices_map[choice_num]["id"]
            else:
                print(f"{C_YELLOW}无效编号，使用默认模型: {default_name}{C_RESET}")
                return default_name
        except ValueError:
            # If user directly typed a model name string
            return user_input

    except (KeyboardInterrupt, EOFError):
        print("\n[!] 取消选择，退出。")
        sys.exit(130)


def main():
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "--list-json":
            print(json.dumps(get_full_model_menu(), ensure_ascii=False, indent=2))
            return
        elif cmd == "--interactive":
            default_mod = sys.argv[2] if len(sys.argv) > 2 else None
            chosen = render_interactive_menu(default_model=default_mod)
            print(chosen)
            return

    # Default: interactive select and print model name to stdout
    chosen = render_interactive_menu()
    print(chosen)


if __name__ == "__main__":
    main()
