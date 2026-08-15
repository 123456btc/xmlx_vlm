#!/usr/bin/env python3
"""
XMLX-VLM Model Selector & Hardware Matcher
Discovers locally cached models and recommended MLX community models,
evaluating hardware compatibility based on Apple Silicon unified memory.
"""

from __future__ import annotations

import os
import sys
import json
import glob
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

# Curated catalog of tested high-performance MLX models
CURATED_MODELS = [
    {
        "id": "mlx-community/Qwen3.8-27B-4bit",
        "name": "Qwen 3.8 27B (4-bit)",
        "ram_required_gb": 18,
        "desc": "🌟 强烈推荐: 最新 Qwen 3.8 旗舰视觉语言模型，高阶推理/全模态/代码",
        "tags": ["VLM", "Thinking", "Agent", "Recommended"],
    },
    {
        "id": "mlx-community/Qwen3.6-35B-A3B-4bit",
        "name": "Qwen 3.6 35B A3B (4-bit)",
        "ram_required_gb": 22,
        "desc": "35B 高性能 MoE 架构模型，极速推理与复杂量化决策",
        "tags": ["MoE", "Quant", "Fast"],
    },
    {
        "id": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
        "name": "Diffusion Gemma 26B A4B (4-bit)",
        "ram_required_gb": 16,
        "desc": "Gemma 架构深度微调量化模型，通用对话与指令遵循",
        "tags": ["Gemma", "Conversational"],
    },
    {
        "id": "mlx-community/gemma-4-26b-a4b-it-4bit",
        "name": "Gemma 4 26B A4B (4-bit)",
        "ram_required_gb": 16,
        "desc": "Gemma 4 架构 26B 4-bit 量化模型",
        "tags": ["Gemma4", "Instruct"],
    },
    {
        "id": "mlx-community/gemma-4-31b-it-4bit",
        "name": "Gemma 4 31B (4-bit)",
        "ram_required_gb": 20,
        "desc": "Gemma 4 架构 31B 大容量指令模型",
        "tags": ["Gemma4", "High-Capacity"],
    },
    {
        "id": "mlx-community/idefics2-8b-chatty-4bit",
        "name": "Idefics2 8B Chatty (4-bit)",
        "ram_required_gb": 6,
        "desc": "8B 快速轻量级多模态视觉模型，占用极小",
        "tags": ["Lightweight", "Vision"],
    },
    {
        "id": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        "name": "Qwen 2.5 VL 7B (4-bit)",
        "ram_required_gb": 6,
        "desc": "7B 极速视觉问答与图表分析模型",
        "tags": ["Vision", "Fast", "Lightweight"],
    },
    {
        "id": "mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
        "name": "Qwen 2.5 Coder 32B (4-bit)",
        "ram_required_gb": 20,
        "desc": "32B 编程与量化策略代码生成专家",
        "tags": ["Coding", "Quant"],
    },
    {
        "id": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
        "name": "DeepSeek R1 Distill Qwen 32B (4-bit)",
        "ram_required_gb": 20,
        "desc": "32B 深度思维链推理模型 (DeepSeek R1 蒸馏版)",
        "tags": ["Reasoning", "CoT"],
    },
    {
        "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "name": "TinyLlama 1.1B Chat",
        "ram_required_gb": 2,
        "desc": "1.1B 超轻量快速调试与本地冒烟测试",
        "tags": ["Debug", "Ultra-Fast"],
    },
]

# Non-chat auxiliary model prefixes to filter from user model list
IGNORED_MODEL_PREFIXES = ("timesfm", "DFlash", "assistant")


def get_hardware_info() -> Dict[str, Any]:
    """Retrieve Mac Apple Silicon hardware specs (Chip, Unified RAM in GB)."""
    ram_gb = 16.0
    chip_name = "Apple Silicon"

    # Memory
    try:
        res = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True)
        ram_bytes = int(res.stdout.strip())
        ram_gb = round(ram_bytes / (1024 ** 3), 1)
    except Exception:
        pass

    # CPU/Chip
    try:
        res = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True)
        brand = res.stdout.strip()
        if brand:
            chip_name = brand
    except Exception:
        pass

    return {
        "chip": chip_name,
        "ram_gb": ram_gb,
    }


def get_cached_models() -> List[Dict[str, Any]]:
    """Scan ~/.cache/huggingface/hub for downloaded models."""
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    cached = []
    if not hub_dir.exists():
        return cached

    for model_dir in sorted(hub_dir.glob("models--*")):
        dir_name = model_dir.name
        # format: models--org--name
        parts = dir_name.replace("models--", "").split("--")
        if len(parts) >= 2:
            repo_id = f"{parts[0]}/{'--'.join(parts[1:])}"
        else:
            repo_id = parts[0]

        # Filter out drafters / auxiliary weights
        if any(ignored.lower() in repo_id.lower() for ignored in IGNORED_MODEL_PREFIXES):
            continue

        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
            # Calculate approx size on disk
            try:
                size_bytes = sum(f.stat().st_size for f in model_dir.glob("**/*") if f.is_file())
                size_gb = round(size_bytes / (1024 ** 3), 1)
            except Exception:
                size_gb = 0.0

            cached.append({
                "id": repo_id,
                "size_gb": size_gb,
                "local_path": str(snapshots_dir),
            })

    return cached


def get_model_catalog() -> List[Dict[str, Any]]:
    """Merge locally cached models with curated community recommendations."""
    cached_map = {item["id"]: item for item in get_cached_models()}
    hw = get_hardware_info()
    ram_gb = hw["ram_gb"]

    catalog = []
    seen_ids = set()

    # 1. Process curated models first
    for item in CURATED_MODELS:
        repo_id = item["id"]
        is_cached = repo_id in cached_map
        disk_size = cached_map[repo_id]["size_gb"] if is_cached else None
        
        req_ram = item.get("ram_required_gb", 16)
        fits_memory = ram_gb >= req_ram

        catalog.append({
            "id": repo_id,
            "name": item.get("name", repo_id),
            "cached": is_cached,
            "disk_size_gb": disk_size,
            "ram_required_gb": req_ram,
            "fits_memory": fits_memory,
            "desc": item.get("desc", ""),
            "tags": item.get("tags", []),
        })
        seen_ids.add(repo_id)

    # 2. Append any extra locally cached models not in curated list
    for repo_id, info in cached_map.items():
        if repo_id not in seen_ids:
            catalog.append({
                "id": repo_id,
                "name": repo_id,
                "cached": True,
                "disk_size_gb": info["size_gb"],
                "ram_required_gb": int(info["size_gb"] * 1.2) if info["size_gb"] > 0 else 16,
                "fits_memory": True,
                "desc": f"本地自定义缓存模型 ({info['size_gb']} GB)",
                "tags": ["Local"],
            })
            seen_ids.add(repo_id)

    return catalog


def render_interactive_menu(default_model: Optional[str] = None) -> str:
    """Display interactive CLI selection menu and return chosen model ID."""
    hw = get_hardware_info()
    catalog = get_model_catalog()

    # Separate cached vs available to download
    cached_items = [m for m in catalog if m["cached"]]
    remote_items = [m for m in catalog if not m["cached"]]

    # ANSI Colors
    C_RESET = "\033[0m"
    C_BOLD = "\033[1m"
    C_CYAN = "\033[36m"
    C_GREEN = "\033[32m"
    C_YELLOW = "\033[33m"
    C_GRAY = "\033[90m"
    C_MAGENTA = "\033[35m"

    print(f"\n{C_BOLD}{C_CYAN}================================================================================{C_RESET}")
    print(f"{C_BOLD}  🚀 XMLX-VLM 模型选择器{C_RESET} {C_GRAY}(硬件: {C_GREEN}{hw['chip']}{C_GRAY} | 统一内存: {C_GREEN}{hw['ram_gb']} GB{C_GRAY}){C_RESET}")
    print(f"{C_BOLD}{C_CYAN}================================================================================{C_RESET}")

    choices_map: Dict[int, Dict[str, Any]] = {}
    idx = 1

    # Section 1: Local Cached Models
    print(f"\n{C_BOLD}{C_GREEN}📦 本地已下载模型 (无需下载，即选即用):{C_RESET}")
    if not cached_items:
        print(f"  {C_GRAY}(暂无本地缓存模型){C_RESET}")
    else:
        for m in cached_items:
            choices_map[idx] = m
            size_str = f" [{m['disk_size_gb']}G]" if m.get("disk_size_gb") else ""
            desc = f" - {m['desc']}" if m.get("desc") else ""
            print(f"  {C_BOLD}{C_CYAN}[{idx}]{C_RESET} {C_BOLD}{m['id']}{C_RESET}{C_GREEN}{size_str}{C_RESET}{C_GRAY}{desc}{C_RESET}")
            idx += 1

    # Section 2: Recommended Remote Models
    print(f"\n{C_BOLD}{C_YELLOW}🌐 社区精选推荐 (初次使用将自动下载权重):{C_RESET}")
    for m in remote_items:
        choices_map[idx] = m
        mem_tag = f"需 ~{m['ram_required_gb']}G 内存"
        mem_color = C_GREEN if m["fits_memory"] else C_YELLOW
        desc = f" - {m['desc']}" if m.get("desc") else ""
        print(f"  {C_BOLD}{C_YELLOW}[{idx}]{C_RESET} {m['id']} {mem_color}({mem_tag}){C_RESET}{C_GRAY}{desc}{C_RESET}")
        idx += 1

    print(f"\n  {C_BOLD}[0]{C_RESET} {C_MAGENTA}输入自定义 HuggingFace Repo ID 或本地路径...{C_RESET}")
    print(f"{C_BOLD}{C_CYAN}================================================================================{C_RESET}")

    # Determine default option index
    default_idx = 1
    if default_model:
        for i, m in choices_map.items():
            if m["id"] == default_model:
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
            sys.stdout.write(f"{C_BOLD}请输入自定义模型 ID (如 mlx-community/Qwen2.5-7B-Instruct-4bit): {C_RESET}")
            sys.stdout.flush()
            custom_model = sys.stdin.readline().strip()
            return custom_model if custom_model else default_name

        try:
            choice_num = int(user_input)
            if choice_num in choices_map:
                selected = choices_map[choice_num]["id"]
                return selected
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
            print(json.dumps({
                "hardware": get_hardware_info(),
                "models": get_model_catalog(),
            }, ensure_ascii=False, indent=2))
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
