#!/usr/bin/env python3
"""
MLX-VLM model benchmark for coding/development.

Measures TTFT, TPOT, TPS and memory usage for text generation.
Usage:
    python -m xmlx_vlm.bench --model mlx-community/diffusiongemma-26B-A4B-it-4bit
    python -m xmlx_vlm.bench --model mlx-community/Llama-3.2-3B-Instruct-4bit --prompts 10 --max-tokens 512
"""

import os
import time
import statistics
import argparse
from dataclasses import dataclass, field
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import mlx.core as mx
from tabulate import tabulate

from .config import DEFAULT_MODEL
from .utils import load
from .generate import generate
from .optimizations import detect_hardware


@dataclass
class BenchResult:
    prompt_tokens: int = 0
    generated_tokens: int = 0
    ttft: float = 0.0
    total_time: float = 0.0
    tpot: float = 0.0
    gen_tps: float = 0.0
    proc_tps: float = 0.0


def _reset_peak_memory():
    try:
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        elif hasattr(mx.metal, "reset_peak_memory"):
            mx.metal.reset_peak_memory()
    except Exception:
        pass


def _get_mlx_mem():
    try:
        return {
            "active": mx.get_active_memory() / (1024**3),
            "cache": mx.get_cache_memory() / (1024**3),
            "peak": mx.get_peak_memory() / (1024**3),
        }
    except Exception:
        return {}


def benchmark_once(model, processor, prompt: str, max_tokens: int, temperature: float) -> Optional[BenchResult]:
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    prompt_tokens = len(tokenizer.encode(prompt))

    _reset_peak_memory()
    start = time.perf_counter()
    ttft = None
    gen_tokens = 0

    for chunk in generate(
        model=model,
        processor=processor,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        verbose=False,
    ):
        gen_tokens += 1
        if ttft is None:
            ttft = time.perf_counter() - start

    total = time.perf_counter() - start
    if ttft is None:
        ttft = total

    gen_time = total - ttft
    return BenchResult(
        prompt_tokens=prompt_tokens,
        generated_tokens=gen_tokens,
        ttft=ttft,
        total_time=total,
        tpot=gen_time / max(1, gen_tokens - 1) if gen_tokens > 1 else 0,
        gen_tps=(gen_tokens - 1) / max(gen_time, 1e-6) if gen_tokens > 1 else 0,
        proc_tps=prompt_tokens / max(ttft, 1e-6),
    )


def run(model_name: str, num_prompts: int = 5, max_tokens: int = 256, temperature: float = 0.7, warmup: int = 1):
    hw = detect_hardware()
    prompts = [
        "Hello, how are you?",
        "What is 2+2?",
        "Write a Python function to calculate fibonacci numbers.",
        "Explain the difference between list and tuple in Python.",
        "Design a REST API for a todo app with Flask.",
        "You are a senior architect. Design a microservices system for e-commerce.",
        "Explain quantum computing in detail.",
        "Write a comprehensive guide to building production-ready Python APIs.",
        "Describe photosynthesis in scientific detail.",
        "Implement a LRU cache in Python with O(1) get and put.",
    ][:num_prompts]

    print(f"\n{'='*60}")
    print("MLX-VLM Benchmark")
    print(f"{'='*60}")
    print(tabulate([
        ["Model", model_name],
        ["Hardware", f"{hw.chip_name} ({hw.total_memory_gb:.0f} GB)"],
        ["Memory BW", f"{hw.memory_bandwidth_gbs} GB/s"],
        ["GPU Cores", hw.gpu_cores],
        ["Prompts", num_prompts],
        ["Max Tokens", max_tokens],
    ], tablefmt="plain"))
    print(f"{'='*60}\n")

    print(f"Loading {model_name}...")
    t0 = time.perf_counter()
    model, processor = load(model_name)
    print(f"Loaded in {time.perf_counter()-t0:.2f}s\n")

    # Warmup
    if warmup > 0:
        print(f"Warmup ({warmup} run{'s' if warmup>1 else ''})...")
        for _ in range(warmup):
            benchmark_once(model, processor, prompts[0], max_tokens=16, temperature=0.0)
        print("Done\n")

    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"  Prompt {i}/{len(prompts)} ...", end=" ", flush=True)
        r = benchmark_once(model, processor, prompt, max_tokens, temperature)
        if r:
            results.append(r)
            print(f"TTFT={r.ttft*1000:.0f}ms TPOT={r.tpot*1000:.0f}ms TPS={r.gen_tps:.1f}")
        else:
            print("FAILED")

    if not results:
        print("No successful runs.")
        return

    mem = _get_mlx_mem()
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(tabulate([
        ["TTFT (ms)", f"{statistics.mean(r.ttft for r in results)*1000:.1f}", f"min={min(r.ttft for r in results)*1000:.1f}", f"max={max(r.ttft for r in results)*1000:.1f}"],
        ["TPOT (ms)", f"{statistics.mean(r.tpot for r in results)*1000:.1f}", f"min={min(r.tpot for r in results)*1000:.1f}", f"max={max(r.tpot for r in results)*1000:.1f}"],
        ["Gen TPS", f"{statistics.mean(r.gen_tps for r in results):.1f}", f"max={max(r.gen_tps for r in results):.1f}", ""],
        ["Proc TPS", f"{statistics.mean(r.proc_tps for r in results):.1f}", "", ""],
        ["MLX Memory", f"active={mem.get('active',0):.2f}GB", f"cache={mem.get('cache',0):.2f}GB", f"peak={mem.get('peak',0):.2f}GB"],
    ], headers=["Metric", "Mean", "Min", "Max"], tablefmt="simple"))
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="MLX-VLM benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name or path (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompts", type=int, default=5, help="Number of prompts")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    args = parser.parse_args()
    run(args.model, args.prompts, args.max_tokens, args.temperature, args.warmup)


if __name__ == "__main__":
    main()
