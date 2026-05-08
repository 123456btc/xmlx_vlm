#!/usr/bin/env python3
"""
Prompt warmup for mlx_vlm server.

At server startup, pre-populates the KV cache by running one short
generation per warm-up prompt. The first user request that shares a
prefix with a warmed prompt sees cache-hit TTFT instead of cold prefill.

For coding assistants the system prompt is usually fixed, so warming it
once makes the first completion feel instant.

Usage in server.py lifespan:
    from .prompt_warmup import warm_prompts
    warm_prompts(model, processor, [
        [{"role": "system", "content": "You are a helpful coding assistant."}]
    ])
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, List

from .prompt_utils import apply_chat_template
from .generate import generate

logger = logging.getLogger(__name__)


def load_warmup_file(path: str) -> List[List[dict[str, Any]]]:
    """Load warmup prompts from a JSON file.

    Format: list of message arrays, same as OpenAI messages field.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Warmup file not found: {p}")
    data = json.loads(p.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError("Warmup file must contain a non-empty JSON list")
    for i, entry in enumerate(data):
        if not isinstance(entry, list):
            raise ValueError(f"Entry {i}: expected list of messages")
        for j, msg in enumerate(entry):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ValueError(f"Entry {i} message {j}: missing role/content")
    return data


def warm_prompts(
    model,
    processor,
    prompts: List[List[dict[str, Any]]],
    max_tokens: int = 1,
    temperature: float = 0.0,
):
    """Run each prompt through generate() once to warm the KV cache.

    Args:
        model: loaded MLX model
        processor: tokenizer/processor
        prompts: list of message arrays
        max_tokens: tokens to generate (1 is enough for cache warmup)
        temperature: sampling temperature
    """
    if not prompts:
        return

    t0 = time.perf_counter()
    completed = 0
    total_prompt_tokens = 0

    for i, messages in enumerate(prompts):
        try:
            formatted = apply_chat_template(processor, model.config, messages)
            # Count prompt tokens
            tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            prompt_toks = len(tok.encode(formatted))
            total_prompt_tokens += prompt_toks

            # Generate 1 token to warm cache
            for _ in generate(
                model=model,
                processor=processor,
                prompt=formatted,
                max_tokens=max_tokens,
                temperature=temperature,
                verbose=False,
            ):
                pass
            completed += 1
            logger.info("[warmup] prompt %d/%d done (%d tokens)", i + 1, len(prompts), prompt_toks)
        except Exception as e:
            logger.warning("[warmup] prompt %d failed: %s", i, e)

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "[warmup] %d/%d completed in %.0fms (%d prompt tokens)",
        completed,
        len(prompts),
        elapsed,
        total_prompt_tokens,
    )
