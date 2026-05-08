# SPDX-License-Identifier: Apache-2.0
"""
Runtime patches for model-specific fixes in mlx_vlm.

These are monkey-patches applied at model load time to fix compatibility
issues or enable features (MTP, BatchKVCache, etc.) that the base
mlx_vlm / mlx_lm implementations don't yet support.
"""

import logging

logger = logging.getLogger(__name__)


def apply_all_patches() -> dict[str, bool]:
    """Apply all available runtime patches. Returns {name: applied}."""
    results = {}

    # Qwen3.5 BatchKVCache fix
    try:
        from .qwen3_5_mllm import patch_qwen35_attention_for_batching

        results["qwen35_batchkv"] = patch_qwen35_attention_for_batching()
    except Exception as e:
        logger.debug("Qwen3.5 patch skipped: %s", e)
        results["qwen35_batchkv"] = False

    # Gemma4 MLLM patches
    try:
        from .gemma4_mllm import apply_gemma4_patches

        results["gemma4"] = apply_gemma4_patches()
    except Exception as e:
        logger.debug("Gemma4 patch skipped: %s", e)
        results["gemma4"] = False

    # GLM4V MoE patches
    try:
        from .glm4v_moe_mllm import apply_glm4v_moe_patches

        results["glm4v_moe"] = apply_glm4v_moe_patches()
    except Exception as e:
        logger.debug("GLM4V MoE patch skipped: %s", e)
        results["glm4v_moe"] = False

    applied = [k for k, v in results.items() if v]
    if applied:
        logger.info("Runtime patches applied: %s", ", ".join(applied))
    return results
