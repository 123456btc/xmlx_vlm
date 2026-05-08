# SPDX-License-Identifier: Apache-2.0
"""MoE top-k override for inference speedup.

Reduces the number of experts activated per token in Mixture-of-Experts
models (Qwen3 MoE, Qwen3.5 MoE, Gemma-MoE, etc.), trading a small amount
of quality for higher decode throughput.

Usage:
    from xmlx_vlm.moe_topk import apply_moe_top_k_override
    apply_moe_top_k_override(model, k=4)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def apply_moe_top_k_override(model: Any, k: int) -> None:
    """Override top_k on every sparse-MoE layer in the model.

    Args:
        model: Loaded MLX model.
        k: New top_k value (must be <= trained top_k).

    Raises:
        ValueError: If k > trained top_k on any layer.
    """
    if k <= 0:
        raise ValueError(f"moe-top-k must be > 0, got {k}")

    layers = getattr(model, "layers", None)
    if layers is None:
        logger.warning("moe_topk: model has no .layers, skipping")
        return

    patched = 0
    for idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        switch_mlp = getattr(mlp, "switch_mlp", None)
        if switch_mlp is None:
            continue

        trained_top_k = getattr(switch_mlp, "top_k", None)
        if trained_top_k is not None and k > trained_top_k:
            raise ValueError(
                f"moe-top-k {k} exceeds trained top_k={trained_top_k} on layer {idx}"
            )

        switch_mlp.top_k = k
        patched += 1

    if patched:
        logger.info("MoE top_k overridden to %d on %d layer(s)", k, patched)
    else:
        logger.debug("moe_topk: no sparse-MoE layers found, flag is a no-op")
