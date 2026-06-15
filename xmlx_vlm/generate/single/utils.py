from __future__ import annotations
import contextlib
import functools
import logging
import os
import time
import warnings
from collections.abc import Sequence
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from mlx_lm.generate import maybe_quantize_kv_cache as mlx_maybe_quantize_kv_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ... import apc as _apc
from ... import diffusion_generate
from ...models import cache
from ...prompt_utils import apply_chat_template
from ...tokenizer_utils import make_streaming_detokenizer
from ...turboquant import TurboQuantKVCache, turboquant_enabled
from ...utils import StoppingCriteria, ThinkingBudgetCriteria, prepare_inputs

from ..types import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_SEED,
    DEFAULT_TOP_K,
    DEFAULT_MIN_P,
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_THINKING_START_TOKEN,
    DEFAULT_THINKING_END_TOKEN,
    DEFAULT_PREFILL_STEP_SIZE,
    GenerationResult,
    PromptCacheState,
    generation_stream,
)

logger = logging.getLogger('xmlx_vlm.generate')

generation_stream = mx.new_thread_local_stream(mx.default_device())

def normalize_resize_shape(
    values: Optional[Sequence[int]],
) -> Optional[Tuple[int, int]]:
    if values is None:
        return None
    if not (
        isinstance(values, Sequence)
        and not isinstance(values, (str, bytes))
        and len(values) in (1, 2)
        and all(type(value) is int for value in values)
    ):
        raise ValueError("resize_shape must contain 1 or 2 integers")
    return (values[0], values[0]) if len(values) == 1 else tuple(values)

def _get_adaptive_bits_list(num_layers: int, kv_bits: float) -> list[float]:
    bits_list = [kv_bits] * num_layers
    if num_layers < 6 or kv_bits - 0.5 < 1.0:
        return bits_list

    # Critical layers: first 2 and last 2
    # Middle layers: 2 to num_layers - 3 (M layers)
    M = num_layers - 4
    if M >= 4:
        # Subtract 0.5 from exactly 4 middle layers to compensate for boosting 4 critical layers by 0.5
        indices_to_reduce = {2, 2 + M // 3, num_layers - 3 - M // 3, num_layers - 3}
        if len(indices_to_reduce) < 4:
            indices_to_reduce = {2, 3, num_layers - 4, num_layers - 3}
        
        for idx in [0, 1, num_layers - 2, num_layers - 1]:
            bits_list[idx] = kv_bits + 0.5
        for idx in sorted(indices_to_reduce):
            bits_list[idx] = kv_bits - 0.5
    elif M >= 2:
        # Subtract 0.5 from exactly 2 middle layers to compensate for boosting 2 critical layers by 0.5
        bits_list[0] = kv_bits + 0.5
        bits_list[-1] = kv_bits + 0.5
        bits_list[2] = kv_bits - 0.5
        bits_list[num_layers - 3] = kv_bits - 0.5
        
    return bits_list

def maybe_quantize_kv_cache(
    prompt_cache,
    quantized_kv_start,
    kv_group_size,
    kv_bits,
    kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
):
    if kv_bits is None:
        return

    if turboquant_enabled(kv_bits, kv_quant_scheme):

        def quantize_entry(entry, layer_bits):
            if isinstance(entry, TurboQuantKVCache):
                return entry
            if isinstance(entry, cache.RotatingKVCache):
                return entry
            if isinstance(entry, cache.KVCache):
                if entry.offset == 0:
                    # Empty: replace so update_and_fetch quantizes on the fly
                    return TurboQuantKVCache(bits=layer_bits)
                if entry.offset < quantized_kv_start:
                    return entry
                return TurboQuantKVCache.from_cache(entry, bits=layer_bits)
            if isinstance(entry, cache.CacheList):
                entry.caches = [quantize_entry(sub_entry, layer_bits) for sub_entry in entry.caches]
                return entry
            if isinstance(entry, list):
                for i, sub_entry in enumerate(entry):
                    entry[i] = quantize_entry(sub_entry, layer_bits)
                return entry
            if isinstance(entry, tuple):
                return tuple(quantize_entry(sub_entry, layer_bits) for sub_entry in entry)
            return entry

        # Skip the last layer (before final norm/LM head) — it's highly
        # sensitive to quantization in deep models (e.g. gemma-4-31b).
        last_idx = len(prompt_cache) - 1 if len(prompt_cache) > 2 else -1
        num_quant_layers = len(prompt_cache) - 1 if last_idx != -1 else len(prompt_cache)
        adaptive_bits = _get_adaptive_bits_list(num_quant_layers, kv_bits)
        
        quant_idx = 0
        for index, layer_cache in enumerate(prompt_cache):
            if index == last_idx:
                continue
            layer_bits = adaptive_bits[quant_idx]
            prompt_cache[index] = quantize_entry(layer_cache, layer_bits)
            quant_idx += 1
        return

    mlx_maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=int(kv_bits),
    )

@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        yield
        return

    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    max_rec_size = mx.device_info()["max_recommended_working_set_size"]
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        print(
            f"[WARNING] Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    old_limit = mx.set_wired_limit(max_rec_size)
    try:
        yield
    finally:
        if streams is not None:
            for s in streams:
                mx.synchronize(s)
        else:
            mx.synchronize()
        mx.set_wired_limit(old_limit)

def _prime_cached_prefix_rope_state(
    model: nn.Module,
    full_input_ids: mx.array,
    mask: Optional[mx.array],
    kwargs: Dict[str, Any],
) -> bool:
    """Prime Qwen-style mRoPE metadata before a cached-prefix trim.

    Qwen VL language models keep ``_rope_deltas`` on the model object and use
    it when continuing from a non-empty KV cache. If APC trims the prompt to
    only the uncached suffix, the suffix alone is not enough to recompute the
    original prompt's RoPE delta, so derive it from the full prompt first.
    """
    lm = getattr(model, "language_model", None)
    get_rope_index = getattr(lm, "get_rope_index", None)
    if not callable(get_rope_index):
        return True
    if not (hasattr(lm, "_rope_deltas") or hasattr(lm, "_position_ids")):
        return True
    try:
        position_ids, rope_deltas = get_rope_index(
            full_input_ids,
            kwargs.get("image_grid_thw", None),
            kwargs.get("video_grid_thw", None),
            mask,
        )
    except Exception as e:
        logger.warning(
            "Could not prime cached-prefix RoPE state; falling back to cold prefill: %s",
            e,
        )
        return False
    if hasattr(lm, "_position_ids"):
        lm._position_ids = position_ids
    if hasattr(lm, "_rope_deltas"):
        lm._rope_deltas = rope_deltas
    kwargs["rope_deltas"] = rope_deltas
    return True

def _apply_rep_penalty(logits: mx.array, ctx_tokens: mx.array, penalty: float) -> mx.array:
    """Apply repetition penalty to a block of logits.

    Args:
        logits:     ``[B, L, V]`` verify logits (float).
        ctx_tokens: ``[K]`` int array of recent token ids to penalise.
        penalty:    Penalty factor > 1.0.  Positive logits are divided;
                    negative logits are multiplied (standard formulation).

    Returns logits with the same shape as input.
    """
    if ctx_tokens is None or ctx_tokens.size == 0 or penalty == 1.0:
        return logits
    # Boolean mask over vocabulary — True for context tokens
    V = logits.shape[-1]
    mask = mx.zeros(V, dtype=mx.bool_)
    mask = mask.at[ctx_tokens].add(True)  # [V]
    is_pos = logits > 0                                           # [B, L, V]
    return mx.where(
        mask[None, None, :],
        mx.where(is_pos, logits / penalty, logits * penalty),
        logits,
    )

