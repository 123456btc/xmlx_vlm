from __future__ import annotations
import contextlib
import logging
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ... import apc as _apc
from ...models import cache
from ...prompt_utils import apply_chat_template
from ...tokenizer_utils import make_streaming_detokenizer
from ...turboquant import BatchTurboQuantKVCache, turboquant_enabled
from ...utils import StoppingCriteria, ThinkingBudgetCriteria, group_images_by_shape, prepare_inputs

from ..types import (
    DEFAULT_COMPLETION_BATCH_SIZE,
    DEFAULT_PREFILL_BATCH_SIZE,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_THINKING_START_TOKEN,
    DEFAULT_THINKING_END_TOKEN,
    DEFAULT_MAX_TOKENS,
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_QUANTIZED_KV_START,
    BatchGenerationResult,
    BatchStats,
    BatchResponse,
    generation_stream,
)
from ..single import normalize_resize_shape, wired_limit

logger = logging.getLogger('xmlx_vlm.generate')

def _extend_cache(cache_a, cache_b):
    """Extend cache_a with cache_b along the batch dimension."""
    if not cache_a:
        return cache_b
    if not cache_b:
        return cache_a
    for ca, cb in zip(cache_a, cache_b):
        ca.extend(cb)
    return cache_a

def _make_cache(
    model,
    left_padding,
    kv_bits=None,
    kv_group_size=64,
    kv_quant_scheme=DEFAULT_KV_QUANT_SCHEME,
    kv_bits_per_layer=None,
):
    """
    Convert a list of regular caches into their corresponding
    batch-aware caches.

    When *kv_bits* or *kv_bits_per_layer* is set, a quantized batch cache is used
    instead of ``BatchKVCache`` so that KV states are quantized on-the-fly during
    generation, reducing memory usage for long sequences.

    *kv_quant_scheme* selects the quantization backend:
    - ``"uniform"`` → ``BatchQuantizedKVCache`` (``mx.quantize``)
    - ``"turboquant"`` or fractional *kv_bits* → ``BatchTurboQuantKVCache``
    """
    if hasattr(model, "make_cache"):
        model_cache = model.make_cache()
        num_layers = len(model_cache)
    elif hasattr(model, "layers"):
        model_cache = None
        num_layers = len(model.layers)
    else:
        model_cache = None
        num_layers = 1

    from ...config import get_kv_bits_per_layer

    bits_per_layer = None
    if kv_bits_per_layer is not None:
        if isinstance(kv_bits_per_layer, str):
            bits_per_layer = get_kv_bits_per_layer(num_layers, kv_bits, raw_config=kv_bits_per_layer)
        elif isinstance(kv_bits_per_layer, (list, tuple)):
            bits_per_layer = list(kv_bits_per_layer)
            if len(bits_per_layer) < num_layers:
                pad_val = kv_bits if kv_bits is not None else bits_per_layer[-1]
                bits_per_layer = bits_per_layer + [pad_val] * (num_layers - len(bits_per_layer))
            elif len(bits_per_layer) > num_layers:
                bits_per_layer = bits_per_layer[:num_layers]
    elif kv_bits is not None:
        bits_per_layer = get_kv_bits_per_layer(num_layers, kv_bits)
        if bits_per_layer is None:
            bits_per_layer = [kv_bits] * num_layers

    def _make_quant_cache(lp, layer_bits):
        if layer_bits is None:
            return cache.BatchKVCache(lp)
        use_turbo = turboquant_enabled(layer_bits, kv_quant_scheme)
        if use_turbo:
            return BatchTurboQuantKVCache(lp, bits=layer_bits)
        return cache.BatchQuantizedKVCache(
            lp, group_size=kv_group_size, bits=int(layer_bits)
        )

    def to_batch_cache(c, layer_idx=0, quantize=True):
        layer_bits = bits_per_layer[layer_idx] if (bits_per_layer and layer_idx < len(bits_per_layer)) else kv_bits
        if isinstance(c, (cache.KVCache, cache.ChunkedKVCache, cache.SimpleKVCache)):
            if layer_bits is not None and quantize:
                return _make_quant_cache(left_padding, layer_bits)
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.ArraysCache):
            c.left_padding = mx.array(left_padding)
            return c
        elif isinstance(c, cache.RotatingKVCache):
            if c.keep > 0:
                raise ValueError("RotatingKVCache with keep tokens is not supported.")
            return cache.BatchRotatingKVCache(c.max_size, left_padding)
        elif isinstance(c, cache.CacheList):
            return cache.CacheList(*(to_batch_cache(sub_c, layer_idx, quantize) for sub_c in c.caches))
        elif isinstance(c, tuple):
            return cache.CacheList(*(to_batch_cache(sub_c, layer_idx, quantize) for sub_c in c))
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    explicit_per_layer = bool(kv_bits_per_layer is not None or os.environ.get("XMLX_VLM_KV_BITS_PER_LAYER"))

    if model_cache is not None:
        n = len(model_cache)
        return [
            to_batch_cache(c, layer_idx=i, quantize=(True if explicit_per_layer else (i < n - 1 if n > 2 else True)))
            for i, c in enumerate(model_cache)
        ]
    else:
        if bits_per_layer is not None:
            n = len(model.layers)
            return [
                (
                    _make_quant_cache(left_padding, bits_per_layer[i])
                    if (explicit_per_layer or i < n - 1 or n <= 2)
                    else cache.BatchKVCache(left_padding)
                )
                for i in range(n)
            ]
        return [cache.BatchKVCache(left_padding) for _ in model.layers]

