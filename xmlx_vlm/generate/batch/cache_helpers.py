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
):
    """
    Convert a list of regular caches into their corresponding
    batch-aware caches.

    When *kv_bits* is set, a quantized batch cache is used instead of
    ``BatchKVCache`` so that KV states are quantized on-the-fly during
    generation, reducing memory usage for long sequences.

    *kv_quant_scheme* selects the quantization backend:
    - ``"uniform"`` → ``BatchQuantizedKVCache`` (``mx.quantize``)
    - ``"turboquant"`` or fractional *kv_bits* → ``BatchTurboQuantKVCache``
    """
    use_turbo = kv_bits is not None and turboquant_enabled(kv_bits, kv_quant_scheme)

    def _make_quant_cache(lp):
        if use_turbo:
            return BatchTurboQuantKVCache(lp, bits=kv_bits)
        return cache.BatchQuantizedKVCache(
            lp, group_size=kv_group_size, bits=int(kv_bits)
        )

    def to_batch_cache(c, quantize=True):
        if isinstance(c, cache.KVCache):
            if kv_bits is not None and quantize:
                return _make_quant_cache(left_padding)
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.ChunkedKVCache):
            if kv_bits is not None and quantize:
                return _make_quant_cache(left_padding)
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.SimpleKVCache):
            if kv_bits is not None and quantize:
                return _make_quant_cache(left_padding)
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.ArraysCache):
            c.left_padding = mx.array(left_padding)
            return c
        elif isinstance(c, cache.RotatingKVCache):
            if c.keep > 0:
                raise ValueError("RotatingKVCache with keep tokens is not supported.")
            return cache.BatchRotatingKVCache(c.max_size, left_padding)
        elif isinstance(c, cache.CacheList):
            return cache.CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
        elif isinstance(c, tuple):
            return cache.CacheList(*(to_batch_cache(sub_c) for sub_c in c))
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    if hasattr(model, "make_cache"):
        model_cache = model.make_cache()
        n = len(model_cache)
        # Skip quantizing the last layer — it's sensitive to quantization
        return [
            to_batch_cache(c, quantize=(i < n - 1 if n > 2 else True))
            for i, c in enumerate(model_cache)
        ]
    else:
        if kv_bits is not None:
            n = len(model.layers)
            return [
                (
                    _make_quant_cache(left_padding)
                    if i < n - 1 or n <= 2
                    else cache.BatchKVCache(left_padding)
                )
                for i in range(n)
            ]
        return [cache.BatchKVCache(left_padding) for _ in model.layers]

