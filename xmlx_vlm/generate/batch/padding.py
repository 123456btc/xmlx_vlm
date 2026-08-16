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

_SEQUENCE_ALIGNED_PROMPT_KWARGS = {'attention_mask', 'decoder_inputs_embeds', 'deepstack_visual_embeds', 'visual_pos_masks', 'per_layer_inputs', 'full_text_row_masked_out_mask', 'position_ids', 'pos_hw'}
APC_PRIVATE_PROMPT_KEYS = ('_apc_tenant', '_apc_image_hash')

def _left_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)

    return mx.array([[0] * (max_length - len(p)) + p for p in prompts])

def _right_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)

    return mx.array([list(p) + [0] * (max_length - len(p)) for p in prompts])

def _prompt_kwarg_row(v: mx.array, row_idx: int, batch_size: int) -> mx.array:
    if v.shape[0] == batch_size:
        return v[row_idx : row_idx + 1]
    return v[:1]

def _split_prompt_kwargs_per_row(prompt_kwargs: dict, batch_size: int) -> List[dict]:
    """Normalize batched prompt kwargs into one dict per batch row.

    ``model.get_input_embeddings()`` commonly returns batch-sized tensors
    (notably ``inputs_embeds``). ``BatchGenerator.insert()`` stores prompt
    kwargs per sequence, so passing the same batched dict for every row causes
    the prompt builder to concatenate those batched tensors ``batch_size``
    times, effectively squaring the batch dimension.
    """
    if batch_size <= 1:
        return [prompt_kwargs or {}]

    rows = [{} for _ in range(batch_size)]
    for k, v in (prompt_kwargs or {}).items():
        if isinstance(v, mx.array) and v.ndim > 0 and v.shape[0] >= 1:
            for i in range(batch_size):
                rows[i][k] = _prompt_kwarg_row(v, i, batch_size)
        else:
            for row in rows:
                row[k] = v
    return rows

def _is_sequence_aligned_prompt_kwarg(
    key: str, v: mx.array, sequence_length: int
) -> bool:
    return (
        key in _SEQUENCE_ALIGNED_PROMPT_KWARGS
        and v.ndim >= 2
        and v.shape[1] == sequence_length
    )

def _pad_sequence_aligned_prompt_kwarg(
    v: mx.array, target_length: int, *, left: bool
) -> mx.array:
    pad = target_length - v.shape[1]
    if pad <= 0:
        return v
    pad_shape = (v.shape[0], pad) + tuple(v.shape[2:])
    pad_v = mx.zeros(pad_shape, dtype=v.dtype)
    parts = [pad_v, v] if left else [v, pad_v]
    return mx.concatenate(parts, axis=1)

def _merge_prefill_prompt_kwargs(
    prompt_kwargs_list: List[Optional[dict]],
    input_ids: List[List[int]],
) -> Tuple[mx.array, dict]:
    """Batch per-row prompt kwargs for a left-padded prefill forward."""
    lengths = [len(ids) for ids in input_ids]
    max_length = max(lengths)

    has_any_embeds = any(kw and kw.get("inputs_embeds") is not None for kw in prompt_kwargs_list)
    inputs_embeds = None
    if has_any_embeds:
        row_embeds: List[mx.array] = []
        embed_dtype = None
        embed_dim = None
        for kw, length in zip(prompt_kwargs_list, lengths):
            if not kw or kw.get("inputs_embeds") is None:
                raise ValueError("inputs_embeds is required when mixing embedded prompts")
            embeds = kw["inputs_embeds"]  # [1, length, D]
            embed_dtype = embeds.dtype
            embed_dim = embeds.shape[-1]
            if length < max_length:
                pad = mx.zeros(
                    (embeds.shape[0], max_length - length, embed_dim),
                    dtype=embed_dtype,
                )
                embeds = mx.concatenate([pad, embeds], axis=1)
            row_embeds.append(embeds)
        inputs_embeds = mx.concatenate(row_embeds, axis=0)

    merged_kwargs: dict = {}
    per_row_keys: dict = {}
    batch_size = len(prompt_kwargs_list)
    for i, (kw, length) in enumerate(zip(prompt_kwargs_list, lengths)):
        if not kw:
            continue
        for k, v in kw.items():
            if k == "inputs_embeds" or k in APC_PRIVATE_PROMPT_KEYS:
                continue
            if isinstance(v, mx.array) and v.ndim > 0 and v.shape[0] >= 1:
                row_v = _prompt_kwarg_row(v, i, batch_size)
                if _is_sequence_aligned_prompt_kwarg(k, row_v, length):
                    row_v = _pad_sequence_aligned_prompt_kwarg(
                        row_v, max_length, left=True
                    )
                per_row_keys.setdefault(k, []).append(row_v)
            elif k not in merged_kwargs:
                merged_kwargs[k] = v
    for k, vs in per_row_keys.items():
        merged_kwargs[k] = mx.concatenate(vs, axis=0)

    return inputs_embeds, merged_kwargs

