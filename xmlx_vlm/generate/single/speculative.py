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

from .utils import normalize_resize_shape, maybe_quantize_kv_cache, wired_limit

def _speculative_walk(
    draft_tokens: mx.array,
    target_tokens: mx.array,
    budget: int,
) -> Tuple[int, List[int]]:
    """Exact-greedy speculative-decoding walk.

    Accept drafted tokens up to the first mismatch with the target's
    greedy choice, then take the target's bonus at that position.
    Returns ``(accepted_count, new_tokens)`` with ``new_tokens``
    truncated to ``budget``.
    """
    n_draft = draft_tokens.shape[1]
    combined = mx.concatenate(
        [draft_tokens.reshape(-1), target_tokens.reshape(-1)]
    ).tolist()
    d = combined[:n_draft]
    t = combined[n_draft:]
    accepted = next((i for i in range(len(d)) if d[i] != t[i]), len(d))
    new_tokens = (d[:accepted] + [t[accepted]])[:budget]
    return accepted, new_tokens

def _speculative_walk_batch(
    draft_tokens: mx.array,
    target_tokens: mx.array,
    budgets: List[int],
) -> Tuple[List[int], List[List[int]]]:
    """Per-sequence speculative walk for B > 1.

    Returns ``(accepted_list, new_tokens_list)`` where each entry
    corresponds to one sequence in the batch.
    """
    B = draft_tokens.shape[0]
    n_draft = draft_tokens.shape[1]
    combined = mx.concatenate(
        [draft_tokens.reshape(B, -1), target_tokens.reshape(B, -1)], axis=1
    ).tolist()
    accepted_list: List[int] = []
    new_tokens_list: List[List[int]] = []
    for i in range(B):
        d = combined[i][:n_draft]
        t = combined[i][n_draft:]
        acc = next((j for j in range(len(d)) if d[j] != t[j]), len(d))
        new = (d[:acc] + [t[acc]])[: budgets[i]]
        accepted_list.append(acc)
        new_tokens_list.append(new)
    return accepted_list, new_tokens_list

