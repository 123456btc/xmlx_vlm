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

def _dflash_rounds(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    first_bonus: int,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    processors: tuple = (),
    tokens_ctx: Optional[mx.array] = None,
) -> Generator[Tuple[int, None], None, None]:
    """DFlash speculative-decoding **round loop**.

    draft → verify → walk → rollback. ``generate_step`` is responsible
    for prefill, sampling the first bonus token, and packaging the
    captured hidden states into ``hidden``.

    ``processors`` / ``tokens_ctx`` enable repetition-penalty on the
    target model's verify logits: the penalty is applied per-position
    using the accumulated token context before each sampler call.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache. "
            "Speculative decoding with a DFlash drafter currently only "
            "supports mlx_vlm.models.qwen3_5."
        )

    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = (
        draft_block_size
        if draft_block_size is not None
        else int(draft_model.config.block_size)
    )
    draft_cache = draft_model.reset(model)

    b = first_bonus
    emitted = 1  # the first bonus has already been yielded by the caller

    while emitted < max_tokens:
        bs = min(block_total, max_tokens - emitted + 1)
        if bs <= 1:
            break

        draft_tokens = draft_model.draft_block(
            b, hidden, draft_cache, bs, sampler, token_dtype
        )
        mx.async_eval(draft_tokens)

        with mx.stream(generation_stream):
            verify_input = mx.concatenate(
                [mx.array([[b]], dtype=token_dtype), draft_tokens],
                axis=1,
            )
            verify_out = lm(
                verify_input,
                cache=prompt_cache,
                capture_layer_ids=target_layer_ids,
            )
            hidden = mx.concatenate(verify_out.hidden_states, axis=-1)
            # Apply repetition penalty to verify logits when processors are set
            vlogits = verify_out.logits
            if processors and tokens_ctx is not None and tokens_ctx.size > 0:
                ctx_window = tokens_ctx[-20:]
                penalized = []
                for pos in range(bs):
                    pos_l = vlogits[:, pos, :]  # [1, V]
                    for proc in processors:
                        pos_l = proc(ctx_window, pos_l)
                    penalized.append(pos_l)
                vlogits = mx.stack(penalized, axis=1)
            target_tokens = sampler(vlogits)
        mx.async_eval(target_tokens, hidden)

        # Walk
        accepted, new_tokens = _speculative_walk(
            draft_tokens, target_tokens, max_tokens - emitted
        )
        draft_model.accept_lens.append(accepted)

        # Emit
        for tok in new_tokens:
            yield tok, None
            emitted += 1
            if emitted >= max_tokens:
                return

        # Update rolling token context for next round's penalty
        if tokens_ctx is not None and new_tokens:
            tokens_ctx = mx.concat(
                [tokens_ctx, mx.array(new_tokens, dtype=tokens_ctx.dtype)]
            )

        if accepted < bs - 1:
            hidden = hidden[:, : accepted + 1, :]
        b = new_tokens[-1] if new_tokens else b

        if accepted < bs - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache, verify_out.gdn_states, accepted, bs
                )

        if emitted % 256 == 0:
            mx.clear_cache()

def _dflash_rounds_batch(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    *,
    first_bonus: mx.array,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    stop_check: Optional[Callable[[int, int], bool]] = None,
    rep_penalty_fn: Optional[Callable] = None,
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    """Batch DFlash speculative-decoding round loop (B > 1).

    Supports continuous batching: when a sequence finishes (EOS or
    max_tokens), it is filtered out of the target caches and the
    drafter cache is reinitialized for the new batch size.

    ``stop_check(seq_idx, token_id) -> bool`` is an optional callback
    that returns True to stop a sequence (e.g. EOS detection).

    Yields ``(tokens_list, None)`` where ``tokens_list[i]`` is the
    token for sequence ``i`` (or ``None`` if that sequence has nothing
    to emit this step).
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement " "rollback_speculative_cache."
        )

    B = first_bonus.shape[0]
    target_layer_ids = list(draft_model.config.target_layer_ids)
    block_total = (
        draft_block_size
        if draft_block_size is not None
        else int(draft_model.config.block_size)
    )
    draft_cache = draft_model.reset(model)

    # Per-sequence state tracked by ORIGINAL index so the caller sees
    # stable indices in the yielded token lists.
    b = first_bonus.tolist()  # active bonus tokens
    emitted = [1] * B
    finished = [False] * B
    active_idx = list(range(B))  # maps active-slot → original-index

    def _reinit_drafter():
        """Cold-restart the drafter cache after a batch change."""
        nonlocal draft_cache
        draft_cache = draft_model.make_cache()

    total_emitted = sum(emitted)

    while len(active_idx) > 0:
        remaining = [
            max(1, max_tokens - emitted[active_idx[j]] + 1)
            for j in range(len(active_idx))
        ]
        bs = min(block_total, min(remaining))
        if bs <= 1:
            break

        n_active = len(active_idx)
        b_active = [b[active_idx[j]] for j in range(n_active)]
        b_arr = mx.array(b_active, dtype=token_dtype)

        # Draft
        draft_tokens = draft_model.draft_block(
            b_arr, hidden, draft_cache, bs, sampler, token_dtype
        )
        mx.async_eval(draft_tokens)

        # Verify
        with mx.stream(generation_stream):
            verify_input = mx.concatenate([b_arr[:, None], draft_tokens], axis=1)
            verify_out = lm(
                verify_input,
                cache=prompt_cache,
                capture_layer_ids=target_layer_ids,
            )
            hidden_full = mx.concatenate(verify_out.hidden_states, axis=-1)
            vlogits = verify_out.logits
            if rep_penalty_fn is not None:
                vlogits = rep_penalty_fn(active_idx, vlogits)
            target_tokens = sampler(vlogits)
        mx.async_eval(target_tokens, hidden_full)

        # Walk (per-sequence)
        budgets = [max_tokens - emitted[active_idx[j]] for j in range(n_active)]
        accepted_list, new_tokens_list = _speculative_walk_batch(
            draft_tokens, target_tokens, budgets
        )

        min_accepted = min(accepted_list)
        accepted_arr = mx.array(accepted_list)

        if min_accepted < bs - 1:
            # Trim to min_accepted+1 (conservative but correct).
            # Using max_a would feed rejected-token hidden states to sequences
            # with fewer acceptances, biasing the next draft round.
            hidden = hidden_full[:, : min_accepted + 1, :]
        else:
            hidden = hidden_full

        for a in accepted_list:
            draft_model.accept_lens.append(a)

        # Emit (map active slots back to original indices)
        max_new = max(len(nt) for nt in new_tokens_list) if new_tokens_list else 0
        for pos in range(max_new):
            tokens_out: List[Optional[int]] = [None] * B
            for j in range(n_active):
                orig = active_idx[j]
                if pos < len(new_tokens_list[j]) and not finished[orig]:
                    tok = new_tokens_list[j][pos]
                    tokens_out[orig] = tok
                    emitted[orig] += 1
                    if emitted[orig] >= max_tokens:
                        finished[orig] = True
                    if stop_check is not None and stop_check(orig, tok):
                        finished[orig] = True
            yield tokens_out, None

        # Update bonus tokens
        for j in range(n_active):
            orig = active_idx[j]
            if new_tokens_list[j]:
                b[orig] = new_tokens_list[j][-1]

        if min_accepted < bs - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(
                    prompt_cache, verify_out.gdn_states, accepted_arr, bs
                )

        # --- Continuous batching: filter out finished sequences ---
        keep_slots = [j for j in range(n_active) if not finished[active_idx[j]]]
        if len(keep_slots) < n_active:
            if len(keep_slots) == 0:
                break
            # Filter target caches (BatchKVCache supports this)
            keep_mx = mx.array(keep_slots, dtype=mx.int32)
            for c in prompt_cache:
                if hasattr(c, "filter"):
                    c.filter(keep_mx)
            # Filter hidden
            hidden = hidden[keep_mx]
            # Update active index mapping
            active_idx = [active_idx[j] for j in keep_slots]
            # Cold-restart drafter for the new batch size
            _reinit_drafter()

        new_total = sum(emitted)
        if new_total // 256 > total_emitted // 256:
            mx.clear_cache()
        total_emitted = new_total

