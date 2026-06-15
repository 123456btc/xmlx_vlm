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

def _mtp_rounds(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    shared_kv_states: dict,
    *,
    first_bonus: int,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
) -> Generator[Tuple[int, None], None, None]:
    """Gemma 4 MTP (Single-Position Multi-Token) speculative-decoding round loop.

    Mirrors ``_dflash_rounds`` but with three differences:
    (1) the drafter consumes the target's last-layer hidden + last-layer
    shared K/V per layer-type rather than concatenated multi-layer hiddens;
    (2) ``draft_block`` is autoregressive (K small forwards) rather than a
    single masked forward; (3) ``rollback_speculative_cache`` ignores
    ``gdn_states`` (Gemma 4 has no SSM/GDN state).
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache. "
            "MTP speculative decoding currently only supports gemma4."
        )

    block_total = (
        draft_block_size
        if draft_block_size is not None
        else int(draft_model.config.block_size)
    )
    draft_model.reset(model)

    # Hidden from prefill is full prompt-length; reduce to a single slot.
    # The semantically-correct choice is the *last* prompt token's hidden:
    # the just-sampled bonus is the next-token prediction from that position,
    # so its embedding paired with that hidden is what the drafter expects.
    # (HF's literal ``[:, n_last_matches:n_last_matches+1]`` with ``n_matches=0``
    # on the first round picks position 0, which is BOS — they get away with
    # it because subsequent rounds slice into the per-call verify hidden, but
    # the round-1 acceptance is wasted. We don't replicate that quirk.)
    if hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]

    kv_offset = int(prompt_cache[0].offset)
    draft_model.set_shared_kv(shared_kv_states, kv_offset)

    b = first_bonus
    emitted = 1  # caller already yielded the first bonus

    while emitted < max_tokens:
        bs = min(block_total, max_tokens - emitted + 1)
        if bs <= 1:
            break

        draft_tokens = draft_model.draft_block(
            b, hidden, None, bs, sampler, token_dtype
        )
        mx.async_eval(draft_tokens)

        with mx.stream(generation_stream):
            verify_input = mx.concatenate(
                [mx.array([[b]], dtype=token_dtype), draft_tokens], axis=1
            )
            verify_out = lm(
                verify_input,
                cache=prompt_cache,
                return_hidden=True,
                return_shared_kv=True,
            )
            hidden_full = verify_out.hidden_states[-1]  # [B, bs, backbone]
            target_tokens = sampler(verify_out.logits)
        mx.async_eval(target_tokens, hidden_full)

        accepted, new_tokens = _speculative_walk(
            draft_tokens, target_tokens, max_tokens - emitted
        )
        draft_model.accept_lens.append(accepted)

        for tok in new_tokens:
            yield tok, None
            emitted += 1
            if emitted >= max_tokens:
                return

        # Hidden for next round: pick the slot of the newly accepted bonus.
        hidden = hidden_full[:, accepted : accepted + 1, :]
        b = new_tokens[-1] if new_tokens else b

        if accepted < bs - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(prompt_cache, None, accepted, bs)

        # Slice shared_kv_states to the post-rollback length and rebind.
        rejected = bs - (accepted + 1)
        next_shared_kv = {}
        for k, kv in verify_out.shared_kv_states.items():
            K, V = kv
            valid = K.shape[-2] - rejected
            if valid <= 0 or valid >= K.shape[-2]:
                next_shared_kv[k] = (
                    (K, V) if valid >= K.shape[-2] else (K[..., :1, :], V[..., :1, :])
                )
            else:
                next_shared_kv[k] = (K[..., :valid, :], V[..., :valid, :])
        kv_offset = int(prompt_cache[0].offset)
        draft_model.set_shared_kv(next_shared_kv, kv_offset)

        if emitted % 256 == 0:
            mx.clear_cache()

def _batch_cache_left_padding(prompt_cache: List[Any]) -> Optional[mx.array]:
    for cache_entry in prompt_cache:
        left_padding = getattr(cache_entry, "left_padding", None)
        if left_padding is not None:
            return left_padding
    return None

def _mtp_rounds_batch(
    model: nn.Module,
    draft_model: nn.Module,
    prompt_cache: List[Any],
    hidden: mx.array,
    shared_kv_states: dict,
    *,
    first_bonus: mx.array,
    max_tokens: int,
    sampler: Callable[[mx.array], mx.array],
    draft_block_size: Optional[int] = None,
    token_dtype: mx.Dtype = mx.int32,
    stop_check: Optional[Callable[[int, int], bool]] = None,
    eos_token_ids: Optional[set] = None,
) -> Generator[Tuple[List[Optional[int]], None], None, None]:
    """Batched Gemma 4 MTP round loop (B > 1).

    Mirrors ``_dflash_rounds_batch``: per-row state tracked by original
    index, continuous-batching filter on row finish. Differences vs DFlash
    batched: drafter consumes ``shared_kv_states`` (per-layer-type K/V
    snapshot) instead of multi-layer hidden capture, ``draft_block`` is
    autoregressive, and the per-round ``shared_kv`` snapshot is normalized
    back to the unbatched prefix-valid layout before each drafter rebind.
    """
    lm = model.language_model if hasattr(model, "language_model") else model
    if not hasattr(lm, "rollback_speculative_cache"):
        raise RuntimeError(
            f"{type(lm).__name__} does not implement rollback_speculative_cache."
        )

    B = first_bonus.shape[0]
    block_total = (
        draft_block_size
        if draft_block_size is not None
        else int(draft_model.config.block_size)
    )
    draft_model.reset(model)

    # First-round hidden: prefill output may have shape [B, L, H]; reduce
    # to a single slot per row (last prompt token's hidden — see comment in
    # ``_mtp_rounds`` for rationale).
    if hidden.shape[1] > 1:
        hidden = hidden[:, -1:, :]

    # Per-row state. ``positions`` is the absolute position id of each
    # row's pending bonus (= row's logical KV length). All rows start at
    # ``L_prefill`` and advance by ``accepted_i + 1`` per round.
    offset0 = prompt_cache[0].offset
    if isinstance(offset0, mx.array):
        L_prefill = int(offset0.max().item())
        positions = [int(x) for x in offset0.tolist()]
    else:
        L_prefill = int(offset0)
        positions = [L_prefill] * B
    draft_model.set_shared_kv(
        shared_kv_states,
        kv_offset=L_prefill,
        position=mx.array(positions),
        left_padding=_batch_cache_left_padding(prompt_cache),
    )

    b = first_bonus.tolist()
    emitted = [1] * B
    finished = [False] * B
    active_idx = list(range(B))

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

        # Draft (autoregressive K-step). hidden / shared_kv state was set
        # via set_shared_kv above; the drafter pulls it from there.
        draft_tokens = draft_model.draft_block(
            b_arr, hidden, None, bs, sampler, token_dtype
        )
        mx.async_eval(draft_tokens)

        # Verify
        with mx.stream(generation_stream):
            verify_input = mx.concatenate([b_arr[:, None], draft_tokens], axis=1)
            verify_out = lm(
                verify_input,
                cache=prompt_cache,
                return_hidden=True,
                return_shared_kv=True,
            )
            hidden_full = verify_out.hidden_states[-1]  # [B_active, bs, H]
            target_tokens = sampler(verify_out.logits)
        mx.async_eval(target_tokens, hidden_full)

        # Walk per-row
        budgets = [max_tokens - emitted[active_idx[j]] for j in range(n_active)]
        accepted_list, new_tokens_list = _speculative_walk_batch(
            draft_tokens, target_tokens, budgets
        )
        for a in accepted_list:
            draft_model.accept_lens.append(a)

        max_a = max(accepted_list)
        accepted_arr = mx.array(accepted_list)

        # Per-row hidden: each row picks its own accepted slot from
        # hidden_full. Build [B_active, 1, H] with row-i's hidden at
        # position accepted_list[i].
        if max_a < bs - 1 or any(a < max_a for a in accepted_list):
            row_idx = mx.arange(n_active)
            col_idx = mx.array(accepted_list)
            # gather: hidden_full[row_idx, col_idx, :] -> [B_active, H]
            hidden = hidden_full[row_idx, col_idx, :][:, None, :]
        else:
            hidden = hidden_full[:, -1:, :]

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
                    if eos_token_ids is not None and tok in eos_token_ids:
                        finished[orig] = True
                    if stop_check is not None and stop_check(orig, tok):
                        finished[orig] = True
            yield tokens_out, None

        # Update bonus tokens and per-row positions
        for j in range(n_active):
            orig = active_idx[j]
            if new_tokens_list[j]:
                b[orig] = new_tokens_list[j][-1]
            positions[orig] = positions[orig] + accepted_list[j] + 1

        # Rollback target cache (uniform trim by ``bs - max_a - 1`` plus
        # per-row tail-zero on rows that accepted less).
        if max_a < bs - 1:
            with mx.stream(generation_stream):
                lm.rollback_speculative_cache(prompt_cache, None, accepted_arr, bs)

        # Slice + tail-zero ``verify_out.shared_kv_states`` to match the
        # post-rollback target cache. ``set_shared_kv()`` will normalize the
        # resulting hybrid layout back into a prefix-valid drafter view.
        rejected_global = bs - (max_a + 1)
        next_shared_kv = {}
        for k, kv in verify_out.shared_kv_states.items():
            K, V = kv
            valid = K.shape[-2] - rejected_global
            if valid >= K.shape[-2]:
                K_next, V_next = K, V
            elif valid <= 0:
                K_next = K[..., :1, :]
                V_next = V[..., :1, :]
            else:
                K_next = K[..., :valid, :]
                V_next = V[..., :valid, :]
            # Per-row tail-zero on rows that accepted less than max_a.
            if any(a < max_a for a in accepted_list):
                # K_next/V_next shape: [B_active, H, valid, D]
                # For row i, zero positions [valid - max_a + accepted_i, valid).
                # (verify_start = valid - (max_a + 1), and tail begins at
                # verify_start + accepted_i + 1 = valid - max_a + accepted_i.)
                K_arr = mx.array(K_next)  # ensure materialized for slicing
                V_arr = mx.array(V_next)
                K_arr = mx.array(K_arr)
                V_arr = mx.array(V_arr)
                mask_rows = mx.arange(K_next.shape[-2])  # [valid]
                # Build per-row mask: True where position should be kept.
                # Shape [B_active, valid]. Row i keeps positions [0, valid - max_a + accepted_i).
                keep_lens = mx.array(
                    [valid - max_a + a for a in accepted_list], dtype=mx.int32
                )  # [B_active]
                keep_mask = mask_rows[None, :] < keep_lens[:, None]  # [B_active, valid]
                keep_f = keep_mask.astype(K_next.dtype)[:, None, :, None]  # broadcast
                K_next = K_next * keep_f
                V_next = V_next * keep_f
            next_shared_kv[k] = (K_next, V_next)

        # Continuous batching: filter finished sequences. Only safe when
        # the caches expose a .filter() method (e.g. BatchKVCache); the
        # plain KVCache / RotatingKVCache do not, so we keep all rows
        # in the batch and just stop emitting for finished rows. End the
        # round-loop when every row has finished.
        cache_filterable = all(hasattr(c, "filter") for c in prompt_cache)
        if all(finished[active_idx[j]] for j in range(n_active)):
            break
        if cache_filterable:
            keep_slots = [j for j in range(n_active) if not finished[active_idx[j]]]
            if len(keep_slots) < n_active:
                keep_mx = mx.array(keep_slots, dtype=mx.int32)
                for c in prompt_cache:
                    c.filter(keep_mx)
                hidden = hidden[keep_mx]
                for k in next_shared_kv:
                    K_next, V_next = next_shared_kv[k]
                    next_shared_kv[k] = (K_next[keep_mx], V_next[keep_mx])
                active_idx = [active_idx[j] for j in keep_slots]

        # Re-bind drafter with new shared_kv and per-row positions.
        positions_active = [positions[active_idx[j]] for j in range(len(active_idx))]
        offset0 = prompt_cache[0].offset
        new_kv_offset = (
            int(offset0.max().item()) if isinstance(offset0, mx.array) else int(offset0)
        )
        draft_model.set_shared_kv(
            next_shared_kv,
            kv_offset=new_kv_offset,
            position=mx.array(positions_active),
            left_padding=_batch_cache_left_padding(prompt_cache),
        )

        if sum(emitted) % 256 == 0:
            mx.clear_cache()

