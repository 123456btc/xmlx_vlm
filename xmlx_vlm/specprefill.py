# SPDX-License-Identifier: Apache-2.0
"""SpecPrefill: Attention-based sparse prefill for MLX.

Full pipeline for reducing TTFT on long prompts:
  Step 1 (score_tokens): Use a small draft model to identify important tokens
  Step 2 (sparse_prefill): Prefill target model with only selected tokens,
                           preserving original positional encoding via manual RoPE

Reference: arxiv.org/abs/2502.02789 (SpecPrefill: Speculative Prefilling)
"""

import math

import mlx.core as mx

from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler


# ===========================================================================
# Step 1: Token importance scoring (draft model)
# ===========================================================================

class _AttentionCapture:
    """Wrapper that captures post-RoPE query vectors and delegates to original."""

    def __init__(self, original, buf_idx, query_buffer, query_extractor=None):
        self._original = original
        self._buf_idx = buf_idx
        self._query_buffer = query_buffer
        self._query_extractor = query_extractor or _qwen35_extract_queries

    def __call__(self, x, mask=None, cache=None):
        queries = self._query_extractor(self._original, x, cache)
        self._query_buffer[self._buf_idx].append(queries)
        return self._original(x, mask=mask, cache=cache)

    def __getattr__(self, name):
        return getattr(self._original, name)


def _qwen35_extract_queries(attn, x, cache=None):
    """Extract post-RoPE queries from Qwen3.5 attention (gate split + q_norm)."""
    B, L, D = x.shape
    q_out = attn.q_proj(x)
    queries, _gate = mx.split(
        q_out.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1
    )
    queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
    else:
        queries = attn.rope(queries)
    return queries


def _llama_extract_queries(attn, x, cache=None):
    """Extract post-RoPE queries from standard transformer attention."""
    B, L, D = x.shape
    n_heads = getattr(
        attn,
        "num_attention_heads",
        getattr(attn, "n_heads", getattr(attn, "num_heads", None)),
    )
    queries = attn.q_proj(x)
    queries = queries.reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = attn.rope(queries, offset=cache.offset)
    else:
        queries = attn.rope(queries)
    return queries


def _nemotron_h_extract_queries(attn, x, cache=None):
    """Extract queries from Nemotron-H attention (no RoPE, no gate, no q_norm)."""
    B, L, D = x.shape
    queries = attn.q_proj(x).reshape(B, L, attn.num_heads, -1).transpose(0, 2, 1, 3)
    return queries


def _patch_attention_for_capture(model, query_buffer, query_extractor=None):
    """Replace attention modules on full-attention layers with capture wrappers."""
    originals = []
    attn_indices = []
    for layer_idx, layer in _find_attention_layers(model):
        buf_idx = len(attn_indices)
        attn_indices.append(layer_idx)
        orig = _get_attn_module(layer)
        _set_attn_module(
            layer,
            _AttentionCapture(
                orig, buf_idx, query_buffer, query_extractor=query_extractor
            ),
        )
        originals.append((layer_idx, orig))
    return originals, attn_indices


def _unpatch_attention_capture(model, originals):
    """Restore original attention modules after capture."""
    for layer_idx, orig in originals:
        _set_attn_module(model.layers[layer_idx], orig)


def _prefill_draft(model, tokens, cache, step_size=2048, cancel_check=None):
    """Prefill prompt tokens into cache. Returns logits from last token."""
    prompt = mx.array(tokens) if not isinstance(tokens, mx.array) else tokens
    n = len(tokens)
    processed = 0
    while n - processed > 1:
        if cancel_check is not None:
            cancel_check()
        chunk = min(step_size, n - processed - 1)
        model(prompt[processed : processed + chunk][None], cache=cache)
        mx.eval([c.state for c in cache])
        processed += chunk
        mx.clear_cache()
    if cancel_check is not None:
        cancel_check()
    logits = model(prompt[processed:][None], cache=cache)
    mx.eval(logits)
    return logits


def _lookahead_decode(
    model,
    first_logits,
    cache,
    n_steps,
    temp=0.6,
    top_p=0.95,
    cancel_check=None,
):
    """Run n_steps autoregressive decode, returning generated token ids."""
    sampler = make_sampler(temp=temp, top_p=top_p)
    if cancel_check is not None:
        cancel_check()
    y = sampler(first_logits[:, -1, :])
    mx.eval(y)
    generated = [y.item()]
    for _ in range(n_steps):
        if cancel_check is not None:
            cancel_check()
        logits = model(y.reshape(1, -1), cache=cache)
        y = sampler(logits[:, -1, :])
        mx.eval(y)
        generated.append(y.item())
    return generated


def _avg_pool1d(x, kernel_size):
    """1D average pooling along last axis via prefix-sum."""
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    padded = mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)])
    zeros = mx.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    prefix = mx.concatenate([zeros, mx.cumsum(padded, axis=-1)], axis=-1)
    return (prefix[..., kernel_size:] - prefix[..., :-kernel_size]) / kernel_size


def _compute_importance(
    query_buffer, attn_caches, n_prompt, n_attn_heads, n_kv_heads, pool_kernel=13
):
    """Compute per-token importance from captured queries and cached keys."""
    heads_per_group = n_attn_heads // n_kv_heads
    all_scores = []

    for layer_i, captures in enumerate(query_buffer):
        if not captures:
            continue
        cache = attn_caches[layer_i]
        prompt_keys = cache.keys[..., :n_prompt, :]
        if prompt_keys.shape[-2] < n_prompt:
            continue
        head_dim = prompt_keys.shape[-1]
        q_stack = mx.concatenate(captures, axis=2)
        if heads_per_group > 1:
            expanded_keys = mx.repeat(prompt_keys, heads_per_group, axis=1)
        else:
            expanded_keys = prompt_keys
        scale = head_dim**-0.5
        scores = (q_stack @ expanded_keys.transpose(0, 1, 3, 2)) * scale
        weights = mx.softmax(scores.astype(mx.float32), axis=-1)
        all_scores.append(weights.squeeze(0))

    if not all_scores:
        raise RuntimeError("No attention scores captured — check model/patching")

    combined = mx.concatenate(all_scores, axis=0)
    if pool_kernel and pool_kernel > 1:
        combined = _avg_pool1d(combined, pool_kernel)
    max_scores = mx.max(combined, axis=0)
    importance = mx.mean(max_scores, axis=0)
    return importance


def score_tokens(
    model,
    tokens,
    n_lookahead=8,
    pool_kernel=13,
    temp=0.6,
    top_p=0.95,
    prefill_step_size=2048,
    query_extractor=None,
    cancel_check=None,
):
    """Score token importance using attention-based analysis on a draft model."""
    if isinstance(tokens, mx.array):
        tokens = tokens.tolist()
    n_prompt = len(tokens)

    attn_layers = _find_attention_layers(model)
    n_attn_layers = len(attn_layers)
    attn_obj = _get_attn_module(attn_layers[0][1])
    n_attn_heads = getattr(
        attn_obj,
        "num_attention_heads",
        getattr(attn_obj, "n_heads", getattr(attn_obj, "num_heads", None)),
    )
    n_kv_heads = getattr(
        attn_obj, "num_key_value_heads", getattr(attn_obj, "n_kv_heads", None)
    )

    if query_extractor is None:
        model_type = getattr(getattr(model, "config", None), "model_type", "")
        _EXTRACTOR_REGISTRY = {
            "qwen3_5": _qwen35_extract_queries,
            "qwen3_5_moe": _qwen35_extract_queries,
            "qwen3_vl": _qwen35_extract_queries,
            "qwen3_vl_moe": _qwen35_extract_queries,
            "nemotron_h": _nemotron_h_extract_queries,
        }
        query_extractor = _EXTRACTOR_REGISTRY.get(model_type)
        if query_extractor is None:
            if _get_rope(attn_obj) is not None:
                query_extractor = _llama_extract_queries
            else:
                query_extractor = _nemotron_h_extract_queries

    cache = make_prompt_cache(model)
    logits = _prefill_draft(
        model,
        tokens,
        cache,
        step_size=prefill_step_size,
        cancel_check=cancel_check,
    )

    query_buffer = [[] for _ in range(n_attn_layers)]
    patches, attn_indices = _patch_attention_for_capture(
        model, query_buffer, query_extractor=query_extractor
    )
    try:
        _lookahead_decode(
            model,
            logits,
            cache,
            n_lookahead,
            temp=temp,
            top_p=top_p,
            cancel_check=cancel_check,
        )
        mx.eval(query_buffer)
    finally:
        _unpatch_attention_capture(model, patches)

    layer_to_cache = _build_layer_to_cache_map(model)
    attn_caches = [cache[layer_to_cache[i]] for i in attn_indices]
    if cancel_check is not None:
        cancel_check()
    importance = _compute_importance(
        query_buffer,
        attn_caches,
        n_prompt,
        n_attn_heads,
        n_kv_heads,
        pool_kernel=pool_kernel if pool_kernel > 0 else None,
    )
    mx.eval(importance)

    del cache, logits, query_buffer, attn_caches
    mx.clear_cache()

    return importance


def select_chunks(importance, keep_pct=0.3, chunk_size=32):
    """Select top-k% token chunks by average importance."""
    M = importance.shape[0]
    if keep_pct >= 1.0:
        return mx.arange(M)

    n_chunks = math.ceil(M / chunk_size)
    keep_n = max(1, math.ceil(n_chunks * keep_pct))

    chunk_scores = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, M)
        chunk_scores.append(mx.mean(importance[start:end]).item())

    top_chunks = sorted(range(n_chunks), key=lambda i: chunk_scores[i], reverse=True)[
        :keep_n
    ]
    top_chunks.sort()

    indices = []
    for ci in top_chunks:
        start = ci * chunk_size
        end = min(start + chunk_size, M)
        indices.extend(range(start, end))

    return mx.array(indices)


# ===========================================================================
# Step 2: Sparse prefill with non-contiguous position IDs (target model)
# ===========================================================================

def manual_rope(x, positions, dims, base=10000.0, scale=1.0):
    """Apply RoPE at arbitrary (non-contiguous) positions."""
    half = dims // 2
    inv_freq = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
    scaled_pos = positions.astype(mx.float32) / scale
    angles = scaled_pos[:, None] * inv_freq[None, :]
    cos_a = mx.cos(angles)[None, None, :, :]
    sin_a = mx.sin(angles)[None, None, :, :]
    x_rot, x_pass = x[..., :dims], x[..., dims:]
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate(
        [x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], axis=-1
    )
    return mx.concatenate([rotated, x_pass], axis=-1)


def manual_rope_with_freqs(x, positions, dims, freqs, pre_scale=1.0):
    """Apply RoPE at arbitrary positions using pre-computed frequencies."""
    half = dims // 2
    inv_freq = (1.0 / freqs).astype(mx.float32)
    angles = positions[:, None].astype(mx.float32) * inv_freq[None, :]
    cos_a = mx.cos(angles)[None, None, :, :]
    sin_a = mx.sin(angles)[None, None, :, :]
    x_rot, x_pass = x[..., :dims], x[..., dims:]
    if pre_scale != 1.0:
        x_rot = pre_scale * x_rot
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = mx.concatenate(
        [x1 * cos_a - x2 * sin_a, x1 * sin_a + x2 * cos_a], axis=-1
    )
    return mx.concatenate([rotated, x_pass], axis=-1)


class _PositionMappedRoPE:
    """Wraps a RoPE module to apply rotation at non-contiguous positions."""

    def __init__(self, original_rope, all_positions, cache_start=0):
        self._original = original_rope
        self._all_positions = all_positions
        self._cache_start = cache_start
        self._has_custom_freqs = hasattr(original_rope, "_freqs")

        if self._has_custom_freqs:
            self._freqs = original_rope._freqs
            self._dims = _get_dims(original_rope)
            self._pre_scale = _get_pre_scale(original_rope)
        else:
            self._dims = original_rope.dims
            self._base = original_rope.base
            self._scale = original_rope.scale

    def __call__(self, x, offset=0):
        L = x.shape[2]
        idx = offset - self._cache_start
        positions = self._all_positions[idx : idx + L]
        if self._has_custom_freqs:
            return manual_rope_with_freqs(
                x, positions, self._dims, self._freqs, pre_scale=self._pre_scale
            )
        return manual_rope(x, positions, self._dims, base=self._base, scale=self._scale)


class _OffsetAdjustedRoPE:
    """Wraps a RoPE module to add a constant offset for decode after sparse prefill."""

    def __init__(self, original_rope, adjustment):
        self._original = original_rope
        self._adjustment = adjustment

    def __call__(self, x, offset=0):
        return self._original(x, offset=offset + self._adjustment)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_dims(rope_module):
    """Extract rotary dimensions from any RoPE variant."""
    for attr in ("_dims", "dim", "dims"):
        if hasattr(rope_module, attr):
            return getattr(rope_module, attr)
    raise ValueError(f"Cannot determine dims from {type(rope_module)}")


def _get_pre_scale(rope_module):
    """Extract pre-scale factor from custom RoPE variants (SuScaled, Yarn)."""
    if hasattr(rope_module, "mscale"):
        return rope_module.mscale
    if hasattr(rope_module, "_scale") and hasattr(rope_module, "dim"):
        return rope_module._scale
    return 1.0


def _find_attention_layers(model):
    """Find all full-attention layers across architectures."""
    results = []
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, "self_attn"):
            results.append((idx, layer))
        elif getattr(layer, "block_type", None) == "*":
            results.append((idx, layer))
    return results


def _get_attn_module(layer):
    """Get the attention module from a layer (self_attn or mixer)."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if getattr(layer, "block_type", None) == "*":
        return layer.mixer
    return None


def _get_rope(attn):
    """Get the RoPE module from an attention layer, or None."""
    return getattr(attn, "rope", None) or getattr(attn, "rotary_emb", None)


def _set_rope(attn, rope_module):
    """Set the RoPE module on an attention layer."""
    if hasattr(attn, "rope"):
        attn.rope = rope_module
    elif hasattr(attn, "rotary_emb"):
        attn.rotary_emb = rope_module


def _set_attn_module(layer, module):
    """Set the attention module on a layer (self_attn or mixer)."""
    if hasattr(layer, "self_attn"):
        layer.self_attn = module
    elif getattr(layer, "block_type", None) == "*":
        layer.mixer = module


def _build_layer_to_cache_map(model):
    """Build mapping from model layer index to cache index."""
    has_block_type = any(hasattr(layer, "block_type") for layer in model.layers)
    if not has_block_type:
        return {i: i for i in range(len(model.layers))}

    layer_to_cache = {}
    cache_idx = 0
    for layer_idx, layer in enumerate(model.layers):
        bt = getattr(layer, "block_type", None)
        if bt in ("M", "*"):
            layer_to_cache[layer_idx] = cache_idx
            cache_idx += 1
    return layer_to_cache


# ---------------------------------------------------------------------------
# Core API — sparse prefill
# ---------------------------------------------------------------------------


def sparse_prefill(
    model,
    tokens,
    selected_indices,
    cache,
    step_size=2048,
    position_offset=0,
    cancel_check=None,
):
    """Prefill the model cache with selected tokens at their original positions.

    Runs the model forward on only the selected tokens while preserving their
    original positional encoding via manual RoPE. After this call, the cache
    contains KV entries with correct RoPE positions, and attention layers have
    _OffsetAdjustedRoPE installed for correct decode positioning.

    Args:
        model: Language model with .layers property.
        tokens: (M,) all prompt token IDs.
        selected_indices: (N,) sorted indices into tokens to keep.
        cache: list of KVCache/ArraysCache from make_prompt_cache().
        step_size: chunk size for processing.
        position_offset: added to selected_indices for RoPE positions.

    Returns:
        logits: (1, 1, vocab_size) from the last selected token.

    Side effects:
        - Populates cache with KV for selected tokens.
        - Installs _OffsetAdjustedRoPE on attention layers for decode.
        - Call cleanup_rope(model) after generation to restore original RoPE.
    """
    if not isinstance(tokens, mx.array):
        tokens = mx.array(tokens)
    if not isinstance(selected_indices, mx.array):
        selected_indices = mx.array(selected_indices)

    M = tokens.shape[0]

    max_rotating_size = 0
    for c in cache:
        if type(c).__name__ == "RotatingKVCache":
            max_rotating_size = max(max_rotating_size, getattr(c, "max_size", 0))
    if max_rotating_size > 0 and M > max_rotating_size:
        tail_start = max(0, M - max_rotating_size)
        tail_indices = set(range(tail_start, M))
        existing = set(selected_indices.tolist())
        merged = sorted(existing | tail_indices)
        selected_indices = mx.array(merged)

    selected_positions = selected_indices.astype(mx.int32) + position_offset
    selected_tokens = tokens[selected_indices]
    N = selected_tokens.shape[0]

    attn_layers = _find_attention_layers(model)
    layer_to_cache = _build_layer_to_cache_map(model)
    first_attn_layer_idx = attn_layers[0][0]
    first_attn_cache_idx = layer_to_cache[first_attn_layer_idx]
    cache_start = (
        cache[first_attn_cache_idx].offset
        if hasattr(cache[first_attn_cache_idx], "offset")
        else 0
    )

    first_attn = _get_attn_module(attn_layers[0][1])
    has_rope = _get_rope(first_attn) is not None

    original_ropes = {}
    if has_rope:
        for layer_idx, layer in attn_layers:
            attn = _get_attn_module(layer)
            rope = _get_rope(attn)
            original_ropes[layer_idx] = (attn, rope)
            _set_rope(
                attn,
                _PositionMappedRoPE(rope, selected_positions, cache_start=cache_start),
            )

    try:
        prompt = selected_tokens
        n = int(N)
        processed = 0

        while n - processed > 1:
            if cancel_check is not None:
                cancel_check()
            chunk = min(step_size, n - processed - 1)
            model(prompt[processed : processed + chunk][None], cache=cache)
            mx.eval([c.state for c in cache])
            processed += chunk
            mx.clear_cache()

        if cancel_check is not None:
            cancel_check()
        logits = model(prompt[processed:][None], cache=cache)
        mx.eval(logits)

    finally:
        total_prompt_len = position_offset + M
        final_cache_offset = cache_start + N
        adjustment = int(total_prompt_len) - int(final_cache_offset)
        if has_rope:
            for layer_idx, layer in attn_layers:
                attn, original = original_ropes[layer_idx]
                if adjustment > 0:
                    _set_rope(attn, _OffsetAdjustedRoPE(original, adjustment))
                else:
                    _set_rope(attn, original)

    return logits


def cleanup_rope(model):
    """Restore original RoPE on all attention layers.

    Call this after generation is complete to remove _OffsetAdjustedRoPE
    wrappers installed by sparse_prefill().
    """
    for _, layer in _find_attention_layers(model):
        attn = _get_attn_module(layer)
        if attn is None:
            continue
        rope = _get_rope(attn)
        if rope is None:
            continue
        if isinstance(rope, (_OffsetAdjustedRoPE, _PositionMappedRoPE)):
            _set_rope(attn, rope._original)
