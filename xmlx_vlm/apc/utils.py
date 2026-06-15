from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import mlx.core as mx
import numpy as np

from .types import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_NUM_BLOCKS,
    SEED_PARENT_HASH,
    APCBlock,
    APCExactCacheEntry,
    _hash_use_sha256,
)
from .manager import APCManager
from .disk_store import DiskBlockStore

logger = logging.getLogger("xmlx_vlm.apc")

def _copy_mlx_array(x: mx.array) -> mx.array:
    """Materialize ``x`` into a fresh MLX-owned contiguous buffer."""
    return mx.contiguous(mx.array(x, dtype=x.dtype))


def _pad_kv_for_capacity(
    keys: mx.array,
    values: mx.array,
    *,
    offset: int,
    min_capacity_tokens: Optional[int],
    step: int,
) -> Tuple[mx.array, mx.array]:
    if min_capacity_tokens is None or min_capacity_tokens <= offset:
        return keys, values
    capacity = int(min_capacity_tokens)
    if step > 0:
        capacity = ((capacity + step - 1) // step) * step
    if capacity <= keys.shape[2]:
        return keys, values
    pad_tokens = capacity - keys.shape[2]
    k_shape = (*keys.shape[:2], pad_tokens, keys.shape[3])
    v_shape = (*values.shape[:2], pad_tokens, values.shape[3])
    keys = mx.concatenate([keys, mx.zeros(k_shape, dtype=keys.dtype)], axis=2)
    values = mx.concatenate([values, mx.zeros(v_shape, dtype=values.dtype)], axis=2)
    return keys, values


def _clone_cache_entry_for_apc(
    c: Any,
    *,
    min_capacity_tokens: Optional[int],
    eval_targets: List[mx.array],
) -> Optional[Any]:
    """Deep-copy one prompt-cache entry, preserving its concrete cache kind."""
    from mlx_lm.models import cache as lm_cache

    if isinstance(c, lm_cache.KVCache):
        out = type(c)()
        off = int(getattr(c, "offset", 0) or 0)
        if c.keys is not None and c.values is not None and off > 0:
            keys = _copy_mlx_array(c.keys[..., :off, :])
            values = _copy_mlx_array(c.values[..., :off, :])
            step = int(getattr(c, "step", getattr(type(c), "step", 256)) or 0)
            keys, values = _pad_kv_for_capacity(
                keys,
                values,
                offset=off,
                min_capacity_tokens=min_capacity_tokens,
                step=step,
            )
            out.keys = keys
            out.values = values
            out.offset = off
            eval_targets.extend([keys, values])
        return out

    if isinstance(c, lm_cache.RotatingKVCache):
        out = type(c)(
            max_size=int(getattr(c, "max_size")),
            keep=int(getattr(c, "keep", 0)),
        )
        out.offset = int(getattr(c, "offset", 0) or 0)
        out._idx = int(getattr(c, "_idx", 0) or 0)
        if c.keys is not None and c.values is not None:
            out.keys = _copy_mlx_array(c.keys)
            out.values = _copy_mlx_array(c.values)
            eval_targets.extend([out.keys, out.values])
        return out

    if isinstance(c, lm_cache.ChunkedKVCache):
        out = type(c)(chunk_size=int(getattr(c, "chunk_size")))
        out.offset = int(getattr(c, "offset", 0) or 0)
        out.start_position = int(getattr(c, "start_position", 0) or 0)
        if c.keys is not None and c.values is not None:
            out.keys = _copy_mlx_array(c.keys)
            out.values = _copy_mlx_array(c.values)
            eval_targets.extend([out.keys, out.values])
        return out

    if isinstance(c, lm_cache.ArraysCache):
        out = lm_cache.ArraysCache(len(c.cache))
        out.cache = []
        for state in c.cache:
            if state is None:
                out.cache.append(None)
                continue
            copied = _copy_mlx_array(state)
            out.cache.append(copied)
            eval_targets.append(copied)
        if c.left_padding is not None:
            out.left_padding = _copy_mlx_array(c.left_padding)
            eval_targets.append(out.left_padding)
        if c.lengths is not None:
            out.lengths = _copy_mlx_array(c.lengths)
            eval_targets.append(out.lengths)
        return out

    if isinstance(c, lm_cache.CacheList):
        copied = [
            _clone_cache_entry_for_apc(
                sub_c,
                min_capacity_tokens=min_capacity_tokens,
                eval_targets=eval_targets,
            )
            for sub_c in c.caches
        ]
        if any(sub_c is None for sub_c in copied):
            return None
        return lm_cache.CacheList(*copied)

    if isinstance(c, tuple):
        copied = [
            _clone_cache_entry_for_apc(
                sub_c,
                min_capacity_tokens=min_capacity_tokens,
                eval_targets=eval_targets,
            )
            for sub_c in c
        ]
        if any(sub_c is None for sub_c in copied):
            return None
        return tuple(copied)

    # Batch caches (used by continuous-batching path and hybrid SSM models)
    if isinstance(c, lm_cache.BatchKVCache):
        out = type(c)(c.left_padding.tolist())
        out._idx = int(getattr(c, "_idx", 0) or 0)
        out._right_padding = getattr(c, "_right_padding", None)
        if c.keys is not None:
            out.keys = _copy_mlx_array(c.keys)
            eval_targets.append(out.keys)
        if c.values is not None:
            out.values = _copy_mlx_array(c.values)
            eval_targets.append(out.values)
        if c.offset is not None:
            out.offset = _copy_mlx_array(c.offset)
            eval_targets.append(out.offset)
        if c.left_padding is not None:
            out.left_padding = _copy_mlx_array(c.left_padding)
            eval_targets.append(out.left_padding)
        return out

    if isinstance(c, lm_cache.BatchRotatingKVCache):
        out = type(c)(
            max_size=int(getattr(c, "max_size", 0)),
            left_padding=c.left_padding.tolist(),
        )
        out._idx = int(getattr(c, "_idx", 0) or 0)
        out._offset = int(getattr(c, "_offset", 0) or 0)
        out.rotated = bool(getattr(c, "rotated", False))
        if c.keys is not None:
            out.keys = _copy_mlx_array(c.keys)
            eval_targets.append(out.keys)
        if c.values is not None:
            out.values = _copy_mlx_array(c.values)
            eval_targets.append(out.values)
        if c.offset is not None:
            out.offset = _copy_mlx_array(c.offset)
            eval_targets.append(out.offset)
        if c.left_padding is not None:
            out.left_padding = _copy_mlx_array(c.left_padding)
            eval_targets.append(out.left_padding)
        if getattr(c, "_lengths", None) is not None:
            out._lengths = _copy_mlx_array(c._lengths)
            eval_targets.append(out._lengths)
        return out

    if isinstance(c, lm_cache.ConcatenateKVCache):
        out = type(c)()
        if c.keys is not None:
            out.keys = _copy_mlx_array(c.keys)
            eval_targets.append(out.keys)
        if c.values is not None:
            out.values = _copy_mlx_array(c.values)
            eval_targets.append(out.values)
        out.offset = int(getattr(c, "offset", 0) or 0)
        return out

    # Generic fallback for _BaseCache subclasses with state/meta_state.
    # Used for custom SSM caches (DeltaNet, Mamba, etc.) that store
    # recurrent state alongside KV tensors.
    if hasattr(c, "state") and hasattr(c, "meta_state"):
        try:
            state = c.state
            meta = c.meta_state
            out = type(c).__new__(type(c))
            out.state = state
            out.meta_state = meta
            # Deep-copy any MLX arrays that were set by reference
            for attr in dir(c):
                if attr.startswith("_"):
                    continue
                val = getattr(c, attr, None)
                if val is None or attr in ("state", "meta_state"):
                    continue
                if isinstance(val, mx.array):
                    copied = _copy_mlx_array(val)
                    setattr(out, attr, copied)
                    eval_targets.append(copied)
                elif isinstance(val, (list, tuple)) and val:
                    new_list = []
                    for item in val:
                        if isinstance(item, mx.array):
                            copied = _copy_mlx_array(item)
                            new_list.append(copied)
                            eval_targets.append(copied)
                        else:
                            new_list.append(item)
                    setattr(out, attr, type(val)(new_list))
            return out
        except Exception:
            pass

    return None


def _clone_prompt_cache_for_apc(
    prompt_cache: Sequence[Any],
    *,
    min_capacity_tokens: Optional[int] = None,
) -> Optional[List[Any]]:
    eval_targets: List[mx.array] = []
    out: List[Any] = []
    for c in prompt_cache:
        copied = _clone_cache_entry_for_apc(
            c,
            min_capacity_tokens=min_capacity_tokens,
            eval_targets=eval_targets,
        )
        if copied is None:
            return None
        out.append(copied)
    if eval_targets:
        mx.eval(eval_targets)
    return out


def _clone_layer_major_kv_cache_for_apc(
    layer_keys: Sequence[mx.array],
    layer_values: Sequence[mx.array],
    prefix_len: int,
) -> Optional[List[Any]]:
    """Deep-copy layer-major K/V tensors into compact ``KVCache`` entries."""
    from mlx_lm.models.cache import KVCache

    if prefix_len <= 0 or len(layer_keys) != len(layer_values):
        return None
    eval_targets: List[mx.array] = []
    out: List[Any] = []
    for k, v in zip(layer_keys, layer_values):
        c = KVCache()
        c.keys = _copy_mlx_array(k[..., :prefix_len, :])
        c.values = _copy_mlx_array(v[..., :prefix_len, :])
        c.offset = prefix_len
        eval_targets.extend([c.keys, c.values])
        out.append(c)
    if eval_targets:
        mx.eval(eval_targets)
    return out


def multimodal_token_ids_from_config(config: Any) -> set[int]:
    """Return token IDs that represent media placeholders in a prompt."""
    ids: set[int] = set()
    for attr in (
        "image_token_id",
        "image_token_index",
        "video_token_id",
        "video_token_index",
    ):
        token_id = getattr(config, attr, None)
        if token_id is not None:
            ids.add(int(token_id))
    return ids


def media_token_spans(
    token_ids: Sequence[int],
    media_token_ids: Iterable[int],
) -> Tuple[Tuple[int, int], ...]:
    """Return contiguous media-token spans as half-open ``(start, end)`` ranges."""
    media_ids = {int(token_id) for token_id in media_token_ids}
    if not media_ids:
        return ()

    spans: list[tuple[int, int]] = []
    start: Optional[int] = None
    for idx, token_id in enumerate(token_ids):
        if int(token_id) in media_ids:
            if start is None:
                start = idx
        elif start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, len(token_ids)))
    return tuple(spans)


def media_safe_prefix_min(
    token_ids: Sequence[int],
    media_token_ids: Iterable[int],
) -> int:
    """Minimum prefix length that leaves a text-only suffix.

    APC restore paths consume full prompt-level image/video feature tensors. Until
    media-feature slicing is model-aware, restored prefixes must include every
    media placeholder token so the suffix can be embedded as text-only.
    """
    spans = media_token_spans(token_ids, media_token_ids)
    if not spans:
        return 0
    return max(end for _start, end in spans)


def prefix_leaves_text_only_suffix(
    token_ids: Sequence[int],
    prefix_len: int,
    media_token_ids: Iterable[int],
) -> bool:
    return int(prefix_len) >= media_safe_prefix_min(token_ids, media_token_ids)


def prefix_contains_media_tokens(
    token_ids: Sequence[int],
    prefix_len: int,
    media_token_ids: Iterable[int],
) -> bool:
    """Return whether the prefix itself contains media placeholder tokens."""
    media_ids = {int(token_id) for token_id in media_token_ids}
    if not media_ids or prefix_len <= 0:
        return False
    prefix_end = min(int(prefix_len), len(token_ids))
    return any(int(token_id) in media_ids for token_id in token_ids[:prefix_end])


def adjust_prefix_to_text_suffix_boundary(
    token_ids: Sequence[int],
    desired_prefix_len: int,
    media_token_ids: Iterable[int],
    *,
    max_prefix_tokens: Optional[int] = None,
) -> int:
    """Move an APC prefix forward until its suffix contains no media tokens.

    Returns ``0`` when no useful safe prefix exists within ``max_prefix_tokens``.
    """
    max_len = (
        len(token_ids) - 1 if max_prefix_tokens is None else int(max_prefix_tokens)
    )
    if max_len <= 0:
        return 0
    desired = max(1, int(desired_prefix_len))
    prefix_len = max(desired, media_safe_prefix_min(token_ids, media_token_ids))
    if prefix_len > max_len:
        return 0
    return prefix_len


def _cache_entry_supports_exact_apc(c: Any) -> bool:
    from mlx_lm.models import cache as lm_cache

    if isinstance(
        c,
        (
            lm_cache.KVCache,
            lm_cache.RotatingKVCache,
            lm_cache.ChunkedKVCache,
            lm_cache.ArraysCache,
            lm_cache.BatchKVCache,
            lm_cache.BatchRotatingKVCache,
            lm_cache.ConcatenateKVCache,
        ),
    ):
        return True
    if isinstance(c, lm_cache.CacheList):
        return all(_cache_entry_supports_exact_apc(sub_c) for sub_c in c.caches)
    if isinstance(c, tuple):
        return all(_cache_entry_supports_exact_apc(sub_c) for sub_c in c)
    # Generic fallback: any _BaseCache subclass with state/meta_state
    if hasattr(c, "state") and hasattr(c, "meta_state"):
        return True
    return False


def _cache_entry_supports_block_apc(c: Any) -> bool:
    from mlx_lm.models import cache as lm_cache

    return isinstance(c, lm_cache.KVCache)



def make_warm_kv_cache(
    matched_blocks: List[APCBlock],
    min_capacity_tokens: Optional[int] = None,
) -> List[Any]:
    """Stitch matched blocks into per-layer ``KVCache`` instances pre-filled
    with the cached prefix's K/V state. Used by the single-stream
    ``stream_generate`` path.
    """
    from mlx_lm.models.cache import KVCache

    if not matched_blocks:
        return []
    num_layers = len(matched_blocks[0].keys)
    out: List[Any] = []
    prefix_len = sum(b.keys[0].shape[-2] for b in matched_blocks)
    step_probe = KVCache()
    kv_step = int(getattr(step_probe, "step", getattr(KVCache, "step", 256)))
    capacity = prefix_len
    if min_capacity_tokens is not None:
        capacity = max(prefix_len, int(min_capacity_tokens))
        if capacity > prefix_len and kv_step > 0:
            capacity = ((capacity + kv_step - 1) // kv_step) * kv_step
    for layer_idx in range(num_layers):
        ks = [b.keys[layer_idx] for b in matched_blocks]
        vs = [b.values[layer_idx] for b in matched_blocks]
        merged_k = mx.concatenate(ks, axis=2)
        merged_v = mx.concatenate(vs, axis=2)
        if capacity > prefix_len:
            pad_tokens = capacity - prefix_len
            k_pad_shape = (*merged_k.shape[:2], pad_tokens, merged_k.shape[3])
            v_pad_shape = (*merged_v.shape[:2], pad_tokens, merged_v.shape[3])
            merged_k = mx.concatenate(
                [merged_k, mx.zeros(k_pad_shape, dtype=merged_k.dtype)], axis=2
            )
            merged_v = mx.concatenate(
                [merged_v, mx.zeros(v_pad_shape, dtype=merged_v.dtype)], axis=2
            )
        c = step_probe if layer_idx == 0 else KVCache()
        c.keys = merged_k
        c.values = merged_v
        c.offset = prefix_len
        out.append(c)
    return out


def make_warm_kv_cache_from_layers(
    layer_keys: List[mx.array],
    layer_values: List[mx.array],
    prefix_len: int,
) -> List[Any]:
    """Build ``KVCache`` objects from already-concatenated disk-restored K/V."""
    from mlx_lm.models.cache import KVCache

    out: List[Any] = []
    for k, v in zip(layer_keys, layer_values):
        c = KVCache()
        c.keys = k
        c.values = v
        c.offset = prefix_len
        out.append(c)
    mx.clear_cache()
    return out


def make_warm_batch_kv_cache(
    matched_blocks: List[APCBlock],
) -> List[Any]:
    """Stitch matched blocks into per-layer single-row ``BatchKVCache``
    instances pre-filled with the cached prefix's K/V state. Used by the
    batched continuous-batching path; the resulting cache list can be
    ``extend()``-ed into a running batch.
    """
    from mlx_lm.models.cache import BatchKVCache

    if not matched_blocks:
        return []
    num_layers = len(matched_blocks[0].keys)
    prefix_len = sum(b.keys[0].shape[-2] for b in matched_blocks)
    out: List[Any] = []
    for layer_idx in range(num_layers):
        ks = [b.keys[layer_idx] for b in matched_blocks]
        vs = [b.values[layer_idx] for b in matched_blocks]
        merged_k = mx.concatenate(ks, axis=2)  # [1, H, prefix_len, D]
        merged_v = mx.concatenate(vs, axis=2)
        c = BatchKVCache(left_padding=[0])
        # state setter: (keys, values, offset, left_padding) → also sets _idx
        c.state = (
            merged_k,
            merged_v,
            mx.array([prefix_len]),
            mx.array([0]),
        )
        out.append(c)
    return out


def make_warm_batch_kv_cache_multi(
    picks: List[Optional[dict]],
    num_layers: int,
) -> Tuple[List[Any], int]:
    """Build a multi-row ``BatchKVCache`` list for mixed warm / cold prefill.

    ``picks`` is per-row, with each entry being ``None`` (cold) or a dict
    with key ``matched_blocks`` (list of APCBlock) and ``prefix_len``.

    Returns ``(cache_list, max_prefix)`` where ``max_prefix`` is the cache's
    ``_idx`` after warm-init (= max prefix_len across rows).

    For row ``i``:
      * left_padding[i] = max_prefix - prefix_len[i]
      * keys[i, :, left_padding[i]:max_prefix, :] = concatenated block K
      * keys[i, :, :left_padding[i], :] = zeros (will be hidden by mask)
    """
    from mlx_lm.models.cache import BatchKVCache

    B = len(picks)
    prefix_lens = [p["prefix_len"] if p else 0 for p in picks]
    max_prefix = max(prefix_lens) if prefix_lens else 0
    if max_prefix == 0:
        return [], 0

    def layer_tensors(pick: dict, layer_idx: int) -> Tuple[mx.array, mx.array]:
        warm_cache = pick.get("warm_cache")
        if warm_cache is not None:
            c = warm_cache[layer_idx]
            prefix_len = pick["prefix_len"]
            return c.keys[..., :prefix_len, :], c.values[..., :prefix_len, :]
        blocks = pick["matched_blocks"]
        ks = [b.keys[layer_idx] for b in blocks]
        vs = [b.values[layer_idx] for b in blocks]
        return mx.concatenate(ks, axis=2), mx.concatenate(vs, axis=2)

    # Discover dtype / head dims from the first non-empty pick.
    sample = next(p for p in picks if p)
    sample_k, _ = layer_tensors(sample, 0)  # [1, H, prefix_len, D]
    n_kv_heads = sample_k.shape[1]
    head_dim = sample_k.shape[-1]
    dtype = sample_k.dtype

    out: List[Any] = []
    for layer_idx in range(num_layers):
        # Build per-row warm K/V tensors of shape [1, H, max_prefix, D]; rows
        # without a hit get zeros, rows with a shorter prefix get zero left-pad.
        row_keys: List[mx.array] = []
        row_values: List[mx.array] = []
        for pick in picks:
            if pick is None:
                # Cold row: full pre-warm zone is left padding (zeros).
                row_keys.append(
                    mx.zeros((1, n_kv_heads, max_prefix, head_dim), dtype=dtype)
                )
                row_values.append(
                    mx.zeros((1, n_kv_heads, max_prefix, head_dim), dtype=dtype)
                )
                continue
            warm_k, warm_v = layer_tensors(pick, layer_idx)
            lp = max_prefix - pick["prefix_len"]
            if lp > 0:
                pad_k = mx.zeros((1, n_kv_heads, lp, head_dim), dtype=dtype)
                pad_v = mx.zeros((1, n_kv_heads, lp, head_dim), dtype=dtype)
                warm_k = mx.concatenate([pad_k, warm_k], axis=2)
                warm_v = mx.concatenate([pad_v, warm_v], axis=2)
            row_keys.append(warm_k)
            row_values.append(warm_v)
        merged_k = mx.concatenate(row_keys, axis=0)  # [B, H, max_prefix, D]
        merged_v = mx.concatenate(row_values, axis=0)

        left_padding = [max_prefix - pl for pl in prefix_lens]
        offset = [pl for pl in prefix_lens]
        c = BatchKVCache(left_padding=[0] * B)  # placeholder; state setter overrides
        c.state = (
            merged_k,
            merged_v,
            mx.array(offset),
            mx.array(left_padding),
        )
        out.append(c)
    return out, max_prefix


def _collect_mx_arrays(x: Any, out: List[mx.array]) -> None:
    if isinstance(x, mx.array):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for item in x:
            _collect_mx_arrays(item, out)


def _merge_arrays_cache_entries(
    entries: Sequence[Any],
    prefix_lens: Sequence[int],
) -> Any:
    from mlx_lm.models import cache as lm_cache

    size = len(entries[0].cache)
    out = lm_cache.ArraysCache(size)
    merged_states: List[Optional[mx.array]] = []
    for state_idx in range(size):
        states = [entry.cache[state_idx] for entry in entries]
        sample = next((s for s in states if s is not None), None)
        if sample is None:
            merged_states.append(None)
            continue
        rows = []
        for state in states:
            if state is None:
                rows.append(mx.zeros((1,) + sample.shape[1:], dtype=sample.dtype))
            else:
                rows.append(state[:1])
        merged_states.append(mx.concatenate(rows, axis=0))
    out.cache = merged_states
    return out


def _merge_exact_cache_entries(
    entries: Sequence[Any],
    prefix_lens: Sequence[int],
) -> Any:
    from mlx_lm.models import cache as lm_cache

    if not entries:
        return None
    first = entries[0]
    if all(isinstance(c, lm_cache.KVCache) for c in entries):
        return lm_cache.BatchKVCache.merge(entries)
    if all(isinstance(c, lm_cache.ChunkedKVCache) for c in entries):
        return lm_cache.BatchKVCache.merge(entries)
    if all(isinstance(c, lm_cache.RotatingKVCache) for c in entries):
        return lm_cache.BatchRotatingKVCache.merge(entries)
    if all(isinstance(c, lm_cache.ArraysCache) for c in entries):
        return _merge_arrays_cache_entries(entries, prefix_lens)
    if all(isinstance(c, lm_cache.CacheList) for c in entries):
        merged = [
            _merge_exact_cache_entries(
                [entry.caches[i] for entry in entries],
                prefix_lens,
            )
            for i in range(len(first.caches))
        ]
        if any(c is None for c in merged):
            return None
        return lm_cache.CacheList(*merged)
    if all(isinstance(c, tuple) for c in entries):
        merged = [
            _merge_exact_cache_entries(
                [entry[i] for entry in entries],
                prefix_lens,
            )
            for i in range(len(first))
        ]
        if any(c is None for c in merged):
            return None
        return lm_cache.CacheList(*merged)
    return None


def make_warm_batch_exact_cache_multi(
    row_caches: Sequence[Sequence[Any]],
    prefix_lens: Sequence[int],
) -> Tuple[Optional[List[Any]], int]:
    """Merge single-row exact-cache snapshots into batch-aware caches."""

    if not row_caches:
        return [], 0
    if len(row_caches) != len(prefix_lens):
        return None, 0
    num_entries = len(row_caches[0])
    if any(len(row) != num_entries for row in row_caches):
        return None, 0

    out: List[Any] = []
    for entry_idx in range(num_entries):
        merged = _merge_exact_cache_entries(
            [row[entry_idx] for row in row_caches],
            prefix_lens,
        )
        if merged is None:
            return None, 0
        out.append(merged)

    eval_targets: List[mx.array] = []
    for c in out:
        _collect_mx_arrays(c.state, eval_targets)
    if eval_targets:
        mx.eval(eval_targets)
    return out, max(prefix_lens) if prefix_lens else 0


def extract_prompt_cache_from_batch(
    batch_caches: Sequence[Any],
    batch_idx: int,
) -> Optional[List[Any]]:
    """Extract one row from batch-aware caches as single-row cache objects."""

    out: List[Any] = []
    eval_targets: List[mx.array] = []
    for c in batch_caches:
        extract = getattr(c, "extract", None)
        if not callable(extract):
            return None
        extracted = extract(batch_idx)
        out.append(extracted)
        _collect_mx_arrays(extracted.state, eval_targets)
    if eval_targets:
        mx.eval(eval_targets)
    return out


def harvest_blocks_from_batch_cache(
    apc_manager: "APCManager",
    batch_caches: List[Any],
    batch_idx: int,
    full_token_ids: Sequence[int],
    *,
    extra_hash: int = 0,
    skip_first_n_tokens: int = 0,
) -> List[APCBlock]:
    """Slice one row out of a batched KV cache and store its full blocks.

    Used at the end of prompt prefill in continuous-batching mode to add
    the new prefix to APC.
    """
    layer_keys: List[mx.array] = []
    layer_values: List[mx.array] = []
    for c in batch_caches:
        keys = getattr(c, "keys", None)
        values = getattr(c, "values", None)
        idx = getattr(c, "_idx", None)
        left_padding = getattr(c, "left_padding", None)
        if keys is None or values is None or idx is None:
            return []
        # Pull this batch row, dropping any left-padding for this seq.
        if left_padding is not None:
            try:
                lp = int(left_padding[batch_idx].item())
            except Exception:
                lp = 0
        else:
            lp = 0
        # shape after slicing: [1, H, idx-lp, D]
        layer_keys.append(keys[batch_idx : batch_idx + 1, :, lp:idx, :])
        layer_values.append(values[batch_idx : batch_idx + 1, :, lp:idx, :])
    return apc_manager.store_kv_blocks(
        full_token_ids,
        layer_keys,
        layer_values,
        extra_hash=extra_hash,
        skip_first_n_tokens=skip_first_n_tokens,
    )


def model_apc_mode(language_model: Any) -> Optional[str]:
    """Return the APC strategy supported by ``language_model``.

    ``"block"`` is the normal block-level KV path. ``"exact"`` is a
    conservative whole-prefix snapshot path for custom mixed cache layouts
    such as hybrid SSM/attention models, where recurrent state cannot be
    reconstructed by concatenating K/V blocks alone.
    """
    if not hasattr(language_model, "make_cache"):
        return "block"
    try:
        prompt_cache = language_model.make_cache()
    except Exception:
        return None
    if prompt_cache and all(_cache_entry_supports_block_apc(c) for c in prompt_cache):
        return "block"
    if prompt_cache and all(_cache_entry_supports_exact_apc(c) for c in prompt_cache):
        return "exact"
    return None


def model_supports_apc(language_model: Any) -> bool:
    return model_apc_mode(language_model) is not None


def from_env(model_namespace: Optional[str] = None) -> Optional[APCManager]:
    """Build an APCManager from env vars when ``APC_ENABLED=1``, else None.

    When ``APC_DISK_PATH`` is set, also wires up the shard-based SSD tier.
    The disk read path defaults to direct file reads so restored K/V tensors
    are MLX-owned buffers rather than mmap-backed safetensors views.
    """
    if os.environ.get("APC_ENABLED", "1").lower() in ("0", "false", "no", "off"):
        return None
    block_size = int(os.environ.get("APC_BLOCK_SIZE", DEFAULT_BLOCK_SIZE))
    num_blocks = int(os.environ.get("APC_NUM_BLOCKS", DEFAULT_NUM_BLOCKS))

    disk: Optional[DiskBlockStore] = None
    disk_path = os.environ.get("APC_DISK_PATH")
    if not disk_path:
        disk_path = os.path.expanduser("~/.cache/xmlx_vlm/apc")
    if disk_path:
        ns = model_namespace or os.environ.get("APC_DISK_NAMESPACE", "default")
        max_gb = float(os.environ.get("APC_DISK_MAX_GB", 0))
        max_bytes = int(max_gb * (1 << 30)) if max_gb > 0 else None
        workers = int(os.environ.get("APC_DISK_WORKERS", "1"))
        try:
            disk = DiskBlockStore(
                Path(disk_path).expanduser(),
                namespace=ns,
                num_workers=workers,
                max_bytes=max_bytes,
            )
            cap_str = f"{max_gb:.1f} GB" if max_bytes else "unbounded"
            logger.info(
                "APC disk tier at %s (ns=%s, cap=%s, read_mode=%s)",
                disk.dir,
                ns,
                cap_str,
                disk._read_mode,
            )
        except Exception as e:
            logger.warning("APC disk tier disabled (init failed): %s", e)

    logger.info(
        "APC enabled (block_size=%d, num_blocks=%d, hash=%s, disk=%s)",
        block_size,
        num_blocks,
        "sha256" if _hash_use_sha256() else "fast",
        bool(disk),
    )
    return APCManager(num_blocks=num_blocks, block_size=block_size, disk=disk)
