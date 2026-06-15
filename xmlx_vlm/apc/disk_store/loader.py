from __future__ import annotations
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import mlx.core as mx
import numpy as np

from .safetensors_ops import (
    _read_safetensors_header, _read_safetensors_metadata, _read_safetensors_tensor,
    _read_safetensors_axis0_slice_bytes, _read_safetensors_axis0_slice, _mlx_array_from_safetensors_bytes,
    _safetensors_dtype_info, _safetensors_tensor_bounds, _numel, _safe_namespace
)
from ..types import (
    APC_DISK_SCHEMA_VERSION, _DiskBlockSnapshot, _DiskLayerMajorBlock, _DiskLayerMajorSnapshot,
    _DiskExactCacheSnapshot, APCExactCacheEntry, SEED_PARENT_HASH
)

logger = logging.getLogger('xmlx_vlm.apc')

class DiskBlockStoreLoaderMixin:
        def _open_shard_header(self, shard_path: Path):
            """Return parsed safetensors header info for a shard, cached."""
            with self._header_cache_lock:
                cached = self._header_cache.get(shard_path)
                if cached is not None:
                    self._header_cache.move_to_end(shard_path)
                    return cached
            parsed = _read_safetensors_header(shard_path)
            if parsed is None:
                logger.warning("APC disk shard header read failed for %s", shard_path)
                return None
            with self._header_cache_lock:
                self._header_cache[shard_path] = parsed
                self._header_cache.move_to_end(shard_path)
                while len(self._header_cache) > self._header_cache_max:
                    self._header_cache.popitem(last=False)
            return parsed

        def _open_shard(self, shard_path: Path):
            """Return (arrays_dict, file_metadata) for a shard, mmap-cached."""
            with self._mmap_cache_lock:
                cached = self._mmap_cache.get(shard_path)
                if cached is not None:
                    self._mmap_cache.move_to_end(shard_path)
                    return cached
            try:
                arrays, metadata = mx.load(str(shard_path), return_metadata=True)
            except Exception as e:
                logger.warning("APC disk shard load failed for %s: %s", shard_path, e)
                return None
            # Touch recency timestamp so LRU eviction prefers truly-cold shards.
            try:
                os.utime(shard_path, None)
            except OSError:
                pass
            bundle = (dict(arrays), dict(metadata))
            with self._mmap_cache_lock:
                self._mmap_cache[shard_path] = bundle
                self._mmap_cache.move_to_end(shard_path)
                while len(self._mmap_cache) > self._mmap_cache_max:
                    self._mmap_cache.popitem(last=False)
            return bundle

        def has(self, block_hash: int) -> bool:
            with self._index_lock:
                return block_hash in self._index

        def has_exact(self, cache_hash: int) -> bool:
            with self._index_lock:
                return cache_hash in self._exact_index

        def find_exact_prefix(
            self,
            token_ids: Sequence[int],
            *,
            extra_hash: int = 0,
            max_prefix_tokens: Optional[int] = None,
            min_prefix_tokens: int = 0,
        ) -> Optional[Tuple[int, int]]:
            token_tuple = tuple(int(t) for t in token_ids)
            max_len = len(token_tuple) - 1
            if max_prefix_tokens is not None and max_prefix_tokens > 0:
                max_len = min(max_len, int(max_prefix_tokens))
            if max_len <= min_prefix_tokens:
                return None

            with self._index_lock:
                entries = list(self._exact_index.items())

            best: Optional[Tuple[int, int]] = None
            for cache_hash, path in entries:
                parsed = self._open_shard_header(path)
                if parsed is None:
                    continue
                _tensor_entries, metadata, _data_start = parsed
                if metadata.get("layout") != "exact_cache_v1":
                    continue
                try:
                    stored_extra = int(metadata.get("extra_hash", "0"))
                    stored_tokens = tuple(
                        int(x) for x in metadata.get("token_ids", "").split(",") if x
                    )
                except (TypeError, ValueError):
                    continue
                prefix_len = len(stored_tokens)
                if (
                    stored_extra != extra_hash
                    or prefix_len <= min_prefix_tokens
                    or prefix_len > max_len
                    or token_tuple[:prefix_len] != stored_tokens
                ):
                    continue
                if best is None or prefix_len > best[1]:
                    best = (int(cache_hash), prefix_len)
            return best

        def load_exact_cache(
            self,
            cache_hash: int,
            *,
            wait_in_flight_ms: float = 0.0,
            min_capacity_tokens: Optional[int] = None,
        ) -> Optional[Tuple[Tuple[int, ...], int, List[Any], Optional[mx.array]]]:
            with self._index_lock:
                path = self._exact_index.get(cache_hash)
            if path is None:
                if wait_in_flight_ms > 0:
                    with self._in_flight_lock:
                        ev = self._in_flight.get(cache_hash)
                    if ev is not None and ev.wait(wait_in_flight_ms / 1000.0):
                        with self._index_lock:
                            path = self._exact_index.get(cache_hash)
                if path is None:
                    return None
            return self._load_exact_cache_file(
                path, min_capacity_tokens=min_capacity_tokens
            )

        def _load_exact_cache_file(
            self,
            path: Path,
            *,
            min_capacity_tokens: Optional[int],
        ) -> Optional[Tuple[Tuple[int, ...], int, List[Any], Optional[mx.array]]]:
            """Load an exact-cache file from disk.

            Returns ``(token_ids, extra_hash, prompt_cache, logits)`` where
            ``logits`` is the saved log-softmax vector (vocab_size,) for the next
            token, or ``None`` when the snapshot was written without logits.
            """
            parsed = self._open_shard_header(path)
            if parsed is None:
                return None
            tensor_entries, metadata, data_start = parsed
            if metadata.get("layout") != "exact_cache_v1":
                return None
            # Schema version guard — reject shards from a different build to prevent
            # silent tensor misreads.  Missing version field → treated as "0" (pre-
            # versioning build) → rejected.  The shard will be overwritten on the
            # next store_exact_cache call, so this is self-healing.
            shard_ver = metadata.get("schema_version", "0")
            if shard_ver != APC_DISK_SCHEMA_VERSION:
                logger.debug(
                    "APC disk: rejecting exact-cache shard %s — schema_version %r "
                    "!= expected %r; will re-prefill and overwrite",
                    path.name, shard_ver, APC_DISK_SCHEMA_VERSION,
                )
                return None
            try:
                token_ids = tuple(
                    int(x) for x in metadata.get("token_ids", "").split(",") if x
                )
                extra_hash = int(metadata.get("extra_hash", "0"))
                n_entries = int(metadata.get("num_entries", "0"))
            except (TypeError, ValueError):
                return None
            if n_entries <= 0:
                return None

            prompt_cache: List[Any] = []
            eval_targets: List[mx.array] = []
            for i in range(n_entries):
                loaded = self._load_exact_cache_entry(
                    path,
                    tensor_entries,
                    metadata,
                    data_start,
                    f"c{i}",
                    min_capacity_tokens=min_capacity_tokens,
                    eval_targets=eval_targets,
                )
                if loaded is None:
                    return None
                prompt_cache.append(loaded)

            # ---- ds4-style next-token logits restore ----
            logits_arr: Optional[mx.array] = None
            if metadata.get("has_logits") == "1":
                le = tensor_entries.get("next_logits")
                if le is not None:
                    raw = _read_safetensors_tensor(path, data_start, le)
                    if raw is not None:
                        # Saved as (1, vocab_size); squeeze to (vocab_size,)
                        logits_arr = raw.reshape(-1)
                        eval_targets.append(logits_arr)

            if eval_targets:
                mx.eval(eval_targets)
            try:
                os.utime(path, None)
            except OSError:
                pass
            return token_ids, extra_hash, prompt_cache, logits_arr

        def _load_exact_cache_entry(
            self,
            path: Path,
            tensor_entries: dict,
            metadata: dict,
            data_start: int,
            prefix: str,
            *,
            min_capacity_tokens: Optional[int],
            eval_targets: List[mx.array],
        ) -> Optional[Any]:
            from mlx_lm.models import cache as lm_cache

            kind = metadata.get(f"{prefix}_kind")
            if kind == "kv":
                if metadata.get(f"{prefix}_empty", "0") == "1":
                    c = lm_cache.KVCache()
                    try:
                        c.offset = int(metadata.get(f"{prefix}_offset", "0"))
                    except (TypeError, ValueError):
                        c.offset = 0
                    return c
                k_entry = tensor_entries.get(f"{prefix}_k")
                v_entry = tensor_entries.get(f"{prefix}_v")
                if k_entry is None or v_entry is None:
                    return None
                k = _read_safetensors_tensor(path, data_start, k_entry)
                v = _read_safetensors_tensor(path, data_start, v_entry)
                if k is None or v is None:
                    return None
                try:
                    off = int(metadata.get(f"{prefix}_offset", str(k.shape[2])))
                    step = int(metadata.get(f"{prefix}_step", "256"))
                except (TypeError, ValueError):
                    return None
                from ..utils import _pad_kv_for_capacity
                k, v = _pad_kv_for_capacity(
                    k,
                    v,
                    offset=off,
                    min_capacity_tokens=min_capacity_tokens,
                    step=step,
                )
                c = lm_cache.KVCache()
                c.keys = k
                c.values = v
                c.offset = off
                eval_targets.extend([k, v])
                return c

            if kind == "rotating_kv":
                try:
                    keep = int(metadata.get(f"{prefix}_keep", "0"))
                    max_size = int(metadata[f"{prefix}_max_size"])
                    offset = int(metadata.get(f"{prefix}_offset", "0"))
                    idx = int(metadata.get(f"{prefix}_idx", "0"))
                except (KeyError, TypeError, ValueError):
                    return None
                c = lm_cache.RotatingKVCache(max_size=max_size, keep=keep)
                c.offset = offset
                c._idx = idx
                if metadata.get(f"{prefix}_empty", "0") == "1":
                    return c
                k_entry = tensor_entries.get(f"{prefix}_k")
                v_entry = tensor_entries.get(f"{prefix}_v")
                if k_entry is None or v_entry is None:
                    return None
                k = _read_safetensors_tensor(path, data_start, k_entry)
                v = _read_safetensors_tensor(path, data_start, v_entry)
                if k is None or v is None:
                    return None
                c.keys = k
                c.values = v
                eval_targets.extend([k, v])
                return c

            if kind == "chunked_kv":
                try:
                    chunk_size = int(metadata[f"{prefix}_chunk_size"])
                    offset = int(metadata.get(f"{prefix}_offset", "0"))
                    start_position = int(metadata.get(f"{prefix}_start_position", "0"))
                except (KeyError, TypeError, ValueError):
                    return None
                c = lm_cache.ChunkedKVCache(chunk_size=chunk_size)
                c.offset = offset
                c.start_position = start_position
                if metadata.get(f"{prefix}_empty", "0") == "1":
                    return c
                k_entry = tensor_entries.get(f"{prefix}_k")
                v_entry = tensor_entries.get(f"{prefix}_v")
                if k_entry is None or v_entry is None:
                    return None
                k = _read_safetensors_tensor(path, data_start, k_entry)
                v = _read_safetensors_tensor(path, data_start, v_entry)
                if k is None or v is None:
                    return None
                c.keys = k
                c.values = v
                eval_targets.extend([k, v])
                return c

            if kind == "arrays":
                try:
                    size = int(metadata.get(f"{prefix}_size", "0"))
                except (TypeError, ValueError):
                    return None
                c = lm_cache.ArraysCache(size=size)
                states: List[Optional[mx.array]] = []
                for j in range(size):
                    if metadata.get(f"{prefix}_s{j}_none", "0") == "1":
                        states.append(None)
                        continue
                    entry = tensor_entries.get(f"{prefix}_s{j}")
                    if entry is None:
                        return None
                    state = _read_safetensors_tensor(path, data_start, entry)
                    if state is None:
                        return None
                    states.append(state)
                    eval_targets.append(state)
                c.cache = states
                lp_entry = tensor_entries.get(f"{prefix}_left_padding")
                if lp_entry is not None:
                    c.left_padding = _read_safetensors_tensor(path, data_start, lp_entry)
                    if c.left_padding is None:
                        return None
                    eval_targets.append(c.left_padding)
                lengths_entry = tensor_entries.get(f"{prefix}_lengths")
                if lengths_entry is not None:
                    c.lengths = _read_safetensors_tensor(path, data_start, lengths_entry)
                    if c.lengths is None:
                        return None
                    eval_targets.append(c.lengths)
                return c

            if kind in ("cache_list", "tuple"):
                try:
                    size = int(metadata.get(f"{prefix}_size", "0"))
                except (TypeError, ValueError):
                    return None
                loaded = []
                for j in range(size):
                    sub_c = self._load_exact_cache_entry(
                        path,
                        tensor_entries,
                        metadata,
                        data_start,
                        f"{prefix}_e{j}",
                        min_capacity_tokens=min_capacity_tokens,
                        eval_targets=eval_targets,
                    )
                    if sub_c is None:
                        return None
                    loaded.append(sub_c)
                if kind == "cache_list":
                    return lm_cache.CacheList(*loaded)
                return tuple(loaded)

            return None

        def load(
            self, block_hash: int, *, wait_in_flight_ms: float = 0.0
        ) -> Optional[Tuple[List[mx.array], List[mx.array], dict]]:
            """Read one block. Returns (keys, values, per-block metadata) or None.

            Per-block metadata is decoded from the shard's ``b{idx}_meta`` JSON
            entry and includes ``token_ids``, ``parent_hash``, ``extra_hash``,
            ``block_hash``.
            """
            with self._index_lock:
                entry = self._index.get(block_hash)
            if entry is None:
                if wait_in_flight_ms > 0:
                    with self._in_flight_lock:
                        ev = self._in_flight.get(block_hash)
                    if ev is not None and ev.wait(wait_in_flight_ms / 1000.0):
                        with self._index_lock:
                            entry = self._index.get(block_hash)
                if entry is None:
                    return None
            shard_path, block_idx = entry
            if self._read_mode == "mmap":
                return self._load_mmap(shard_path, block_idx)
            return self._load_direct(shard_path, block_idx)

        def load_many(
            self, block_hashes: Sequence[int], *, wait_in_flight_ms: float = 0.0
        ) -> List[Optional[Tuple[List[mx.array], List[mx.array], dict]]]:
            """Read multiple blocks, preserving order.

            In direct mode, consecutive requests from the same shard are coalesced
            into larger byte-range reads. In mmap mode, fall back to one-at-a-time
            loads so the old comparison path stays simple and unchanged.
            """
            if not block_hashes:
                return []
            if self._read_mode == "mmap":
                return [
                    self.load(h, wait_in_flight_ms=wait_in_flight_ms) for h in block_hashes
                ]

            entries: List[Optional[Tuple[Path, int]]] = []
            for h in block_hashes:
                with self._index_lock:
                    entry = self._index.get(h)
                if entry is None and wait_in_flight_ms > 0:
                    with self._in_flight_lock:
                        ev = self._in_flight.get(h)
                    if ev is not None and ev.wait(wait_in_flight_ms / 1000.0):
                        with self._index_lock:
                            entry = self._index.get(h)
                entries.append(entry)

            out: List[Optional[Tuple[List[mx.array], List[mx.array], dict]]] = [None] * len(
                block_hashes
            )
            i = 0
            while i < len(entries):
                entry = entries[i]
                if entry is None:
                    i += 1
                    continue
                shard_path = entry[0]
                j = i + 1
                while (
                    j < len(entries)
                    and entries[j] is not None
                    and entries[j][0] == shard_path
                ):
                    j += 1
                block_indices = [entries[k][1] for k in range(i, j)]
                loaded = self._load_direct_many(shard_path, block_indices)
                out[i:j] = loaded
                i = j
            return out

        def _decode_block_metadata(self, file_metadata: dict, block_idx: int) -> dict:
            block_meta_str = file_metadata.get(f"b{block_idx}_meta")
            if not block_meta_str:
                return {}
            try:
                block_meta = json.loads(block_meta_str)
                # Coerce token_ids (encoded as comma-separated ints) to a string
                # for compatibility with the existing verify path.
                if isinstance(block_meta.get("token_ids"), list):
                    block_meta["token_ids"] = ",".join(
                        str(int(t)) for t in block_meta["token_ids"]
                    )
                if "extra_hash" in block_meta:
                    block_meta["extra_hash"] = str(block_meta["extra_hash"])
                return block_meta
            except Exception:
                return {}

        def _load_mmap(
            self, shard_path: Path, block_idx: int
        ) -> Optional[Tuple[List[mx.array], List[mx.array], dict]]:
            bundle = self._open_shard(shard_path)
            if bundle is None:
                return None
            arrays, file_metadata = bundle
            try:
                num_layers = int(file_metadata.get("num_layers", "0"))
            except (TypeError, ValueError):
                return None
            try:
                keys = [arrays[f"b{block_idx}_k{l}"] for l in range(num_layers)]
                values = [arrays[f"b{block_idx}_v{l}"] for l in range(num_layers)]
            except KeyError as e:
                logger.warning("APC disk shard %s missing tensor: %s", shard_path, e)
                return None
            return keys, values, self._decode_block_metadata(file_metadata, block_idx)

        def _load_direct(
            self, shard_path: Path, block_idx: int
        ) -> Optional[Tuple[List[mx.array], List[mx.array], dict]]:
            loaded = self._load_direct_many(shard_path, [block_idx])
            return loaded[0] if loaded else None

        def _load_layer_major_segment(
            self,
            shard_path: Path,
            block_indices: Sequence[int],
            *,
            preserve_capacity: bool = False,
        ) -> Optional[Tuple[List[mx.array], List[mx.array], List[dict]]]:
            if not block_indices:
                return None
            start_idx = block_indices[0]
            if list(block_indices) != list(
                range(start_idx, start_idx + len(block_indices))
            ):
                return None
            parsed = self._open_shard_header(shard_path)
            if parsed is None:
                return None
            tensor_entries, file_metadata, data_start = parsed
            layout = file_metadata.get("layout")
            if layout == "token_major_v2":
                return self._load_token_major_segment(
                    shard_path, tensor_entries, file_metadata, data_start, block_indices
                )
            if layout not in ("layer_major_v1", "layer_major_v2"):
                return None
            # Schema version guard (same policy as exact-cache)
            shard_ver = file_metadata.get("schema_version", "0")
            if shard_ver != APC_DISK_SCHEMA_VERSION:
                logger.debug(
                    "APC disk: rejecting layer-major shard %s — schema_version %r "
                    "!= expected %r; will re-prefill",
                    shard_path.name, shard_ver, APC_DISK_SCHEMA_VERSION,
                )
                return None
            try:
                num_layers = int(file_metadata.get("num_layers", "0"))
                block_size = int(file_metadata.get("block_size", "0"))
            except (TypeError, ValueError):
                return None
            if num_layers <= 0 or block_size <= 0:
                return None

            token_start = start_idx * block_size
            token_end = token_start + len(block_indices) * block_size
            shard_n_blocks = len(
                [x for x in file_metadata.get("block_hashes", "").split(",") if x]
            )
            requested_to_shard_end = (
                shard_n_blocks > 0
                and start_idx == 0
                and start_idx + len(block_indices) >= shard_n_blocks
            )
            slice_end = (
                None
                if (
                    preserve_capacity
                    and layout == "layer_major_v2"
                    and requested_to_shard_end
                )
                else token_end
            )
            keys: List[mx.array] = []
            values: List[mx.array] = []
            for l in range(num_layers):
                k_entry = tensor_entries.get(f"k{l}")
                v_entry = tensor_entries.get(f"v{l}")
                if k_entry is None or v_entry is None:
                    return None
                k = _read_safetensors_tensor(shard_path, data_start, k_entry)
                v = _read_safetensors_tensor(shard_path, data_start, v_entry)
                if k is None or v is None:
                    return None
                keys.append(k[..., token_start:slice_end, :])
                values.append(v[..., token_start:slice_end, :])

            metadata = [
                self._decode_block_metadata(file_metadata, idx) for idx in block_indices
            ]
            try:
                os.utime(shard_path, None)
            except OSError:
                pass
            mx.eval(keys + values)
            return keys, values, metadata

        def _load_layer_major_prefix_segments_layerwise(
            self,
            segments: Sequence[Tuple[Path, List[int]]],
            *,
            preserve_capacity: bool,
        ) -> Optional[Tuple[List[mx.array], List[mx.array], List[dict]]]:
            """Load layer-major segments without holding all segment tensors.

            The older restore path first read every segment's K/V for every layer,
            then concatenated all layers at once. For long prefixes this doubled
            peak MLX memory: segment tensors plus final KVCache tensors. This
            routine reads all segments for one layer, emits that layer's final K/V,
            clears temporary allocator state, and moves to the next layer.
            """
            if not segments:
                return None

            segment_infos: List[
                Tuple[Path, dict, dict, int, int, Optional[int], List[int]]
            ] = []
            metadata: List[dict] = []
            num_layers: Optional[int] = None
            block_size_ref: Optional[int] = None
            last_segment_idx = len(segments) - 1

            for segment_idx, (shard_path, block_indices) in enumerate(segments):
                if not block_indices:
                    return None
                start_idx = block_indices[0]
                if list(block_indices) != list(
                    range(start_idx, start_idx + len(block_indices))
                ):
                    return None
                parsed = self._open_shard_header(shard_path)
                if parsed is None:
                    return None
                tensor_entries, file_metadata, data_start = parsed
                layout = file_metadata.get("layout")
                if layout not in ("layer_major_v1", "layer_major_v2"):
                    return None
                try:
                    shard_layers = int(file_metadata.get("num_layers", "0"))
                    block_size = int(file_metadata.get("block_size", "0"))
                except (TypeError, ValueError):
                    return None
                if shard_layers <= 0 or block_size <= 0:
                    return None
                if num_layers is None:
                    num_layers = shard_layers
                    block_size_ref = block_size
                elif shard_layers != num_layers or block_size != block_size_ref:
                    return None

                token_start = start_idx * block_size
                token_end = token_start + len(block_indices) * block_size
                shard_n_blocks = len(
                    [x for x in file_metadata.get("block_hashes", "").split(",") if x]
                )
                requested_to_shard_end = (
                    shard_n_blocks > 0
                    and start_idx == 0
                    and start_idx + len(block_indices) >= shard_n_blocks
                )
                slice_end = (
                    None
                    if (
                        preserve_capacity
                        and segment_idx == last_segment_idx
                        and layout == "layer_major_v2"
                        and requested_to_shard_end
                    )
                    else token_end
                )
                segment_infos.append(
                    (
                        shard_path,
                        tensor_entries,
                        file_metadata,
                        data_start,
                        token_start,
                        slice_end,
                        list(block_indices),
                    )
                )
                metadata.extend(
                    self._decode_block_metadata(file_metadata, idx) for idx in block_indices
                )
                try:
                    os.utime(shard_path, None)
                except OSError:
                    pass

            if num_layers is None:
                return None

            keys: List[mx.array] = []
            values: List[mx.array] = []
            for layer_idx in range(num_layers):
                k_parts: List[mx.array] = []
                v_parts: List[mx.array] = []
                for (
                    shard_path,
                    tensor_entries,
                    _file_metadata,
                    data_start,
                    token_start,
                    slice_end,
                    _block_indices,
                ) in segment_infos:
                    k_entry = tensor_entries.get(f"k{layer_idx}")
                    v_entry = tensor_entries.get(f"v{layer_idx}")
                    if k_entry is None or v_entry is None:
                        return None
                    k = _read_safetensors_tensor(shard_path, data_start, k_entry)
                    v = _read_safetensors_tensor(shard_path, data_start, v_entry)
                    if k is None or v is None:
                        return None
                    k_parts.append(k[..., token_start:slice_end, :])
                    v_parts.append(v[..., token_start:slice_end, :])

                k_out = k_parts[0] if len(k_parts) == 1 else mx.concatenate(k_parts, axis=2)
                v_out = v_parts[0] if len(v_parts) == 1 else mx.concatenate(v_parts, axis=2)
                mx.eval(k_out, v_out)
                keys.append(k_out)
                values.append(v_out)
                del k_parts, v_parts, k_out, v_out
                if (
                    self._restore_clear_every > 0
                    and (layer_idx + 1) % self._restore_clear_every == 0
                ):
                    mx.clear_cache()

            return keys, values, metadata

        def _load_token_major_segment(
            self,
            shard_path: Path,
            tensor_entries: dict,
            file_metadata: dict,
            data_start: int,
            block_indices: Sequence[int],
        ) -> Optional[Tuple[List[mx.array], List[mx.array], List[dict]]]:
            try:
                num_layers = int(file_metadata.get("num_layers", "0"))
                block_size = int(file_metadata.get("block_size", "0"))
            except (TypeError, ValueError):
                return None
            if num_layers <= 0 or block_size <= 0:
                return None

            start_idx = block_indices[0]
            token_start = start_idx * block_size
            token_end = token_start + len(block_indices) * block_size
            k_entry = tensor_entries.get("k_all")
            v_entry = tensor_entries.get("v_all")
            if k_entry is None or v_entry is None:
                return None

            k_all = _read_safetensors_axis0_slice(
                shard_path, data_start, k_entry, token_start, token_end
            )
            v_all = _read_safetensors_axis0_slice(
                shard_path, data_start, v_entry, token_start, token_end
            )
            if k_all is None or v_all is None:
                return None
            if len(k_all.shape) != 5 or len(v_all.shape) != 5:
                return None
            if k_all.shape[1] != num_layers or v_all.shape[1] != num_layers:
                return None

            keys = [mx.transpose(k_all[:, l, ...], (1, 2, 0, 3)) for l in range(num_layers)]
            values = [
                mx.transpose(v_all[:, l, ...], (1, 2, 0, 3)) for l in range(num_layers)
            ]
            metadata = [
                self._decode_block_metadata(file_metadata, idx) for idx in block_indices
            ]
            try:
                os.utime(shard_path, None)
            except OSError:
                pass
            mx.eval([k_all, v_all])
            return keys, values, metadata

        def _load_token_major_prefix_segments(
            self, segments: Sequence[Tuple[Path, List[int]]]
        ) -> Optional[Tuple[List[mx.array], List[mx.array], List[dict]]]:
            """Fast path for token-major shards.

            Concatenate raw token-major byte ranges before constructing MLX arrays.
            This avoids a first-request MLX compile of 72 per-layer concatenations
            when a prefix spans a common-prefix shard plus a request-specific shard.
            """
            if not segments:
                return None

            num_layers: Optional[int] = None
            block_size_ref: Optional[int] = None
            k_tail_shape: Optional[Tuple[int, ...]] = None
            v_tail_shape: Optional[Tuple[int, ...]] = None
            k_dtype: Optional[str] = None
            v_dtype: Optional[str] = None
            total_tokens = 0
            k_buf = bytearray()
            v_buf = bytearray()
            metadata: List[dict] = []

            for shard_path, block_indices in segments:
                parsed = self._open_shard_header(shard_path)
                if parsed is None:
                    return None
                tensor_entries, file_metadata, data_start = parsed
                if file_metadata.get("layout") != "token_major_v2":
                    return None
                try:
                    shard_layers = int(file_metadata.get("num_layers", "0"))
                    block_size = int(file_metadata.get("block_size", "0"))
                except (TypeError, ValueError):
                    return None
                if shard_layers <= 0 or block_size <= 0:
                    return None
                if num_layers is None:
                    num_layers = shard_layers
                    block_size_ref = block_size
                elif shard_layers != num_layers or block_size != block_size_ref:
                    return None

                k_entry = tensor_entries.get("k_all")
                v_entry = tensor_entries.get("v_all")
                if k_entry is None or v_entry is None:
                    return None
                start_idx = block_indices[0]
                token_start = start_idx * block_size
                token_end = token_start + len(block_indices) * block_size
                k_sliced = _read_safetensors_axis0_slice_bytes(
                    shard_path, data_start, k_entry, token_start, token_end
                )
                v_sliced = _read_safetensors_axis0_slice_bytes(
                    shard_path, data_start, v_entry, token_start, token_end
                )
                if k_sliced is None or v_sliced is None:
                    return None
                k_raw, k_sliced_entry = k_sliced
                v_raw, v_sliced_entry = v_sliced
                k_shape = tuple(int(x) for x in k_sliced_entry["shape"])
                v_shape = tuple(int(x) for x in v_sliced_entry["shape"])
                if len(k_shape) != 5 or len(v_shape) != 5:
                    return None
                if k_shape[1] != num_layers or v_shape[1] != num_layers:
                    return None
                if k_shape[0] != v_shape[0]:
                    return None
                if k_tail_shape is None:
                    k_tail_shape = k_shape[1:]
                    v_tail_shape = v_shape[1:]
                    k_dtype = str(k_sliced_entry["dtype"])
                    v_dtype = str(v_sliced_entry["dtype"])
                elif (
                    k_tail_shape != k_shape[1:]
                    or v_tail_shape != v_shape[1:]
                    or k_dtype != str(k_sliced_entry["dtype"])
                    or v_dtype != str(v_sliced_entry["dtype"])
                ):
                    return None

                k_buf.extend(k_raw)
                v_buf.extend(v_raw)
                total_tokens += k_shape[0]
                metadata.extend(
                    self._decode_block_metadata(file_metadata, idx) for idx in block_indices
                )
                try:
                    os.utime(shard_path, None)
                except OSError:
                    pass

            if (
                num_layers is None
                or k_tail_shape is None
                or v_tail_shape is None
                or k_dtype is None
                or v_dtype is None
                or total_tokens <= 0
            ):
                return None

            k_dtype_info = _safetensors_dtype_info(k_dtype)
            v_dtype_info = _safetensors_dtype_info(v_dtype)
            if k_dtype_info is None or v_dtype_info is None:
                return None
            k_np_dtype, k_mlx_dtype, k_bitcast_to = k_dtype_info
            v_np_dtype, v_mlx_dtype, v_bitcast_to = v_dtype_info
            try:
                k_np = np.frombuffer(k_buf, dtype=k_np_dtype).reshape(
                    (total_tokens, *k_tail_shape)
                )
                v_np = np.frombuffer(v_buf, dtype=v_np_dtype).reshape(
                    (total_tokens, *v_tail_shape)
                )
            except ValueError:
                return None

            # Build standard contiguous KVCache slabs with one decode step of spare
            # capacity. Exact-size restored caches make KVCache.update_and_fetch()
            # grow via 72 MLX concatenations on the first generated token, which is
            # a large first-use compile. Padding here is a plain NumPy copy.
            kv_step = 256
            capacity = ((total_tokens + 1 + kv_step - 1) // kv_step) * kv_step
            keys: List[mx.array] = []
            values: List[mx.array] = []
            for l in range(num_layers):
                k_layer = np.zeros(
                    (k_tail_shape[1], k_tail_shape[2], capacity, k_tail_shape[3]),
                    dtype=k_np_dtype,
                )
                v_layer = np.zeros(
                    (v_tail_shape[1], v_tail_shape[2], capacity, v_tail_shape[3]),
                    dtype=v_np_dtype,
                )
                k_layer[..., :total_tokens, :] = k_np[:, l, ...].transpose(1, 2, 0, 3)
                v_layer[..., :total_tokens, :] = v_np[:, l, ...].transpose(1, 2, 0, 3)
                keys.append(mx.array(k_layer, dtype=k_mlx_dtype))
                values.append(mx.array(v_layer, dtype=v_mlx_dtype))
            from ..utils import _copy_mlx_array
            if k_bitcast_to is not None:
                keys = [k.view(k_bitcast_to) for k in keys]
            if v_bitcast_to is not None:
                values = [v.view(v_bitcast_to) for v in values]
            if k_bitcast_to is not None:
                keys = [_copy_mlx_array(k) for k in keys]
            if v_bitcast_to is not None:
                values = [_copy_mlx_array(v) for v in values]
            mx.eval(keys + values)
            return keys, values, metadata

        def load_layer_major_prefix(
            self, block_hashes: Sequence[int], *, preserve_capacity: bool = True
        ) -> Optional[Tuple[List[mx.array], List[mx.array], List[dict]]]:
            """Load a cached prefix directly as per-layer K/V tensors.

            Handles prefixes that span several layer-major shards. Returns
            ``(keys, values, per_block_metadata)`` where each key/value tensor
            covers the full requested prefix for one layer. This is the warm-disk
            fast path: 72 tensors for a Qwen3-VL-4B prefix instead of 209 * 72
            block slabs.
            """
            if not block_hashes:
                return None
            trace = os.environ.get("APC_DISK_TRACE", "").lower() in ("1", "true", "yes")
            trace_t0 = time.perf_counter()

            entries: List[Tuple[Path, int]] = []
            for h in block_hashes:
                with self._index_lock:
                    entry = self._index.get(h)
                if entry is None:
                    return None
                entries.append(entry)

            segments: List[Tuple[Path, List[int]]] = []
            for shard_path, block_idx in entries:
                if (
                    not segments
                    or segments[-1][0] != shard_path
                    or segments[-1][1][-1] + 1 != block_idx
                ):
                    segments.append((shard_path, [block_idx]))
                else:
                    segments[-1][1].append(block_idx)

            trace_raw_t0 = time.perf_counter()
            raw_token_major = self._load_token_major_prefix_segments(segments)
            trace_raw_t1 = time.perf_counter()
            if raw_token_major is not None:
                if trace:
                    print(
                        "APC_DISK_TRACE restore "
                        f"blocks={len(block_hashes)} segments={len(segments)} "
                        f"raw_token_major={trace_raw_t1 - trace_raw_t0:.3f}s "
                        f"total={trace_raw_t1 - trace_t0:.3f}s",
                        flush=True,
                    )
                return raw_token_major

            trace_layerwise_t0 = time.perf_counter()
            layerwise = self._load_layer_major_prefix_segments_layerwise(
                segments,
                preserve_capacity=preserve_capacity,
            )
            trace_layerwise_t1 = time.perf_counter()
            if layerwise is not None:
                if trace:
                    print(
                        "APC_DISK_TRACE restore "
                        f"blocks={len(block_hashes)} segments={len(segments)} "
                        f"layerwise={trace_layerwise_t1 - trace_layerwise_t0:.3f}s "
                        f"total={trace_layerwise_t1 - trace_t0:.3f}s",
                        flush=True,
                    )
                return layerwise

            trace_load_t0 = time.perf_counter()
            loaded_segments = []
            last_segment_idx = len(segments) - 1
            for segment_idx, (shard_path, block_indices) in enumerate(segments):
                loaded_segments.append(
                    self._load_layer_major_segment(
                        shard_path,
                        block_indices,
                        preserve_capacity=(
                            preserve_capacity
                            and segment_idx == last_segment_idx
                            and bool(block_indices)
                            and block_indices[0] == 0
                        ),
                    )
                )
            trace_load_t1 = time.perf_counter()
            if any(seg is None for seg in loaded_segments):
                return None

            first_keys, first_values, _ = loaded_segments[0]
            num_layers = len(first_keys)
            if num_layers == 0 or len(first_values) != num_layers:
                return None

            keys: List[mx.array] = []
            values: List[mx.array] = []
            metadata: List[dict] = []
            for seg in loaded_segments:
                seg_keys, seg_values, seg_metadata = seg
                if len(seg_keys) != num_layers or len(seg_values) != num_layers:
                    return None
                metadata.extend(seg_metadata)

            trace_concat_t0 = time.perf_counter()
            for l in range(num_layers):
                keys.append(mx.concatenate([seg[0][l] for seg in loaded_segments], axis=2))
                values.append(
                    mx.concatenate([seg[1][l] for seg in loaded_segments], axis=2)
                )
            mx.eval(keys + values)
            trace_concat_t1 = time.perf_counter()
            if trace:
                print(
                    "APC_DISK_TRACE restore "
                    f"blocks={len(block_hashes)} segments={len(segments)} "
                    f"load={trace_load_t1 - trace_load_t0:.3f}s "
                    f"concat_eval={trace_concat_t1 - trace_concat_t0:.3f}s "
                    f"total={trace_concat_t1 - trace_t0:.3f}s",
                    flush=True,
                )
            return keys, values, metadata

        def _collect_direct_specs(
            self,
            tensor_entries: dict,
            num_layers: int,
            block_indices: Sequence[int],
            shard_path: Path,
        ):
            specs = []
            total_bytes = 0
            for block_idx in block_indices:
                for l in range(num_layers):
                    for suffix in ("k", "v"):
                        name = f"b{block_idx}_{suffix}{l}"
                        entry = tensor_entries.get(name)
                        if entry is None:
                            logger.warning(
                                "APC disk shard %s missing tensor: %s", shard_path, name
                            )
                            return None
                        bounds = _safetensors_tensor_bounds(entry)
                        if bounds is None:
                            logger.warning(
                                "APC disk shard %s has unsupported/corrupt tensor: %s",
                                shard_path,
                                name,
                            )
                            return None
                        start, end, _ = bounds
                        specs.append((block_idx, name, entry, start, end))
                        total_bytes += end - start
            return specs, total_bytes

        def _load_direct_many(
            self, shard_path: Path, block_indices: Sequence[int]
        ) -> List[Optional[Tuple[List[mx.array], List[mx.array], dict]]]:
            if not block_indices:
                return []
            parsed = self._open_shard_header(shard_path)
            if parsed is None:
                return [None] * len(block_indices)
            tensor_entries, file_metadata, data_start = parsed
            try:
                num_layers = int(file_metadata.get("num_layers", "0"))
            except (TypeError, ValueError):
                return [None] * len(block_indices)

            if file_metadata.get("layout") in (
                "layer_major_v1",
                "layer_major_v2",
                "token_major_v2",
            ):
                try:
                    block_hashes = [
                        int(json.loads(file_metadata[f"b{idx}_meta"])["block_hash"])
                        for idx in block_indices
                    ]
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    return [None] * len(block_indices)
                loaded = self.load_layer_major_prefix(block_hashes, preserve_capacity=False)
                if loaded is None:
                    return [None] * len(block_indices)
                layer_keys, layer_values, metadatas = loaded
                out = []
                try:
                    block_size = int(file_metadata.get("block_size", "0"))
                except (TypeError, ValueError):
                    return [None] * len(block_indices)
                for i, md in enumerate(metadatas):
                    start = i * block_size
                    end = start + block_size
                    out.append(
                        (
                            [k[..., start:end, :] for k in layer_keys],
                            [v[..., start:end, :] for v in layer_values],
                            md,
                        )
                    )
                return out

            collected = self._collect_direct_specs(
                tensor_entries, num_layers, block_indices, shard_path
            )
            if collected is None:
                return [None] * len(block_indices)
            specs, total_bytes = collected
            if not specs:
                return [
                    ([], [], self._decode_block_metadata(file_metadata, block_idx))
                    for block_idx in block_indices
                ]

            min_start = min(start for _, _, _, start, _ in specs)
            max_end = max(end for _, _, _, _, end in specs)
            span = max_end - min_start
            if (
                len(block_indices) > 1
                and span > total_bytes + self._direct_max_overread_bytes
            ):
                mid = len(block_indices) // 2
                return self._load_direct_many(
                    shard_path, block_indices[:mid]
                ) + self._load_direct_many(shard_path, block_indices[mid:])

            try:
                with open(shard_path, "rb") as f:
                    # ``mx.save_safetensors`` may reorder tensors in the data
                    # buffer, so we compute the exact span from the header. For a
                    # chain-contiguous shard restore this is usually one compact
                    # range, turning hundreds of small reads into one larger read.
                    f.seek(data_start + min_start)
                    slab = f.read(span)
                    if len(slab) != span:
                        return [None] * len(block_indices)
                    view = memoryview(slab)
                    raw_by_name = {
                        name: view[start - min_start : end - min_start]
                        for _, name, _, start, end in specs
                    }
            except OSError as e:
                logger.warning("APC disk direct read failed for %s: %s", shard_path, e)
                return [None] * len(block_indices)

            entries_by_name = {name: entry for _, name, entry, _, _ in specs}
            out: List[Optional[Tuple[List[mx.array], List[mx.array], dict]]] = []
            for block_idx in block_indices:
                keys: List[mx.array] = []
                values: List[mx.array] = []
                ok = True
                for l in range(num_layers):
                    k_name = f"b{block_idx}_k{l}"
                    v_name = f"b{block_idx}_v{l}"
                    k = _mlx_array_from_safetensors_bytes(
                        raw_by_name[k_name], entries_by_name[k_name]
                    )
                    v = _mlx_array_from_safetensors_bytes(
                        raw_by_name[v_name], entries_by_name[v_name]
                    )
                    if k is None or v is None:
                        ok = False
                        break
                    keys.append(k)
                    values.append(v)
                if ok:
                    out.append(
                        (
                            keys,
                            values,
                            self._decode_block_metadata(file_metadata, block_idx),
                        )
                    )
                else:
                    out.append(None)

            # Touch recency timestamp so LRU eviction prefers truly-cold shards.
            try:
                os.utime(shard_path, None)
            except OSError:
                pass
            return out

