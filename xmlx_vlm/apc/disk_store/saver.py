from __future__ import annotations
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import mlx.core as mx

from ..types import (
    APC_DISK_SCHEMA_VERSION, _DiskBlockSnapshot, _DiskLayerMajorBlock, _DiskLayerMajorSnapshot,
    _DiskExactCacheSnapshot, APCExactCacheEntry, SEED_PARENT_HASH
)

logger = logging.getLogger('xmlx_vlm.apc')

class DiskBlockStoreSaverMixin:
        def save_batch(self, blocks: List["APCBlock"]) -> None:
            """Schedule segment-shard writes containing ``blocks``. Returns
            immediately; the writer thread does the safetensors save + atomic
            rename + index update.
            """
            if not blocks:
                return

            snapshots: List[_DiskBlockSnapshot] = []
            for b in blocks:
                if b.block_hash is None or b.keys is None or b.values is None:
                    continue
                snapshots.append(
                    _DiskBlockSnapshot(
                        block_hash=int(b.block_hash),
                        parent_hash=int(b.parent_hash),
                        extra_hash=int(b.extra_hash),
                        token_ids=tuple(int(t) for t in b.token_ids),
                        keys=list(b.keys),
                        values=list(b.values),
                    )
                )
                if len(snapshots) >= self._shard_max_blocks:
                    self._enqueue_block_snapshots(snapshots)
                    snapshots = []
            if not snapshots:
                return

            self._enqueue_block_snapshots(snapshots)

        def save_exact_cache(
            self,
            cache_hash: int,
            token_ids: Sequence[int],
            extra_hash: int,
            prompt_cache: Sequence[Any],
            logits: Optional[mx.array] = None,
        ) -> None:
            """Schedule an exact prompt-cache snapshot write.

            Exact snapshots are used for custom cache layouts that cannot be
            reconstructed from independently concatenated K/V blocks.

            ``logits`` is the optional full log-softmax vector (vocab_size,) for
            the first token after the cached prefix.  When provided it is stored
            alongside the KV tensors and returned by ``load_exact_cache`` so
            callers can skip one decode step on restore.
            """
            token_tuple = tuple(int(t) for t in token_ids)
            if not token_tuple or not prompt_cache:
                return
            snapshot = _DiskExactCacheSnapshot(
                cache_hash=int(cache_hash),
                token_ids=token_tuple,
                extra_hash=int(extra_hash),
                prompt_cache=list(prompt_cache),
                logits=logits,
            )
            self._enqueue_exact_snapshot(snapshot)

        def save_layer_major_blocks(
            self,
            blocks: List[_DiskLayerMajorBlock],
            layer_keys: Sequence[mx.array],
            layer_values: Sequence[mx.array],
            block_size: int,
        ) -> None:
            """Schedule a layer-major shard write directly from a KV cache.

            This avoids building thousands of per-block MLX tensors when the
            caller only needs durable disk persistence for future warm restores.
            """
            if not blocks or not layer_keys or not layer_values:
                return
            shared_layer_keys = list(layer_keys)
            shared_layer_values = list(layer_values)
            all_block_hashes = [b.block_hash for b in blocks]
            store_id = self._shard_id_for(all_block_hashes)
            segment_count = (
                len(blocks) + self._shard_max_blocks - 1
            ) // self._shard_max_blocks
            for start in range(0, len(blocks), self._shard_max_blocks):
                chunk = list(blocks[start : start + self._shard_max_blocks])
                block_hashes = [b.block_hash for b in chunk]
                snapshot = _DiskLayerMajorSnapshot(
                    blocks=chunk,
                    layer_keys=shared_layer_keys,
                    layer_values=shared_layer_values,
                    block_size=int(block_size),
                    store_id=store_id,
                    segment_index=start // self._shard_max_blocks,
                    segment_count=segment_count,
                )
                self._enqueue_shard(
                    self._shard_id_for(block_hashes), block_hashes, snapshot
                )

        def _enqueue_block_snapshots(self, snapshots: List[_DiskBlockSnapshot]) -> None:
            block_hashes = [b.block_hash for b in snapshots]
            self._enqueue_shard(
                self._shard_id_for(block_hashes), block_hashes, list(snapshots)
            )

        def _enqueue_exact_snapshot(self, snapshot: _DiskExactCacheSnapshot) -> None:
            cache_hash = int(snapshot.cache_hash)
            shard_id = self._exact_id_for(cache_hash)
            path = self._shard_path(shard_id)
            if path.exists():
                with self._index_lock:
                    self._exact_index.setdefault(cache_hash, path)
                return

            ev = threading.Event()
            with self._in_flight_lock:
                self._in_flight[cache_hash] = ev
            try:
                self._q.put_nowait((shard_id, [cache_hash], snapshot, ev))
            except queue.Full:
                with self._in_flight_lock:
                    self._in_flight.pop(cache_hash, None)
                ev.set()
                logger.warning("APC disk write queue full; dropping exact-cache snapshot")

        def _enqueue_shard(
            self,
            shard_id: str,
            block_hashes: Sequence[int],
            payload: Any,
        ) -> None:
            path = self._shard_path(shard_id)
            # Already on disk? Just dedup.
            if path.exists():
                with self._index_lock:
                    # Make sure index reflects it (e.g. after restart).
                    for idx, block_hash in enumerate(block_hashes):
                        self._index.setdefault(int(block_hash), (path, idx))
                return

            ev = threading.Event()
            with self._in_flight_lock:
                for block_hash in block_hashes:
                    self._in_flight[int(block_hash)] = ev
            try:
                self._q.put_nowait((shard_id, list(block_hashes), payload, ev))
            except queue.Full:
                with self._in_flight_lock:
                    for block_hash in block_hashes:
                        self._in_flight.pop(int(block_hash), None)
                ev.set()
                logger.warning(
                    "APC disk write queue full; dropping shard with %d blocks",
                    len(block_hashes),
                )

        @staticmethod
        def _pad_layer_major_arrays(
            layer_keys: List[mx.array],
            layer_values: List[mx.array],
        ) -> Tuple[List[mx.array], List[mx.array]]:
            if not layer_keys:
                return layer_keys, layer_values
            total_tokens = int(layer_keys[0].shape[2])
            kv_step = 256
            capacity = ((total_tokens + 1 + kv_step - 1) // kv_step) * kv_step
            pad_tokens = capacity - total_tokens
            if pad_tokens <= 0:
                return layer_keys, layer_values

            padded_keys: List[mx.array] = []
            padded_values: List[mx.array] = []
            for k, v in zip(layer_keys, layer_values):
                if len(k.shape) != 4 or len(v.shape) != 4:
                    padded_keys.append(k)
                    padded_values.append(v)
                    continue
                k_pad_shape = (*k.shape[:2], pad_tokens, k.shape[3])
                v_pad_shape = (*v.shape[:2], pad_tokens, v.shape[3])
                padded_keys.append(
                    mx.concatenate([k, mx.zeros(k_pad_shape, dtype=k.dtype)], axis=2)
                )
                padded_values.append(
                    mx.concatenate([v, mx.zeros(v_pad_shape, dtype=v.dtype)], axis=2)
                )
            return padded_keys, padded_values

        @staticmethod
        def _contiguous_ranges(indices: Sequence[int]) -> List[Tuple[int, int]]:
            if not indices:
                return []
            ranges: List[Tuple[int, int]] = []
            start = prev = int(indices[0])
            for idx_raw in indices[1:]:
                idx = int(idx_raw)
                if idx == prev + 1:
                    prev = idx
                    continue
                ranges.append((start, prev + 1))
                start = prev = idx
            ranges.append((start, prev + 1))
            return ranges

        def _snapshot_exact_cache_entry(
            self,
            c: Any,
            prefix: str,
            arrays: dict[str, mx.array],
            metadata: dict[str, str],
        ) -> bool:
            from mlx_lm.models import cache as lm_cache

            if isinstance(c, lm_cache.KVCache):
                off = int(getattr(c, "offset", 0) or 0)
                metadata[f"{prefix}_kind"] = "kv"
                metadata[f"{prefix}_offset"] = str(off)
                metadata[f"{prefix}_step"] = str(
                    int(getattr(c, "step", getattr(type(c), "step", 256)) or 0)
                )
                if c.keys is None or c.values is None or off <= 0:
                    metadata[f"{prefix}_empty"] = "1"
                    return True
                arrays[f"{prefix}_k"] = c.keys[..., :off, :]
                arrays[f"{prefix}_v"] = c.values[..., :off, :]
                return True

            if isinstance(c, lm_cache.RotatingKVCache):
                metadata[f"{prefix}_kind"] = "rotating_kv"
                metadata[f"{prefix}_keep"] = str(int(getattr(c, "keep", 0) or 0))
                metadata[f"{prefix}_max_size"] = str(int(getattr(c, "max_size")))
                metadata[f"{prefix}_offset"] = str(int(getattr(c, "offset", 0) or 0))
                metadata[f"{prefix}_idx"] = str(int(getattr(c, "_idx", 0) or 0))
                if c.keys is None or c.values is None:
                    metadata[f"{prefix}_empty"] = "1"
                    return True
                arrays[f"{prefix}_k"] = c.keys
                arrays[f"{prefix}_v"] = c.values
                return True

            if isinstance(c, lm_cache.ChunkedKVCache):
                metadata[f"{prefix}_kind"] = "chunked_kv"
                metadata[f"{prefix}_chunk_size"] = str(int(getattr(c, "chunk_size")))
                metadata[f"{prefix}_offset"] = str(int(getattr(c, "offset", 0) or 0))
                metadata[f"{prefix}_start_position"] = str(
                    int(getattr(c, "start_position", 0) or 0)
                )
                if c.keys is None or c.values is None:
                    metadata[f"{prefix}_empty"] = "1"
                    return True
                arrays[f"{prefix}_k"] = c.keys
                arrays[f"{prefix}_v"] = c.values
                return True

            if isinstance(c, lm_cache.ArraysCache):
                metadata[f"{prefix}_kind"] = "arrays"
                metadata[f"{prefix}_size"] = str(len(c.cache))
                for j, state in enumerate(c.cache):
                    if state is None:
                        metadata[f"{prefix}_s{j}_none"] = "1"
                    else:
                        arrays[f"{prefix}_s{j}"] = state
                if c.left_padding is not None:
                    arrays[f"{prefix}_left_padding"] = c.left_padding
                if c.lengths is not None:
                    arrays[f"{prefix}_lengths"] = c.lengths
                return True

            if isinstance(c, lm_cache.CacheList):
                metadata[f"{prefix}_kind"] = "cache_list"
                metadata[f"{prefix}_size"] = str(len(c.caches))
                return all(
                    self._snapshot_exact_cache_entry(
                        sub_c, f"{prefix}_e{j}", arrays, metadata
                    )
                    for j, sub_c in enumerate(c.caches)
                )

            if isinstance(c, tuple):
                metadata[f"{prefix}_kind"] = "tuple"
                metadata[f"{prefix}_size"] = str(len(c))
                return all(
                    self._snapshot_exact_cache_entry(
                        sub_c, f"{prefix}_e{j}", arrays, metadata
                    )
                    for j, sub_c in enumerate(c)
                )

            return False

        def _write_exact_cache_snapshot(
            self,
            path: Path,
            snapshot: _DiskExactCacheSnapshot,
        ) -> List[int]:
            metadata: dict[str, str] = {
                "layout": "exact_cache_v1",
                "schema_version": APC_DISK_SCHEMA_VERSION,
                "cache_hash": str(int(snapshot.cache_hash)),
                "extra_hash": str(int(snapshot.extra_hash)),
                "token_ids": ",".join(str(int(t)) for t in snapshot.token_ids),
                "num_entries": str(len(snapshot.prompt_cache)),
                "store_id": self._exact_id_for(snapshot.cache_hash),
            }
            arrays: dict[str, mx.array] = {}
            for i, c in enumerate(snapshot.prompt_cache):
                if not self._snapshot_exact_cache_entry(c, f"c{i}", arrays, metadata):
                    raise ValueError(f"unsupported exact-cache entry at index {i}")
            if not arrays:
                return []

            # ---- ds4-style next-token logits snapshot ----
            # Store the log-softmax vector for the first token after this prefix
            # so that a future restore can skip one decode step.
            if snapshot.logits is not None:
                try:
                    lgt = snapshot.logits
                    if isinstance(lgt, mx.array):
                        lgt = lgt.reshape(-1)           # ensure 1-D (vocab_size,)
                        arrays["next_logits"] = lgt[None]  # save as (1, vocab_size)
                        metadata["has_logits"] = "1"
                        metadata["logits_vocab_size"] = str(int(lgt.shape[0]))
                except Exception as _le:
                    logger.debug("APC disk: skipping logits save: %s", _le)

            mx.eval(list(arrays.values()))
            tag = f"{os.getpid()}-{threading.get_ident()}"
            tmp = path.parent / f"{path.stem}.{tag}{self.SUFFIX}"
            mx.save_safetensors(str(tmp), arrays, metadata=metadata)
            os.replace(tmp, path)
            try:
                self._disk_bytes += path.stat().st_size
            except OSError:
                pass
            with self._index_lock:
                self._exact_index[int(snapshot.cache_hash)] = path
            self._maybe_evict()
            return [int(snapshot.cache_hash)]

        def _write_layer_major_snapshot(
            self,
            path: Path,
            snapshot: _DiskLayerMajorSnapshot,
        ) -> List[int]:
            blocks = snapshot.blocks
            if not blocks:
                return []
            if len(snapshot.layer_keys) != len(snapshot.layer_values):
                raise ValueError("layer-major disk snapshot has mismatched K/V layers")

            metadata: dict[str, str] = {}
            metadata["schema_version"] = APC_DISK_SCHEMA_VERSION
            metadata["store_id"] = snapshot.store_id
            metadata["segment_index"] = str(int(snapshot.segment_index))
            metadata["segment_count"] = str(int(snapshot.segment_count))
            for idx, b in enumerate(blocks):
                metadata[f"b{idx}_meta"] = json.dumps(
                    {
                        "block_hash": int(b.block_hash),
                        "parent_hash": int(b.parent_hash),
                        "extra_hash": int(b.extra_hash),
                        "token_ids": [int(t) for t in b.token_ids],
                    }
                )

            ranges = self._contiguous_ranges([b.source_block_idx for b in blocks])
            layer_keys: List[mx.array] = []
            layer_values: List[mx.array] = []
            bs = int(snapshot.block_size)
            for k_src, v_src in zip(snapshot.layer_keys, snapshot.layer_values):
                k_parts = [k_src[..., start * bs : end * bs, :] for start, end in ranges]
                v_parts = [v_src[..., start * bs : end * bs, :] for start, end in ranges]
                layer_keys.append(
                    k_parts[0] if len(k_parts) == 1 else mx.concatenate(k_parts, axis=2)
                )
                layer_values.append(
                    v_parts[0] if len(v_parts) == 1 else mx.concatenate(v_parts, axis=2)
                )

            layer_keys, layer_values = self._pad_layer_major_arrays(
                layer_keys, layer_values
            )
            self._save_layer_major_shard(
                path, blocks, metadata, layer_keys, layer_values, bs
            )
            return [b.block_hash for b in blocks]

        def _write_block_snapshot(
            self,
            path: Path,
            blocks: List[_DiskBlockSnapshot],
        ) -> List[int]:
            metadata: dict[str, str] = {"schema_version": APC_DISK_SCHEMA_VERSION}
            num_layers = len(blocks[0].keys) if blocks and blocks[0].keys else 0
            for idx, b in enumerate(blocks):
                if b.keys is None or b.values is None:
                    continue
                metadata[f"b{idx}_meta"] = json.dumps(
                    {
                        "block_hash": int(b.block_hash),
                        "parent_hash": int(b.parent_hash),
                        "extra_hash": int(b.extra_hash),
                        "token_ids": [int(t) for t in b.token_ids],
                    }
                )
            layer_keys: List[mx.array] = []
            layer_values: List[mx.array] = []
            for l in range(num_layers):
                layer_keys.append(
                    mx.concatenate(
                        [b.keys[l] for b in blocks if b.keys is not None], axis=2
                    )
                )
                layer_values.append(
                    mx.concatenate(
                        [b.values[l] for b in blocks if b.values is not None], axis=2
                    )
                )
            layer_keys, layer_values = self._pad_layer_major_arrays(
                layer_keys, layer_values
            )
            block_size = len(blocks[0].token_ids) if blocks and blocks[0].token_ids else 0
            self._save_layer_major_shard(
                path, blocks, metadata, layer_keys, layer_values, block_size
            )
            return [b.block_hash for b in blocks]

        def _save_layer_major_shard(
            self,
            path: Path,
            blocks: Sequence[Any],
            metadata: dict[str, str],
            layer_keys: List[mx.array],
            layer_values: List[mx.array],
            block_size: int,
        ) -> None:
            arrays: dict[str, mx.array] = {}
            for l, (k, v) in enumerate(zip(layer_keys, layer_values)):
                arrays[f"k{l}"] = k
                arrays[f"v{l}"] = v
            metadata["layout"] = "layer_major_v2"
            metadata["block_hashes"] = ",".join(str(int(b.block_hash)) for b in blocks)
            metadata["num_layers"] = str(len(layer_keys))
            metadata["block_size"] = str(int(block_size))
            mx.eval(list(arrays.values()))
            # mx.save_safetensors only accepts ".safetensors"; route
            # the temp through a sibling that retains the suffix.
            tag = f"{os.getpid()}-{threading.get_ident()}"
            tmp = path.parent / f"{path.stem}.{tag}{self.SUFFIX}"
            mx.save_safetensors(str(tmp), arrays, metadata=metadata)
            os.replace(tmp, path)
            try:
                self._disk_bytes += path.stat().st_size
            except OSError:
                pass
            with self._index_lock:
                for idx, b in enumerate(blocks):
                    self._index[int(b.block_hash)] = (path, idx)
            self._maybe_evict()

        def _writer_loop(self) -> None:
            while True:
                item = self._q.get()
                if item is None:
                    self._q.task_done()
                    break
                shard_id, block_hashes, payload, ev = item
                path = self._shard_path(shard_id)
                try:
                    if isinstance(payload, _DiskExactCacheSnapshot):
                        block_hashes = self._write_exact_cache_snapshot(path, payload)
                    elif isinstance(payload, _DiskLayerMajorSnapshot):
                        block_hashes = self._write_layer_major_snapshot(path, payload)
                    else:
                        block_hashes = self._write_block_snapshot(path, payload)
                except Exception as e:
                    logger.warning("APC disk shard save failed for %s: %s", path, e)
                finally:
                    with self._in_flight_lock:
                        for block_hash in block_hashes:
                            self._in_flight.pop(int(block_hash), None)
                    ev.set()
                    self._q.task_done()
                    # Layer-major segment payloads share references to the full
                    # source KV cache. Drop the last processed payload promptly
                    # instead of retaining it in this thread's frame until the
                    # next queue item arrives.
                    payload = None
                    item = None

