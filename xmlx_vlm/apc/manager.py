from __future__ import annotations
import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Iterable, List, Optional, Sequence, Tuple
import mlx.core as mx

from .types import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_NUM_BLOCKS,
    SEED_PARENT_HASH,
    APCBlock,
    APCExactCacheEntry,
    APCStats,
    _hash_tokens,
    _sequence_hash,
    _DiskLayerMajorBlock,
)
from .disk_store import DiskBlockStore, _free_ram_bytes

logger = logging.getLogger("xmlx_vlm.apc")

class APCManager:
    """Block pool, hash table, LRU free queue, and stats."""

    def __init__(
        self,
        num_blocks: int = DEFAULT_NUM_BLOCKS,
        block_size: int = DEFAULT_BLOCK_SIZE,
        disk: Optional["DiskBlockStore"] = None,
    ):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.pool: List[APCBlock] = [APCBlock(block_id=i) for i in range(num_blocks)]
        self._free_head: Optional[APCBlock] = None
        self._free_tail: Optional[APCBlock] = None
        for b in self.pool:
            self._free_push(b)
        self.hash_table: dict[int, APCBlock] = {}
        self._exact_cache: "OrderedDict[int, APCExactCacheEntry]" = OrderedDict()
        self.stats = APCStats()
        self.lock = threading.RLock()
        self.disk = disk
        self._exact_cache_max = max(
            0, int(os.environ.get("APC_EXACT_CACHE_ENTRIES", "2"))
        )
        self.exact_cache_guard_tokens = max(
            1, int(os.environ.get("APC_EXACT_PREFIX_GUARD_TOKENS", "16"))
        )
        # If free RAM (best-effort reading) drops below this, skip disk
        # promotion this turn and fall back to memory-only matching. The
        # request still serves correctly — it just doesn't get the warm-
        # disk speed-up. Disabled when set to 0.
        self._disk_min_free_ram_bytes = int(
            float(os.environ.get("APC_DISK_MIN_FREE_RAM_GB", "2.0")) * (1 << 30)
        )
        # Number of disk-loaded blocks to coalesce per ``mx.eval`` during
        # warm-disk restore. The disk read itself is always serial (no
        # thread pool, no buffering of mmap views beyond this batch); the
        # batch only controls eval-dispatch count. With Qwen3-VL-4B's 36
        # layers × bf16 head_dim=128 × block_size=16 × 8 KV-heads, one
        # block of K+V is ~2.3 MB, so the default of 8 puts at most ~18 MB
        # of fresh-block tensors in flight per eval — three orders of
        # magnitude below the all-at-once eval that has crashed Apple
        # Silicon hosts. Set to 1 for the strictly-bounded one-at-a-time
        # path; raise it on a known-roomy machine to claw back wall time.
        self._disk_eval_block_chunk = max(
            1, int(os.environ.get("APC_DISK_EVAL_BLOCK_CHUNK", "8"))
        )
        # Number of disk blocks to coalesce into one direct byte-range read.
        # This is separate from eval chunking: a larger read chunk improves
        # SSD throughput/readahead while eval still happens in small batches.
        # 256 Qwen3-VL-4B blocks are ~576 MB of K/V payload; large enough to
        # restore an ~8k-token prompt shard in one sequential read, still
        # small relative to the model's recommended Apple-Silicon working set.
        self._disk_load_block_chunk = max(
            1, int(os.environ.get("APC_DISK_LOAD_BLOCK_CHUNK", "256"))
        )
        # Apple Metal has a per-process resource-count ceiling separate from
        # byte memory. Qwen3-VL-4B stores 72 MLX tensors per APCBlock, so a
        # very large pool can hit the ceiling before unified memory is scarce.
        # Keep disk persistence going, but stop adding memory-pool blocks near
        # the resource limit. Set to 0 to disable.
        self._max_pool_tensors = max(
            0, int(os.environ.get("APC_MAX_POOL_TENSORS", "450000"))
        )
        # Optional compact warm-memory tier for long KV-only prefixes. When a
        # prompt reaches this many full-block tokens, store one layer-major
        # prompt-cache snapshot instead of thousands of per-block tensors. This
        # avoids Apple Metal's resource-count ceiling while preserving fast
        # warm-memory reuse for repeated long-document prompts.
        self._layer_major_memory_min_tokens = max(
            0, int(os.environ.get("APC_LAYER_MAJOR_MEMORY_MIN_TOKENS", "50000"))
        )

    # ---------- LRU free queue (O(1)) ----------
    def _free_push(self, b: APCBlock) -> None:
        b.prev = self._free_tail
        b.next = None
        if self._free_tail is not None:
            self._free_tail.next = b
        else:
            self._free_head = b
        self._free_tail = b
        b.last_used = time.time()

    def _free_remove(self, b: APCBlock) -> None:
        if b.prev is not None:
            b.prev.next = b.next
        else:
            self._free_head = b.next
        if b.next is not None:
            b.next.prev = b.prev
        else:
            self._free_tail = b.prev
        b.prev = b.next = None

    # ---------- Block lifecycle ----------
    def _evict_lru(self) -> Optional[APCBlock]:
        b = self._free_head
        if b is None:
            return None
        self._free_remove(b)
        if b.block_hash is not None and self.hash_table.get(b.block_hash) is b:
            del self.hash_table[b.block_hash]
            self.stats.evictions += 1
        b.block_hash = None
        b.token_ids = ()
        b.parent_hash = SEED_PARENT_HASH
        b.extra_hash = 0
        b.keys = None
        b.values = None
        return b

    def _acquire_existing(self, b: APCBlock) -> APCBlock:
        if b.ref_cnt == 0:
            self._free_remove(b)
        b.ref_cnt += 1
        return b

    def _release_one(self, b: APCBlock) -> None:
        b.ref_cnt -= 1
        if b.ref_cnt <= 0:
            b.ref_cnt = 0
            self._free_push(b)

    def release(self, blocks: Iterable[APCBlock]) -> None:
        with self.lock:
            for b in blocks:
                self._release_one(b)

    # ---------- Public API ----------
    def lookup_exact_cache(
        self,
        token_ids: Sequence[int],
        extra_hash: int = 0,
        max_prefix_tokens: Optional[int] = None,
        min_prefix_tokens: int = 0,
    ) -> Tuple[Optional[List[Any]], int, Optional[mx.array]]:
        """Return an exact-prefix prompt-cache snapshot for custom caches.

        Mixed architectures such as Nemotron-H use recurrent SSM state in
        addition to attention KV. That state is not block-concatenable, so the
        safe reuse unit is an exact prompt-cache snapshot at a prefix boundary.

        Returns ``(prompt_cache, prefix_len, logits)`` where ``logits`` is the
        saved log-softmax vector (vocab_size,) for the first token after the
        cached prefix (ds4-style session checkpoint), or ``None`` when not
        available.  Callers that previously unpacked 2-tuples should add a
        third ``_`` or ``logits`` binding.
        """
        disk = self.disk
        if self._exact_cache_max <= 0 and disk is None:
            return None, 0, None
        token_tuple = tuple(int(t) for t in token_ids)
        max_len = len(token_tuple) - 1
        if max_prefix_tokens is not None and max_prefix_tokens > 0:
            max_len = min(max_len, int(max_prefix_tokens))
        if max_len <= min_prefix_tokens:
            return None, 0, None

        source_cache: Optional[List[Any]] = None
        source_logits: Optional[mx.array] = None
        prefix_len = 0
        with self.lock:
            best_key: Optional[int] = None
            best_entry: Optional[APCExactCacheEntry] = None
            if self._exact_cache_max > 0:
                for key, entry in self._exact_cache.items():
                    candidate_len = len(entry.token_ids)
                    if (
                        entry.extra_hash != extra_hash
                        or candidate_len <= min_prefix_tokens
                        or candidate_len > max_len
                    ):
                        continue
                    if token_tuple[:candidate_len] != entry.token_ids:
                        continue
                    if best_entry is None or candidate_len > len(best_entry.token_ids):
                        best_key = key
                        best_entry = entry

                if best_entry is not None and best_key is not None:
                    self._exact_cache.move_to_end(best_key)
                    best_entry.last_used = time.time()
                    prefix_len = len(best_entry.token_ids)
                    source_cache = best_entry.prompt_cache
                    source_logits = best_entry.logits

        can_try_disk = disk is not None and prefix_len < max_len
        if can_try_disk and self._disk_min_free_ram_bytes > 0:
            import xmlx_vlm.apc as apc
            free_now = apc._free_ram_bytes()
            if free_now is not None and free_now < self._disk_min_free_ram_bytes:
                logger.info(
                    "APC: skipping exact disk restore " "(free RAM %.1f GB < %.1f GB)",
                    free_now / (1 << 30),
                    self._disk_min_free_ram_bytes / (1 << 30),
                )
                can_try_disk = False

        if can_try_disk and disk is not None:
            disk_match = disk.find_exact_prefix(
                token_tuple,
                extra_hash=extra_hash,
                max_prefix_tokens=max_prefix_tokens,
                min_prefix_tokens=max(min_prefix_tokens, prefix_len),
            )
            if disk_match is not None:
                cache_hash, disk_prefix_len = disk_match
                loaded = disk.load_exact_cache(
                    cache_hash,
                    min_capacity_tokens=len(token_tuple) + 1,
                )
                if loaded is not None:
                    stored_tokens, stored_extra_hash, prompt_cache, disk_logits = loaded
                    if (
                        stored_extra_hash == extra_hash
                        and len(stored_tokens) == disk_prefix_len
                        and token_tuple[:disk_prefix_len] == stored_tokens
                    ):
                        # Promote the disk hit into the in-memory exact cache
                        # so the next request for this prefix avoids another disk
                        # read. The clone is sized to the original prompt length;
                        # callers still need to resize it for their own prompt.
                        with self.lock:
                            if self._exact_cache_max > 0:
                                self._exact_cache[cache_hash] = APCExactCacheEntry(
                                    token_ids=stored_tokens,
                                    extra_hash=int(extra_hash),
                                    prompt_cache=prompt_cache,
                                    last_used=time.time(),
                                    logits=disk_logits,
                                )
                                self._exact_cache.move_to_end(cache_hash)
                                while (
                                    len(self._exact_cache)
                                    > self._exact_cache_max
                                ):
                                    self._exact_cache.popitem(last=False)
                        # Disk reads and warm-cache construction intentionally
                        # happen outside the manager lock. If clear()/reset_stats()
                        # races here, the restored tensors are still valid; only
                        # the hit counter lands in the new stats window.
                        with self.lock:
                            self.stats.exact_hits += 1
                            self.stats.disk_hits += 1
                            self.stats.hits += 1
                            self.stats.matched_tokens += disk_prefix_len
                        return prompt_cache, disk_prefix_len, disk_logits

        if source_cache is None:
            return None, 0, None
        from .utils import _clone_prompt_cache_for_apc
        prompt_cache = _clone_prompt_cache_for_apc(
            source_cache,
            min_capacity_tokens=len(token_tuple) + 1,
        )
        if prompt_cache is None:
            return None, 0, None
        with self.lock:
            self.stats.exact_hits += 1
            self.stats.hits += 1
            self.stats.matched_tokens += prefix_len
        return prompt_cache, prefix_len, source_logits

    def store_exact_cache(
        self,
        token_ids: Sequence[int],
        prompt_cache: Sequence[Any],
        *,
        extra_hash: int = 0,
        logits: Optional[mx.array] = None,
    ) -> bool:
        """Store a full prompt-cache snapshot for exact-prefix reuse.

        ``logits`` is the optional log-softmax vector (vocab_size,) for the
        first token *after* this prefix — the ds4-style session-checkpoint
        logits that allow skipping one decode step on restore.
        """
        if (self._exact_cache_max <= 0 and self.disk is None) or not token_ids:
            return False
        token_tuple = tuple(int(t) for t in token_ids)
        from .utils import _clone_prompt_cache_for_apc
        copied = _clone_prompt_cache_for_apc(prompt_cache)
        if copied is None:
            return False
        key = _sequence_hash(token_tuple, extra_hash, self.block_size)
        stored = False
        with self.lock:
            if self._exact_cache_max > 0:
                self._exact_cache[key] = APCExactCacheEntry(
                    token_ids=token_tuple,
                    extra_hash=int(extra_hash),
                    prompt_cache=copied,
                    last_used=time.time(),
                    logits=logits,
                )
                self._exact_cache.move_to_end(key)
                while len(self._exact_cache) > self._exact_cache_max:
                    self._exact_cache.popitem(last=False)
                stored = True
        if self.disk is not None:
            try:
                self.disk.save_exact_cache(
                    key, token_tuple, extra_hash, copied, logits=logits
                )
                with self.lock:
                    self.stats.disk_writes += 1
                stored = True
            except Exception as e:
                logger.warning("APC exact disk save scheduling failed: %s", e)
        if stored:
            with self.lock:
                self.stats.exact_stores += 1
            return True
        return False

    def lookup_prefix_disk_cache(
        self,
        token_ids: Sequence[int],
        extra_hash: int = 0,
        max_prefix_tokens: Optional[int] = None,
        min_prefix_tokens: int = 0,
        allow_memory_overlap: bool = False,
    ) -> Tuple[Optional[List[Any]], int]:
        """Return a ready prompt cache from a layer-major disk shard.

        This is the warm-disk fast path. It deliberately does not promote
        individual APCBlock slabs into the memory pool; it restores the prefix
        as one per-layer K/V tensor set, matching what generation consumes.
        """
        disk = self.disk
        if disk is None:
            return None, 0
        with self.lock:
            if self._disk_min_free_ram_bytes > 0:
                import xmlx_vlm.apc as apc
                free_now = apc._free_ram_bytes()
                if free_now is not None and free_now < self._disk_min_free_ram_bytes:
                    logger.info(
                        "APC: skipping disk prompt-cache restore "
                        "(free RAM %.1f GB < %.1f GB)",
                        free_now / (1 << 30),
                        self._disk_min_free_ram_bytes / (1 << 30),
                    )
                    return None, 0

            n_full = len(token_ids) // self.block_size
            if max_prefix_tokens is not None and max_prefix_tokens > 0:
                n_full = min(n_full, int(max_prefix_tokens) // self.block_size)
            parent = SEED_PARENT_HASH
            block_hashes: List[int] = []
            chunks: List[Tuple[int, ...]] = []
            for i in range(n_full):
                chunk = tuple(
                    int(t)
                    for t in token_ids[i * self.block_size : (i + 1) * self.block_size]
                )
                h = _hash_tokens(parent, chunk, extra_hash)
                # If the prefix is already in memory, the normal memory path is
                # better and preserves the expected ref-count lifecycle.
                b_mem = self.hash_table.get(h)
                if (
                    not allow_memory_overlap
                    and b_mem is not None
                    and b_mem.token_ids == chunk
                ):
                    return None, 0
                if not disk.has(h):
                    break
                block_hashes.append(h)
                chunks.append(chunk)
                parent = h

            if not block_hashes:
                return None, 0
            matched_tokens = len(block_hashes) * self.block_size
            if matched_tokens <= min_prefix_tokens:
                return None, 0

        loaded = disk.load_layer_major_prefix(block_hashes)
        if loaded is None:
            return None, 0
        keys, values, metadatas = loaded
        if len(metadatas) != len(chunks):
            return None, 0
        for chunk, metadata in zip(chunks, metadatas):
            try:
                stored_tokens = tuple(
                    int(x) for x in metadata.get("token_ids", "").split(",") if x
                )
                stored_extra = int(metadata.get("extra_hash", "0"))
            except (TypeError, ValueError):
                return None, 0
            if stored_tokens != chunk or stored_extra != extra_hash:
                return None, 0

        from .utils import make_warm_kv_cache_from_layers
        warm_cache = make_warm_kv_cache_from_layers(keys, values, matched_tokens)
        # Disk reads and warm-cache construction intentionally happen outside
        # the manager lock. If clear()/reset_stats() races here, the restored
        # tensors are still valid; only the hit counter lands in the new stats
        # window.
        with self.lock:
            self.stats.disk_hits += len(block_hashes)
            self.stats.hits += 1
            self.stats.matched_tokens += matched_tokens
        return warm_cache, matched_tokens

    def lookup_prefix(
        self, token_ids: Sequence[int], extra_hash: int = 0
    ) -> Tuple[List[APCBlock], int]:
        """Walk the hash chain over ``token_ids``; return acquired matched
        blocks and matched_token_count. Caller must release the blocks.

        This memory-only path stops at the first block that is not already
        present in the in-process APCBlock pool.
        """
        with self.lock:
            n_full = len(token_ids) // self.block_size
            matched: List[APCBlock] = []
            parent = SEED_PARENT_HASH
            for i in range(n_full):
                chunk = tuple(
                    int(t)
                    for t in token_ids[i * self.block_size : (i + 1) * self.block_size]
                )
                h = _hash_tokens(parent, chunk, extra_hash)
                b_mem = self.hash_table.get(h)
                if b_mem is None or b_mem.token_ids != chunk:
                    break
                matched.append(self._acquire_existing(b_mem))
                parent = h

            matched_tokens = len(matched) * self.block_size
            if matched_tokens > 0:
                self.stats.hits += 1
                self.stats.matched_tokens += matched_tokens
            else:
                self.stats.misses += 1
            return matched, matched_tokens

    def store_kv_blocks(
        self,
        token_ids: Sequence[int],
        layer_keys: List[mx.array],
        layer_values: List[mx.array],
        *,
        extra_hash: int = 0,
        skip_first_n_tokens: int = 0,
    ) -> List[APCBlock]:
        """Slice ``layer_keys`` / ``layer_values`` into block_size chunks and
        store any new full blocks beyond ``skip_first_n_tokens``.

        Returns newly acquired blocks (caller must release).
        """
        with self.lock:
            n_full = len(token_ids) // self.block_size
            skip_full = skip_first_n_tokens // self.block_size
            full_prefix_tokens = n_full * self.block_size
            guarded_prefix_tokens = max(
                0, len(token_ids) - self.exact_cache_guard_tokens
            )
            layer_major_prefix_tokens = min(
                full_prefix_tokens,
                (guarded_prefix_tokens // self.block_size) * self.block_size,
            )
            new_blocks: List[APCBlock] = []
            disk_blocks: List[_DiskLayerMajorBlock] = []
            per_block_tensors = len(layer_keys) + len(layer_values)
            token_tuple = tuple(int(t) for t in token_ids[:layer_major_prefix_tokens])
            layer_major_stored = False
            if (
                self._layer_major_memory_min_tokens > 0
                and self._exact_cache_max > 0
                and layer_major_prefix_tokens >= self._layer_major_memory_min_tokens
            ):
                from .utils import _clone_layer_major_kv_cache_for_apc
                copied = _clone_layer_major_kv_cache_for_apc(
                    layer_keys,
                    layer_values,
                    layer_major_prefix_tokens,
                )
                if copied is not None:
                    key = _sequence_hash(token_tuple, extra_hash, self.block_size)
                    self._exact_cache[key] = APCExactCacheEntry(
                        token_ids=token_tuple,
                        extra_hash=int(extra_hash),
                        prompt_cache=copied,
                        last_used=time.time(),
                    )
                    self._exact_cache.move_to_end(key)
                    while len(self._exact_cache) > self._exact_cache_max:
                        self._exact_cache.popitem(last=False)
                    self.stats.exact_stores += 1
                    layer_major_stored = True
            parent = SEED_PARENT_HASH
            # Recompute hash chain over already-cached prefix to get parent for first new block.
            for i in range(skip_full):
                chunk = tuple(
                    int(t)
                    for t in token_ids[i * self.block_size : (i + 1) * self.block_size]
                )
                parent = _hash_tokens(parent, chunk, extra_hash)

            for i in range(skip_full, n_full):
                chunk = tuple(
                    int(t)
                    for t in token_ids[i * self.block_size : (i + 1) * self.block_size]
                )
                h = _hash_tokens(parent, chunk, extra_hash)
                if self.disk is not None and not self.disk.has(h):
                    disk_blocks.append(
                        _DiskLayerMajorBlock(
                            block_hash=int(h),
                            parent_hash=int(parent),
                            extra_hash=int(extra_hash),
                            token_ids=chunk,
                            source_block_idx=i,
                        )
                    )
                if layer_major_stored:
                    parent = h
                    continue
                existing = self.hash_table.get(h)
                if existing is not None and existing.token_ids == chunk:
                    acquired = self._acquire_existing(existing)
                    new_blocks.append(acquired)
                    parent = h
                    continue
                if (
                    self._max_pool_tensors > 0
                    and per_block_tensors > 0
                    and (len(self.hash_table) + 1) * per_block_tensors
                    > self._max_pool_tensors
                ):
                    logger.debug(
                        "APC pool tensor limit reached; skipping memory store "
                        "at block %d/%d",
                        i,
                        n_full,
                    )
                    if self.disk is None:
                        break
                    parent = h
                    continue
                b = self._evict_lru()
                if b is None:
                    logger.debug(
                        "APC pool exhausted; skipping memory store at block %d/%d",
                        i,
                        n_full,
                    )
                    if self.disk is None:
                        break
                    parent = h
                    continue
                start = i * self.block_size
                end = start + self.block_size
                # Deep-copy each slice into its own buffer so the block tensor
                # is decoupled from the caller's cache, which mlx.clear_cache
                # may release after generation. mx.contiguous alone can return
                # a view when the source is already row-contiguous.
                from .utils import _copy_mlx_array
                k_slabs = [_copy_mlx_array(k[..., start:end, :]) for k in layer_keys]
                v_slabs = [_copy_mlx_array(v[..., start:end, :]) for v in layer_values]
                mx.eval(k_slabs + v_slabs)
                b.block_hash = h
                b.parent_hash = parent
                b.token_ids = chunk
                b.extra_hash = extra_hash
                b.keys = k_slabs
                b.values = v_slabs
                b.ref_cnt = 1
                self.hash_table[h] = b
                new_blocks.append(b)
                self.stats.stores += 1
                self.stats.served_tokens += self.block_size
                parent = h
            if self.disk is not None and disk_blocks:
                try:
                    self.disk.save_layer_major_blocks(
                        disk_blocks, layer_keys, layer_values, self.block_size
                    )
                    self.stats.disk_writes += len(disk_blocks)
                except Exception as e:
                    logger.warning("APC disk save scheduling failed: %s", e)
            self.stats.pool_used = sum(1 for x in self.pool if x.block_hash is not None)
            return new_blocks

    def stats_snapshot(self) -> dict:
        with self.lock:
            self.stats.pool_used = sum(1 for x in self.pool if x.block_hash is not None)
            snap = self.stats.snapshot(self.num_blocks, self.block_size)
            if self.disk is not None:
                snap["disk_bytes"] = self.disk.disk_bytes
                snap["disk_max_bytes"] = self.disk.max_bytes
                snap["disk_evictions"] = self.disk.evictions
                # files-on-disk count + indexed-block count
                try:
                    snap["disk_files"] = sum(
                        1
                        for p in self.disk.dir.glob(f"*{self.disk.SUFFIX}")
                        if self.disk._is_canonical_store_file(p)
                    )
                except OSError:
                    snap["disk_files"] = -1
                snap["disk_blocks_indexed"] = self.disk.num_blocks_indexed
                snap["disk_exact_indexed"] = self.disk.num_exact_indexed
            return snap

    def reset_stats(self) -> None:
        with self.lock:
            self.stats = APCStats()

    def clear(self) -> None:
        with self.lock:
            for b in self.pool:
                b.block_hash = None
                b.token_ids = ()
                b.parent_hash = SEED_PARENT_HASH
                b.extra_hash = 0
                b.keys = None
                b.values = None
                b.ref_cnt = 0
                b.prev = b.next = None
            self.hash_table.clear()
            self._free_head = self._free_tail = None
            for b in self.pool:
                self._free_push(b)
            self._exact_cache.clear()
            self.stats = APCStats()

    def flush_to_disk(self) -> int:
        """Write all unreferenced in-memory blocks to the disk tier.

        This is useful for agent workloads where the server may restart
        between turns: flushing after a generation ensures the prefix
        cache survives process restarts without waiting for LRU eviction.

        Returns the number of blocks scheduled for disk write.
        """
        if self.disk is None:
            return 0
        with self.lock:
            blocks_to_flush: List[APCBlock] = []
            for b in self.pool:
                if (
                    b.block_hash is not None
                    and b.ref_cnt == 0
                    and b.keys is not None
                    and b.values is not None
                    and not self.disk.has(b.block_hash)
                ):
                    blocks_to_flush.append(b)
            if blocks_to_flush:
                self.disk.save_batch(blocks_to_flush)
                return len(blocks_to_flush)
            return 0

    def close(self) -> None:
        """Best-effort shutdown: flush blocks then close the disk writer thread."""
        self.flush_to_disk()
        if self.disk is not None:
            self.disk.close()


