from __future__ import annotations
import hashlib
import json
import logging
import os
import queue
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import mlx.core as mx

from .safetensors_ops import _safe_namespace, _read_safetensors_metadata
from .ram_ops import _free_ram_bytes
from .loader import DiskBlockStoreLoaderMixin
from .saver import DiskBlockStoreSaverMixin
from ..types import (
    APC_DISK_SCHEMA_VERSION, _DiskBlockSnapshot, _DiskLayerMajorBlock, _DiskLayerMajorSnapshot,
    _DiskExactCacheSnapshot, APCExactCacheEntry, SEED_PARENT_HASH
)

logger = logging.getLogger('xmlx_vlm.apc')

class DiskBlockStore(DiskBlockStoreLoaderMixin, DiskBlockStoreSaverMixin):
        SUFFIX = ".safetensors"

        SHARD_PREFIX = "shard_"

        EXACT_PREFIX = "exact_"

        SHARD_STEM_LEN = len(SHARD_PREFIX) + 32  # "shard_" + 32 hex chars

        EXACT_STEM_LEN = len(EXACT_PREFIX) + 32  # "exact_" + 32 hex chars

        _EVICT_LOW_WATERMARK = 0.9

        def __init__(
            self,
            root: Path,
            namespace: str = "default",
            num_workers: int = 1,
            max_bytes: Optional[int] = None,
        ):
            self.dir = Path(root) / _safe_namespace(namespace)
            self.dir.mkdir(parents=True, exist_ok=True)
            self.max_bytes = max_bytes
            self.evictions = 0  # cumulative shard deletions by _maybe_evict
            self._q: queue.Queue = queue.Queue(maxsize=4096)
            self._stop = threading.Event()
            # Track in-flight hashes (across pending shard writes) so a lookup
            # racing a write can wait briefly for the bytes to land.
            self._in_flight: dict[int, threading.Event] = {}
            self._in_flight_lock = threading.Lock()
            # block_hash -> (shard_path, block_idx_in_shard)
            self._index: dict[int, Tuple[Path, int]] = {}
            # exact full-prefix hash -> snapshot path
            self._exact_index: dict[int, Path] = {}
            self._index_lock = threading.RLock()
            # Direct-read mode avoids mmap-backed MLX arrays entirely. It parses
            # safetensors headers, reads only the requested block's byte ranges
            # with normal file I/O, then constructs MLX-managed arrays from those
            # bytes. Keep the old mmap path available for comparison.
            self._read_mode = os.environ.get("APC_DISK_READ_MODE", "direct").lower()
            if self._read_mode not in ("direct", "mmap"):
                logger.warning(
                    "APC disk: unknown APC_DISK_READ_MODE=%r; using direct",
                    self._read_mode,
                )
                self._read_mode = "direct"
            # Bounded LRU of parsed safetensors headers:
            # shard_path -> (tensor_entries, file_metadata, data_start).
            self._header_cache: "OrderedDict[Path, Tuple[dict, dict, int]]" = OrderedDict()
            self._header_cache_lock = threading.Lock()
            self._header_cache_max = int(os.environ.get("APC_DISK_HEADER_CACHE", 4))
            self._direct_max_overread_bytes = int(
                float(os.environ.get("APC_DISK_DIRECT_MAX_OVERREAD_MB", "8")) * (1 << 20)
            )
            # Bound layer-major shard size so disk eviction is segment-granular
            # instead of one huge all-or-nothing prefix file. A Qwen3-VL-4B block
            # is ~2.25 MiB, so 256 blocks is roughly a 576 MiB shard before the
            # small KV step padding.
            self._shard_max_blocks = max(
                1, int(os.environ.get("APC_DISK_SHARD_MAX_BLOCKS", "256"))
            )
            # Layer-major warm-disk restore concatenates segment shards one layer
            # at a time. Clearing MLX's allocator cache after each layer keeps the
            # temporary segment tensors from coexisting with the fully-restored
            # prompt cache. Set to 0 to disable, or a larger value to trade peak
            # memory for slightly lower restore overhead.
            self._restore_clear_every = max(
                0, int(os.environ.get("APC_DISK_RESTORE_CLEAR_EVERY", "1"))
            )
            # Bounded LRU of mmap'd shards: shard_path -> (arrays_dict, file_metadata).
            # Default capped at 2 — the within-restore working set is typically
            # one shard, occasionally two (for a multi-shard restore). Larger
            # caps risk pinning lots of materialised K/V tensors in unified
            # memory after evicted blocks have already been used. Override with
            # APC_DISK_MMAP_CACHE if you know what you're doing.
            self._mmap_cache: "OrderedDict[Path, Tuple[dict, dict]]" = OrderedDict()
            self._mmap_cache_lock = threading.Lock()
            self._mmap_cache_max = int(os.environ.get("APC_DISK_MMAP_CACHE", 2))

            n_orphans = self._cleanup_partials()
            if n_orphans:
                logger.info(
                    "APC disk: removed %d orphaned partial file(s) from %s",
                    n_orphans,
                    self.dir,
                )
            # Build index from existing shards and compute current byte usage.
            self._disk_bytes = self._rebuild_index()

            self._workers = [
                threading.Thread(
                    target=self._writer_loop, daemon=True, name=f"apc-disk-{i}"
                )
                for i in range(max(1, num_workers))
            ]
            for t in self._workers:
                t.start()

        @classmethod
        def _is_canonical_shard(cls, path: Path) -> bool:
            stem = path.stem
            if not stem.startswith(cls.SHARD_PREFIX):
                return False
            rest = stem[len(cls.SHARD_PREFIX) :]
            if len(rest) != 32:
                return False
            return all(c in "0123456789abcdef" for c in rest)

        @classmethod
        def _is_canonical_exact(cls, path: Path) -> bool:
            stem = path.stem
            if not stem.startswith(cls.EXACT_PREFIX):
                return False
            rest = stem[len(cls.EXACT_PREFIX) :]
            if len(rest) != 32:
                return False
            return all(c in "0123456789abcdef" for c in rest)

        @classmethod
        def _is_canonical_store_file(cls, path: Path) -> bool:
            return cls._is_canonical_shard(path) or cls._is_canonical_exact(path)

        def _shard_path(self, shard_id: str) -> Path:
            return self.dir / f"{shard_id}{self.SUFFIX}"

        @staticmethod
        def _shard_id_for(block_hashes: Sequence[int]) -> str:
            h = hashlib.sha256()
            for bh in block_hashes:
                h.update(int(bh & ((1 << 64) - 1)).to_bytes(8, "little"))
            return f"{DiskBlockStore.SHARD_PREFIX}{h.hexdigest()[:32]}"

        @staticmethod
        def _exact_id_for(cache_hash: int) -> str:
            h = hashlib.sha256()
            h.update(int(cache_hash & ((1 << 64) - 1)).to_bytes(8, "little"))
            return f"{DiskBlockStore.EXACT_PREFIX}{h.hexdigest()[:32]}"

        def _cleanup_partials(self) -> int:
            """Delete anything in the dir that isn't a canonical shard (left
            over from a crashed write, or files from an older block-per-file
            layout that this class no longer recognises)."""
            n = 0
            for p in self.dir.glob(f"*{self.SUFFIX}"):
                if not p.is_file() or self._is_canonical_store_file(p):
                    continue
                try:
                    p.unlink()
                    n += 1
                except OSError as e:
                    logger.warning("APC disk: failed to remove partial %s: %s", p, e)
            return n

        def _rebuild_index(self) -> int:
            """Scan shards, populate ``_index``, return total bytes on disk.

            Uses a header-only safetensors read so each shard scan touches only
            the file's leading few KB — no MLX array construction, no mmap of
            the tensor payload. On a disk with hundreds of cached shards this
            keeps server-startup overhead and Python heap growth minimal.
            """
            total = 0
            with self._index_lock:
                self._index.clear()
                self._exact_index.clear()
                for p in self.dir.glob(f"*{self.SUFFIX}"):
                    if not self._is_canonical_store_file(p):
                        continue
                    try:
                        total += p.stat().st_size
                    except OSError:
                        continue
                    metadata = _read_safetensors_metadata(p)
                    if metadata is None:
                        logger.warning("APC disk: shard %s unreadable, dropping", p)
                        try:
                            p.unlink()
                        except OSError:
                            pass
                        continue
                    if self._is_canonical_exact(p):
                        try:
                            cache_hash = int(metadata.get("cache_hash", ""))
                        except (TypeError, ValueError):
                            continue
                        self._exact_index[cache_hash] = p
                        continue
                    hashes_csv = metadata.get("block_hashes", "")
                    if not hashes_csv:
                        continue
                    try:
                        block_hashes = [int(x) for x in hashes_csv.split(",") if x]
                    except ValueError:
                        continue
                    for idx, bh in enumerate(block_hashes):
                        self._index[bh] = (p, idx)
            return total

        @property
        def disk_bytes(self) -> int:
            return self._disk_bytes

        @property
        def num_blocks_indexed(self) -> int:
            with self._index_lock:
                return len(self._index)

        @property
        def num_exact_indexed(self) -> int:
            with self._index_lock:
                return len(self._exact_index)

        @property
        def load_returns_detached(self) -> bool:
            return self._read_mode == "direct"

        def _maybe_evict(self) -> int:
            """Evict segment shards until under the low watermark.

            Stores are ordered by last-used time; segments within the same store
            are evicted tail-first so a partially-retained store still has a
            useful prefix.
            """
            if self.max_bytes is None or self._disk_bytes <= self.max_bytes:
                return 0
            target = int(self.max_bytes * self._EVICT_LOW_WATERMARK)
            # Don't evict shards whose blocks are still in-flight to other
            # callers (would race a pending writer).
            with self._in_flight_lock:
                in_flight_hashes = set(self._in_flight.keys())
            with self._index_lock:
                in_flight_paths = {
                    self._index[h][0] for h in in_flight_hashes if h in self._index
                }
                in_flight_paths.update(
                    self._exact_index[h] for h in in_flight_hashes if h in self._exact_index
                )

            candidates: list[dict[str, Any]] = []
            for p in self.dir.glob(f"*{self.SUFFIX}"):
                if not self._is_canonical_store_file(p) or p in in_flight_paths:
                    continue
                try:
                    st = p.stat()
                except OSError:
                    continue
                store_id = str(p)
                segment_index = 0
                metadata = _read_safetensors_metadata(p)
                if metadata is not None:
                    store_id = metadata.get("store_id", store_id)
                    try:
                        segment_index = int(metadata.get("segment_index", "0"))
                    except (TypeError, ValueError):
                        segment_index = 0
                candidates.append(
                    {
                        # Header reads during eviction can update atime on some
                        # filesystems. Use mtime as our explicit last-used clock;
                        # the load paths call os.utime(), which updates it.
                        "last_used": st.st_mtime,
                        "size": st.st_size,
                        "path": p,
                        "store_id": store_id,
                        "segment_index": segment_index,
                    }
                )
            store_last_used: dict[str, float] = {}
            for candidate in candidates:
                sid = candidate["store_id"]
                store_last_used[sid] = min(
                    candidate["last_used"],
                    store_last_used.get(sid, candidate["last_used"]),
                )
            candidates.sort(
                key=lambda c: (
                    store_last_used[c["store_id"]],
                    -int(c["segment_index"]),
                    c["last_used"],
                )
            )

            evicted = 0
            for candidate in candidates:
                if self._disk_bytes <= target:
                    break
                size = int(candidate["size"])
                p = candidate["path"]
                try:
                    p.unlink()
                except OSError as e:
                    logger.warning("APC disk: failed to evict %s: %s", p, e)
                    continue
                self._disk_bytes -= size
                evicted += 1
                # Drop index + mmap entries pointing at this shard.
                with self._index_lock:
                    stale = [h for h, (sp, _) in self._index.items() if sp == p]
                    for h in stale:
                        del self._index[h]
                    stale_exact = [h for h, sp in self._exact_index.items() if sp == p]
                    for h in stale_exact:
                        del self._exact_index[h]
                with self._mmap_cache_lock:
                    self._mmap_cache.pop(p, None)
                with self._header_cache_lock:
                    self._header_cache.pop(p, None)
            if evicted:
                self.evictions += evicted
                logger.info(
                    "APC disk: evicted %d shard(s); now %.1f MB / %.1f MB cap",
                    evicted,
                    self._disk_bytes / 1e6,
                    self.max_bytes / 1e6,
                )
            return evicted

        def close(self) -> None:
            self._stop.set()
            for _ in self._workers:
                self._q.put(None)
            for t in self._workers:
                t.join()
            with self._header_cache_lock:
                self._header_cache.clear()
            with self._mmap_cache_lock:
                self._mmap_cache.clear()

