"""Automatic Prefix Caching (APC) for xmlx-vlm.

Hash-based, block-level KV cache reuse across requests. The KV cache is split
into fixed-size blocks (default 16 tokens). Each fully-filled block is
identified by a chained hash::

    block_hash[i] = H(block_hash[i-1], tuple(tokens[i*bs:(i+1)*bs]), extra_hash[i])

``extra_hash[i]`` carries multimodal context (e.g. an image content hash) so
identical token IDs with different images don't collide. ``H`` defaults to
Python's built-in ``hash`` (fast, deterministic within a single process). Set
``APC_HASH=sha256`` to opt into a stable cryptographic hash (~100-200 ns/tok
overhead).

Eviction is LRU with reference counting: blocks are kept alive while
``ref_cnt > 0`` and the free queue is a doubly-linked list embedded in
``APCBlock`` for O(1) move-to-tail. All blocks are pre-allocated as a pool
to avoid Python object churn. When ``APC_DISK_PATH`` is configured, full
blocks are also written to a shard-based SSD tier and can be restored after
process restart through a direct-read prompt-cache path.

Numerical note: APC itself is *exact*. The K/V tensors stored in the block
pool are byte-identical to what a fresh prefill would produce — the cache
introduces no approximation, it just retains tensors. However, cold-vs-warm
runs of the same prompt can produce slightly different logits because of
**batch non-invariance** in the attention kernel: a long Q (cold prefill,
e.g. 60 tokens) and a short Q (warm-start suffix, e.g. 13 tokens against
47 cached tokens) trigger different tile shapes / reduction orders inside
flash-attention, and floating-point matmul is non-associative. The
Thinking Machines analysis (2025) and Microsoft Research's LLM-42 paper
give the formal treatment. The same drift happens without prefix caching
any time dynamic batching changes the batch composition between two
identical requests — APC just makes it visible by giving a clean
cold/warm contrast. Warm-to-warm runs *are* deterministic: identical
prompts repeated under APC always produce identical text. For
bit-equivalent cold==warm, you need batch-invariant RMSNorm / matmul /
attention kernels (vLLM's ``--enable-batch-invariance``, SGLang with
FlashInfer/FA3), not a different cache design.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

logger = logging.getLogger("xmlx_vlm.apc")

DEFAULT_BLOCK_SIZE = 16
DEFAULT_NUM_BLOCKS = 2048
SEED_PARENT_HASH = 0

# ── On-disk format versioning ─────────────────────────────────────────────────
# Bump this integer whenever the binary layout of an APC shard changes in a
# backward-incompatible way (e.g. tensor key renames, dtype changes, new
# required fields that older code would silently misread).
#
# Compatibility policy:
#   • Loaders reject shards whose schema_version > APC_DISK_SCHEMA_VERSION
#     (written by a newer build; we don't know the new layout).
#   • Loaders also reject shards whose schema_version < APC_DISK_SCHEMA_VERSION
#     (old layout; rather than guess, we discard and re-prefill — cheap and
#     correct).  Cached shards are automatically replaced on next write.
#   • Shards written without a schema_version field at all (pre-versioning
#     builds) are treated as version "0" and rejected, forcing a clean rebuild.
APC_DISK_SCHEMA_VERSION = "1"


def _hash_use_sha256() -> bool:
    return os.environ.get("APC_HASH", "fast").lower() == "sha256"


def _hash_tokens(parent: int, tokens: Tuple[int, ...], extra: int) -> int:
    """Chain hash for a single block."""
    if _hash_use_sha256():
        h = hashlib.sha256()
        h.update(int(parent & ((1 << 64) - 1)).to_bytes(8, "little"))
        h.update(np.asarray(tokens, dtype=np.int32).tobytes())
        h.update(int(extra & ((1 << 64) - 1)).to_bytes(8, "little"))
        return int.from_bytes(h.digest()[:8], "little", signed=True)
    return hash((parent, tokens, extra))


def _stable_int_hash(*values: int) -> int:
    h = hashlib.sha256()
    for value in values:
        h.update(int(value & ((1 << 64) - 1)).to_bytes(8, "little"))
    return int.from_bytes(h.digest()[:8], "little", signed=True)


def tenant_scoped_hash(tenant: Optional[str], payload_hash: int = 0) -> int:
    """Stable APC salt for tenant-scoped multimodal context."""
    if not tenant:
        return int(payload_hash)
    tenant_bytes = str(tenant).encode("utf-8")
    h = hashlib.sha256()
    h.update(len(tenant_bytes).to_bytes(4, "little"))
    h.update(tenant_bytes)
    h.update(int(payload_hash & ((1 << 64) - 1)).to_bytes(8, "little"))
    return int.from_bytes(h.digest()[:8], "little", signed=True)



def _sequence_hash(token_ids: Sequence[int], extra_hash: int, block_size: int) -> int:
    h = hashlib.sha256()
    h.update(int(extra_hash & ((1 << 64) - 1)).to_bytes(8, "little"))
    h.update(int(block_size).to_bytes(4, "little", signed=False))
    arr = np.asarray([int(t) for t in token_ids], dtype=np.int32)
    h.update(int(arr.size).to_bytes(8, "little", signed=False))
    h.update(arr.tobytes())
    return int.from_bytes(h.digest()[:8], "little", signed=True)


def hash_image_payload(
    pixel_values: Optional[mx.array] = None,
    image_ref: Any = None,
) -> int:
    """Stable content hash of an image payload.

    Prefers hashing the actual ``pixel_values`` tensor (so resize/transform
    differences invalidate the cache). Falls back to hashing the source
    identifier (path / URL / repr).
    """
    if pixel_values is not None:
        try:
            arr = np.asarray(pixel_values).astype(np.float16, copy=False)
            digest = hashlib.sha256(arr.tobytes()).digest()
            return int.from_bytes(digest[:8], "little", signed=True)
        except Exception:
            pass

    if image_ref is None:
        return 0
    if isinstance(image_ref, (list, tuple)):
        h = SEED_PARENT_HASH
        for it in image_ref:
            h = _stable_int_hash(h, hash_image_payload(image_ref=it))
        return h
    if isinstance(image_ref, str):
        digest = hashlib.sha256(image_ref.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little", signed=True)
    if isinstance(image_ref, bytes):
        return int.from_bytes(
            hashlib.sha256(image_ref).digest()[:8], "little", signed=True
        )
    digest = hashlib.sha256(repr(image_ref).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=True)


@dataclass
class APCBlock:
    """One fixed-size KV block. Holds per-layer K/V slabs once committed."""

    block_id: int
    block_hash: Optional[int] = None
    parent_hash: int = SEED_PARENT_HASH
    token_ids: Tuple[int, ...] = ()
    extra_hash: int = 0
    ref_cnt: int = 0
    keys: Optional[List[mx.array]] = None
    values: Optional[List[mx.array]] = None
    last_used: float = 0.0
    prev: Optional["APCBlock"] = None
    next: Optional["APCBlock"] = None


@dataclass
class APCExactCacheEntry:
    """Exact-prefix prompt-cache snapshot for custom cache layouts.

    ``logits`` is the full log-softmax vector (vocab_size,) for the token
    *immediately after* the cached prefix — i.e. the distribution that
    generated the first output token.  Saving it allows a future request
    whose input is an exact extension of this prefix to skip one model
    forward pass by reusing the pre-computed distribution.  ``None`` when
    the snapshot was captured without logits (e.g. from an older index).
    """

    token_ids: Tuple[int, ...]
    extra_hash: int
    prompt_cache: List[Any]
    last_used: float
    logits: Optional[mx.array] = None


@dataclass(frozen=True)
class _DiskBlockSnapshot:
    """Immutable view of an APC block for the asynchronous disk writer."""

    block_hash: int
    parent_hash: int
    extra_hash: int
    token_ids: Tuple[int, ...]
    keys: List[mx.array]
    values: List[mx.array]


@dataclass(frozen=True)
class _DiskLayerMajorBlock:
    """Per-block metadata for a direct layer-major disk write."""

    block_hash: int
    parent_hash: int
    extra_hash: int
    token_ids: Tuple[int, ...]
    source_block_idx: int


@dataclass(frozen=True)
class _DiskLayerMajorSnapshot:
    """Direct shard snapshot from the live per-layer KV cache."""

    blocks: List[_DiskLayerMajorBlock]
    layer_keys: List[mx.array]
    layer_values: List[mx.array]
    block_size: int
    store_id: str
    segment_index: int
    segment_count: int


@dataclass(frozen=True)
class _DiskExactCacheSnapshot:
    cache_hash: int
    token_ids: Tuple[int, ...]
    extra_hash: int
    prompt_cache: List[Any]
    # Full log-softmax vector (vocab_size,) for the first token *after* the
    # cached prefix, captured at save time.  None when not available.
    logits: Optional[mx.array] = None


@dataclass
class APCStats:
    hits: int = 0
    misses: int = 0
    matched_tokens: int = 0
    served_tokens: int = 0
    evictions: int = 0
    stores: int = 0
    pool_used: int = 0
    disk_hits: int = 0
    disk_writes: int = 0
    exact_hits: int = 0
    exact_stores: int = 0

    def snapshot(self, num_blocks: int, block_size: int) -> dict:
        denom = self.matched_tokens + self.served_tokens
        hit_rate = self.matched_tokens / denom if denom > 0 else 0.0
        return {
            "block_size": block_size,
            "num_blocks": num_blocks,
            "pool_used": self.pool_used,
            "lookups_hit": self.hits,
            "lookups_miss": self.misses,
            "matched_tokens": self.matched_tokens,
            "served_tokens": self.served_tokens,
            "token_hit_rate": hit_rate,
            "evictions": self.evictions,
            "stores": self.stores,
            "disk_hits": self.disk_hits,
            "disk_writes": self.disk_writes,
            "exact_hits": self.exact_hits,
            "exact_stores": self.exact_stores,
        }


