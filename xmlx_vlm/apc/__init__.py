from __future__ import annotations

from .types import (
    DEFAULT_BLOCK_SIZE,
    DEFAULT_NUM_BLOCKS,
    SEED_PARENT_HASH,
    APC_DISK_SCHEMA_VERSION,
    APCBlock,
    APCExactCacheEntry,
    APCStats,
    hash_image_payload,
    tenant_scoped_hash,
    _hash_tokens,
)
from .disk_store import DiskBlockStore, _free_ram_bytes
from .manager import APCManager
from .utils import (
    make_warm_kv_cache,
    make_warm_kv_cache_from_layers,
    make_warm_batch_kv_cache,
    make_warm_batch_kv_cache_multi,
    make_warm_batch_exact_cache_multi,
    extract_prompt_cache_from_batch,
    harvest_blocks_from_batch_cache,
    model_apc_mode,
    model_supports_apc,
    from_env,
    adjust_prefix_to_text_suffix_boundary,
    media_safe_prefix_min,
    media_token_spans,
    multimodal_token_ids_from_config,
    prefix_contains_media_tokens,
    prefix_leaves_text_only_suffix,
    _copy_mlx_array,
)
