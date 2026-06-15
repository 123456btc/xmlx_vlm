from __future__ import annotations

from .utils import (
    normalize_resize_shape,
    maybe_quantize_kv_cache,
    wired_limit,
    _prime_cached_prefix_rope_state,
    _apply_rep_penalty,
    generation_stream,
)
from .speculative import _speculative_walk, _speculative_walk_batch
from .mtp import _mtp_rounds, _mtp_rounds_batch, _batch_cache_left_padding
from .dflash import _dflash_rounds, _dflash_rounds_batch
from .pipeline import generate_step, stream_generate, generate
