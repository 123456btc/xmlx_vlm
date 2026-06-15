from __future__ import annotations

from .types import (
    DEFAULT_MODEL_PATH,
    DEFAULT_IMAGE,
    DEFAULT_AUDIO,
    DEFAULT_VIDEO,
    DEFAULT_PROMPT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_SEED,
    DEFAULT_TOP_K,
    DEFAULT_MIN_P,
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_COMPLETION_BATCH_SIZE,
    DEFAULT_PREFILL_BATCH_SIZE,
    DEFAULT_THINKING_START_TOKEN,
    DEFAULT_THINKING_END_TOKEN,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_PREFILL_STEP_SIZE,
    GenerationResult,
    PromptCacheState,
    BatchGenerationResult,
    BatchStats,
    BatchResponse,
)
from .single import (
    normalize_resize_shape,
    maybe_quantize_kv_cache,
    wired_limit,
    _prime_cached_prefix_rope_state,
    _speculative_walk,
    _speculative_walk_batch,
    _mtp_rounds,
    _batch_cache_left_padding,
    _mtp_rounds_batch,
    _apply_rep_penalty,
    _dflash_rounds,
    _dflash_rounds_batch,
    generate_step,
    stream_generate,
    generate,
)
from .batch import (
    _left_pad_prompts,
    _right_pad_prompts,
    _prompt_kwarg_row,
    _split_prompt_kwargs_per_row,
    _is_sequence_aligned_prompt_kwarg,
    _pad_sequence_aligned_prompt_kwarg,
    _merge_prefill_prompt_kwargs,
    _extend_cache,
    _make_cache,
    GenerationBatch,
    PromptProcessingBatch,
    BatchGenerator,
    batch_generate,
    _clone_or_share_logits_processor,
    _generate_batch,
)
from .cli import parse_arguments, main
from ..models import cache
from .. import apc as _apc
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from ..prompt_utils import apply_chat_template
from ..utils import load, prepare_inputs
