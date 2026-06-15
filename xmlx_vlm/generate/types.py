from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Any, List, Optional, Tuple
import mlx.core as mx

logger = logging.getLogger("xmlx_vlm.generate")

DEFAULT_MODEL_PATH = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
DEFAULT_IMAGE = None
DEFAULT_AUDIO = None
DEFAULT_VIDEO = None
DEFAULT_PROMPT = "What are these?"
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
DEFAULT_TOP_K = 0
DEFAULT_MIN_P = 0.0
DEFAULT_REPETITION_CONTEXT_SIZE = 20
DEFAULT_KV_GROUP_SIZE = 64
DEFAULT_KV_QUANT_SCHEME = "uniform"
DEFAULT_COMPLETION_BATCH_SIZE = 32
DEFAULT_PREFILL_BATCH_SIZE = 8
DEFAULT_THINKING_START_TOKEN = "<think>"
DEFAULT_THINKING_END_TOKEN = "</think>"
DEFAULT_QUANTIZED_KV_START = 5000
DEFAULT_PREFILL_STEP_SIZE = 2048


@dataclass
class GenerationResult:
    text: str = ""
    token: Optional[int] = None
    logprobs: Optional[List[float]] = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0


class PromptCacheState:
    """Holds KV cache and token history across conversation turns.

    Pass this to stream_generate via the ``prompt_cache_state`` kwarg to
    reuse the KV cache from previous turns.  Only the new tokens (after
    the common prefix) are processed, avoiding redundant prefill.
    """

    def __init__(self):
        self.cache: Optional[List[Any]] = None
        self.token_ids: Optional[List[int]] = None

    def find_prefix_length(self, new_ids: list) -> int:
        """Return the number of leading tokens that match the cached ids."""
        if self.token_ids is None:
            return 0
        max_len = min(len(self.token_ids), len(new_ids))
        for i in range(max_len):
            if self.token_ids[i] != new_ids[i]:
                return i
        return max_len

    def update(self, token_ids: list, kv_cache: list):
        """Store the full token sequence and corresponding KV cache."""
        self.token_ids = list(token_ids)
        self.cache = kv_cache



@dataclass
class BatchGenerationResult:
    """
    Result of batch generation with optional image size tracking.

    Attributes:
        texts: Generated text for each sample
        tokens: Last generated token for each sample
        logprobs: Log probabilities for each sample
        prompt_tokens: Number of prompt tokens per sample
        generation_tokens: Number of generated tokens per sample
        total_tokens: Total tokens (prompt + generation) per sample
        prompt_tps: Prompt tokens per second per sample
        generation_tps: Generation tokens per second per sample
        peak_memory: Peak memory usage in GB
        image_sizes: Original (height, width) for each image (for tracking)
    """

    texts: List[str]
    tokens: List[Optional[int]]
    logprobs: List[Optional[List[float]]]
    prompt_tokens: List[int]
    generation_tokens: List[int]
    total_tokens: List[int]
    prompt_tps: List[float]
    generation_tps: List[float]
    peak_memory: float = 0.0
    image_sizes: Optional[List[Tuple[int, int]]] = None


@dataclass
class BatchStats:
    """
    An data object to hold generation stats.

    Args:
        prompt_tokens (int): The number of prompt tokens processed.
        prompt_tps (float): The prompt processing tokens-per-second.
        prompt_time (float): The time in seconds spent in prompt processing.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        generation_time (float): The time in seconds spent in generation .
        peak_memory (float): The peak memory used so far in GB.
    """

    prompt_tokens: int = 0
    prompt_tps: float = 0
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0
    generation_time: float = 0
    peak_memory: float = 0


@dataclass
class BatchResponse:
    """
    An data object to hold a batch generation response.

    Args:
        texts: (List[str]): The generated text for each prompt.
        stats (BatchStats): Statistics about the generation.
        image_sizes: (Optional[List[Tuple[int, int]]]): Original (height, width)
            for each image. Useful for tracking which images produced which responses
            and for debugging padding/batching behavior.
    """

    texts: List[str]
    stats: BatchStats
    image_sizes: Optional[List[Tuple[int, int]]] = None


generation_stream = mx.new_thread_local_stream(mx.default_device())


