from __future__ import annotations
import contextlib
import logging
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ... import apc as _apc
from ...models import cache
from ...prompt_utils import apply_chat_template
from ...tokenizer_utils import make_streaming_detokenizer
from ...turboquant import BatchTurboQuantKVCache, turboquant_enabled
from ...utils import StoppingCriteria, ThinkingBudgetCriteria, group_images_by_shape, prepare_inputs

from ..types import (
    DEFAULT_COMPLETION_BATCH_SIZE,
    DEFAULT_PREFILL_BATCH_SIZE,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_THINKING_START_TOKEN,
    DEFAULT_THINKING_END_TOKEN,
    DEFAULT_MAX_TOKENS,
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_QUANTIZED_KV_START,
    BatchGenerationResult,
    BatchStats,
    BatchResponse,
    generation_stream,
)
from ..single import normalize_resize_shape, wired_limit

logger = logging.getLogger('xmlx_vlm.generate')

from .cache_helpers import _extend_cache

class GenerationBatch:
    """
    Batched token generator with double-buffered pipelining.

    Manages the generation phase after prompt processing, with KV caches,
    sampling, and stop detection for multiple sequences. Uses async_eval
    to overlap GPU computation with CPU processing (decode-ahead pattern).
    """

    @dataclass
    class Response:
        uid: int
        token: int
        token_logprob: float
        finish_reason: Optional[str]
        top_logprobs: Optional[List[Tuple[int, float]]] = None

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        inputs: mx.array,
        prompt_cache: List[Any],
        sampler: Callable[[mx.array], mx.array],
        stop_criteria,
        max_tokens: List[int],
        top_logprobs_k: int = 0,
        token_context: Optional[List[List[int]]] = None,
        logits_processors: Optional[
            List[Optional[List[Callable[[mx.array, mx.array], mx.array]]]]
        ] = None,
    ):
        self.model = model
        self._language_model = getattr(model, "language_model", model)
        self.uids = uids
        self.prompt_cache = prompt_cache
        self.sampler = sampler
        self.stop_criteria = stop_criteria
        self.max_tokens = max_tokens
        self._num_tokens = [0] * len(uids)
        self.compute_logprobs = True
        self.top_logprobs_k = top_logprobs_k
        self.logits_processors = logits_processors or []
        self.token_context = [list(ctx) for ctx in (token_context or [])]

        self._current_tokens = None
        self._current_lps = None
        self._next_tokens = inputs
        self._next_lps = None
        self._next_top_idx = None
        self._next_top_lp = None

        # Per-sequence MRoPE delta
        self._rope_deltas = None

    def __len__(self):
        return len(self.uids)

    def _step(self):
        """Perform one generation step with double buffering."""
        self._current_tokens = self._next_tokens
        self._current_lps = self._next_lps
        inputs = self._current_tokens

        fwd_kwargs = {}
        if self._rope_deltas is not None:
            fwd_kwargs["rope_deltas"] = self._rope_deltas

        output = self._language_model(
            inputs[:, None], cache=self.prompt_cache, **fwd_kwargs
        )
        logits = output.logits if hasattr(output, "logits") else output
        logits = logits[:, -1, :]

        if self.logits_processors and any(self.logits_processors):
            last_tokens = inputs.tolist()
            if not self.token_context:
                self.token_context = [[] for _ in self.uids]
            for i, token in enumerate(last_tokens):
                self.token_context[i].append(token)

            processed_logits = []
            for i in range(logits.shape[0]):
                sample_logits = logits[i : i + 1]
                processors = self.logits_processors[i] or []
                for processor in processors:
                    if hasattr(processor, "process_last_token"):
                        sample_logits = processor.process_last_token(
                            last_tokens[i], sample_logits
                        )
                    else:
                        sample_logits = processor(
                            mx.array(self.token_context[i]), sample_logits
                        )
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = self.sampler(logprobs)

        self._next_tokens = sampled
        prev_top_idx = self._next_top_idx
        prev_top_lp = self._next_top_lp

        eval_targets = [self._next_tokens]
        if self.compute_logprobs:
            self._next_lps = logprobs[mx.arange(sampled.shape[0]), sampled]
            eval_targets.append(self._next_lps)
        else:
            self._next_lps = None

        k = self.top_logprobs_k
        if k > 0:
            # argsort ascending; take last K columns and reverse for descending.
            sort_idx = mx.argsort(logprobs, axis=-1)
            top_idx = sort_idx[..., -k:][..., ::-1].astype(mx.int32)
            top_lp = mx.take_along_axis(logprobs, top_idx, axis=-1)
            self._next_top_idx = top_idx
            self._next_top_lp = top_lp
            eval_targets.extend([top_idx, top_lp])
        else:
            self._next_top_idx = None
            self._next_top_lp = None

        mx.async_eval(*eval_targets)

        if self._current_lps is not None:
            to_eval = [inputs, self._current_lps]
            if prev_top_idx is not None:
                to_eval.extend([prev_top_idx, prev_top_lp])
            mx.eval(*to_eval)
            top_idx_list = prev_top_idx.tolist() if prev_top_idx is not None else None
            top_lp_list = prev_top_lp.tolist() if prev_top_lp is not None else None
            return (
                inputs.tolist(),
                self._current_lps.tolist(),
                top_idx_list,
                top_lp_list,
            )
        else:
            mx.eval(inputs)
            return inputs.tolist(), None, None, None

    def extend(self, other: "GenerationBatch"):
        """Extend this batch with another generation batch."""
        self_was_empty = len(self.uids) == 0
        self.uids.extend(other.uids)
        self.prompt_cache = _extend_cache(self.prompt_cache, other.prompt_cache)
        self.max_tokens.extend(other.max_tokens)
        self._num_tokens.extend(other._num_tokens)
        self.token_context.extend(other.token_context)
        self.logits_processors.extend(other.logits_processors)

        if self._current_tokens is None:
            self._current_tokens = other._current_tokens
            self._current_lps = other._current_lps
        elif other._current_tokens is not None:
            self._current_tokens = mx.concatenate(
                [self._current_tokens, other._current_tokens]
            )
            if self._current_lps is not None and other._current_lps is not None:
                self._current_lps = mx.concatenate(
                    [self._current_lps, other._current_lps]
                )

        if self._next_tokens is None:
            self._next_tokens = other._next_tokens
            self._next_lps = other._next_lps
            self._next_top_idx = other._next_top_idx
            self._next_top_lp = other._next_top_lp
        elif other._next_tokens is not None:
            self._next_tokens = mx.concatenate([self._next_tokens, other._next_tokens])
            if self._next_lps is not None and other._next_lps is not None:
                self._next_lps = mx.concatenate([self._next_lps, other._next_lps])

            if (
                self._next_top_idx is not None
                and other._next_top_idx is not None
                and self._next_top_idx.shape[-1] == other._next_top_idx.shape[-1]
            ):
                self._next_top_idx = mx.concatenate(
                    [self._next_top_idx, other._next_top_idx]
                )
                self._next_top_lp = mx.concatenate(
                    [self._next_top_lp, other._next_top_lp]
                )
            else:
                self._next_top_idx = None
                self._next_top_lp = None

        if self_was_empty:
            self._rope_deltas = other._rope_deltas
        elif (self._rope_deltas is None) != (other._rope_deltas is None):
            raise RuntimeError(
                "extend() mixes MRoPE and non-MRoPE batches; both sides must "
                "carry rope_deltas or neither side may."
            )
        elif self._rope_deltas is not None:
            self._rope_deltas = mx.concatenate([self._rope_deltas, other._rope_deltas])

    def filter(self, keep: List[int]):
        """Filter the batch to keep only the specified indices."""
        self.uids = [self.uids[idx] for idx in keep]
        self.max_tokens = [self.max_tokens[idx] for idx in keep]
        self._num_tokens = [self._num_tokens[idx] for idx in keep]
        if self.token_context:
            self.token_context = [self.token_context[idx] for idx in keep]
        if self.logits_processors:
            self.logits_processors = [self.logits_processors[idx] for idx in keep]

        if not keep:
            self.prompt_cache.clear()
            self._current_tokens = None
            self._current_lps = None
            self._next_tokens = None
            self._next_lps = None
            self._next_top_idx = None
            self._next_top_lp = None
            self._rope_deltas = None
            self.token_context = []
            self.logits_processors = []
        else:
            keep_arr = mx.array(keep, mx.int32)
            for c in self.prompt_cache:
                c.filter(keep_arr)
            if self._next_tokens is not None:
                self._next_tokens = self._next_tokens[keep_arr]
            if self._next_lps is not None:
                self._next_lps = self._next_lps[keep_arr]
            if self._next_top_idx is not None:
                self._next_top_idx = self._next_top_idx[keep_arr]
                self._next_top_lp = self._next_top_lp[keep_arr]
            if self._rope_deltas is not None:
                self._rope_deltas = self._rope_deltas[keep_arr]

    def next(self) -> List[Response]:
        """Generate the next batch of tokens."""
        if not self.uids:
            return []

        tokens, lp_list, top_idx_list, top_lp_list = self._step()

        keep = []
        responses = []
        for i in range(len(self.uids)):
            finish_reason = None
            self._num_tokens[i] += 1
            tok = tokens[i]

            if self.stop_criteria(tok):
                finish_reason = "stop"
            elif self._num_tokens[i] >= self.max_tokens[i]:
                finish_reason = "length"

            if finish_reason is None:
                keep.append(i)

            top_lp = None
            if top_idx_list is not None:
                top_lp = list(zip(top_idx_list[i], top_lp_list[i]))

            responses.append(
                self.Response(
                    uid=self.uids[i],
                    token=tok,
                    token_logprob=lp_list[i] if lp_list is not None else 0.0,
                    finish_reason=finish_reason,
                    top_logprobs=top_lp,
                )
            )

        if len(keep) < len(self.uids):
            self.filter(keep)

        return responses

    @classmethod
    def empty(
        cls, model, sampler, stop_criteria, compute_logprobs=True, top_logprobs_k=0
    ):
        """Create an empty generation batch."""
        batch = cls.__new__(cls)
        batch.model = model
        batch._language_model = getattr(model, "language_model", model)
        batch.uids = []
        batch.prompt_cache = []
        batch.sampler = sampler
        batch.stop_criteria = stop_criteria
        batch.max_tokens = []
        batch._num_tokens = []
        batch.compute_logprobs = compute_logprobs
        batch.top_logprobs_k = top_logprobs_k
        batch.token_context = []
        batch.logits_processors = []
        batch._current_tokens = None
        batch._current_lps = None
        batch._next_tokens = None
        batch._next_lps = None
        batch._next_top_idx = None
        batch._next_top_lp = None
        batch._rope_deltas = None
        return batch

