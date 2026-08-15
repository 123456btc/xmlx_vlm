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

from .padding import _merge_prefill_prompt_kwargs
from .cache_helpers import _make_cache
from .generation_batch import GenerationBatch

class PromptProcessingBatch:
    """
    Handles VLM prompt processing with inputs_embeds and chunked prefill.

    Processes prompt tokens incrementally (one chunk per step) to allow
    interleaving with generation for continuous batching. Transitions to
    a GenerationBatch when prompt processing is complete.
    """

    def __init__(
        self,
        model: nn.Module,
        uids: List[int],
        input_ids: List[List[int]],
        max_tokens: List[int],
        inputs_embeds: mx.array,
        prompt_kwargs: dict,
        logits_processors: Optional[
            List[Optional[List[Callable[[mx.array, mx.array], mx.array]]]]
        ] = None,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        kv_bits=None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
        kv_bits_per_layer=None,
        warm_cache: Optional[List[Any]] = None,
        apc_meta: Optional[List[dict]] = None,
        apc_manager: Optional["_apc.APCManager"] = None,
        right_pad_per_row: Optional[List[int]] = None,
        suffix_lens: Optional[List[int]] = None,
        apc_mode: Optional[str] = None,
    ):
        self.model = model
        self.uids = uids
        self.max_tokens = max_tokens
        self.prefill_step_size = prefill_step_size

        lengths = [len(ids) for ids in input_ids]
        max_length = max(lengths)
        # ``input_ids`` here are the per-row prefill inputs — for warm-start
        # rows this is the suffix, for cold rows the full prompt. When
        # ``right_pad_per_row`` is set the rows are right-padded (used in
        # mixed warm/cold prefill so suffix RoPE positions align). Otherwise
        # we left-pad as before.
        self._right_pad_per_row = right_pad_per_row
        self._suffix_lens = suffix_lens or lengths
        self._left_padding_per_row: List[int]

        if right_pad_per_row is not None:
            # Right-pad each row to max_length (so the last `pad[i]` cells are
            # right-pad and need to be rolled into left-pad by finalize()).
            left_padding = [0] * len(input_ids)
            self._input_ids = _right_pad_prompts(input_ids, max_length=max_length)
        else:
            left_padding = [max_length - l for l in lengths]
            self._input_ids = _left_pad_prompts(input_ids, max_length=max_length)
        self._left_padding_per_row = list(left_padding)
        self._total_prompt_tokens = sum(lengths)
        self._processed_prompt_columns = 0

        self.logits_processors = logits_processors or []
        self._token_context = (
            [list(ids) for ids in input_ids]
            if self.logits_processors and any(self.logits_processors)
            else []
        )
        self._inputs_embeds = inputs_embeds
        self._prompt_kwargs = prompt_kwargs or {}
        self._prompt_length_aware_keys: List[str] = []
        if self._prompt_kwargs and self._inputs_embeds is not None:
            prompt_batch = self._inputs_embeds.shape[0]
            prompt_len = self._inputs_embeds.shape[1]
            for k, v in self._prompt_kwargs.items():
                if (
                    isinstance(v, mx.array)
                    and v.ndim >= 2
                    and v.shape[0] == prompt_batch
                    and v.shape[1] == prompt_len
                ):
                    self._prompt_length_aware_keys.append(k)

        # APC metadata used for post-prefill block harvest (per-row).
        self._apc_meta = apc_meta or []
        self._apc_manager = apc_manager
        self._apc_mode = apc_mode
        self._apc_harvest_enabled = True

        if warm_cache is not None:
            self.prompt_cache = warm_cache
        else:
            self.prompt_cache = _make_cache(
                model,
                left_padding,
                kv_bits=kv_bits,
                kv_group_size=kv_group_size,
                kv_quant_scheme=kv_quant_scheme,
                kv_bits_per_layer=kv_bits_per_layer,
            )

        # Declare per-row right-padding on each cache so finalize() can roll
        # it into left-padding once the prefill forward pass is complete.
        if right_pad_per_row is not None and any(right_pad_per_row):
            for c in self.prompt_cache:
                prepare = getattr(c, "prepare", None)
                if not callable(prepare):
                    self._apc_harvest_enabled = False
                    self._release_apc_meta_blocks()
                    raise RuntimeError(
                        "APC mixed prefill requires a prompt cache with prepare()"
                    )
                prepare(right_padding=right_pad_per_row, lengths=self._suffix_lens)

    def __len__(self):
        return len(self.uids)

    def _release_apc_meta_blocks(self):
        if self._apc_manager is None:
            return
        for meta in self._apc_meta:
            if meta is not None:
                self._apc_manager.release(meta.get("apc_blocks", []))

    def needs_processing(self):
        """True if prompt needs chunked processing before generate()."""
        if self._inputs_embeds is None or self.prefill_step_size is None:
            return self._next_apc_checkpoint_column() is not None
        if self._next_apc_checkpoint_column() is not None:
            return True
        return self._inputs_embeds.shape[1] > self.prefill_step_size

    def _apc_checkpoint_column_for_meta(
        self, batch_idx: int, meta: dict
    ) -> Optional[int]:
        checkpoint_len = int(meta.get("checkpoint_len") or 0)
        if (
            self._apc_mode != "exact"
            or checkpoint_len <= 0
            or meta.get("checkpoint_done")
        ):
            return None
        prefix_len = int(meta.get("prefix_len", 0) or 0)
        if checkpoint_len <= prefix_len:
            meta["checkpoint_done"] = True
            return None
        if self._right_pad_per_row is not None:
            suffix_checkpoint = checkpoint_len - prefix_len
            if suffix_checkpoint >= self._suffix_lens[batch_idx]:
                return None
            return suffix_checkpoint
        return self._left_padding_per_row[batch_idx] + checkpoint_len

    def _next_apc_checkpoint_column(self) -> Optional[int]:
        if (
            self._apc_manager is None
            or self._apc_mode != "exact"
            or not self._apc_meta
            or self._inputs_embeds is None
        ):
            return None
        start = self._processed_prompt_columns
        end = start + self._inputs_embeds.shape[1]
        next_col: Optional[int] = None
        for batch_idx, meta in enumerate(self._apc_meta):
            if meta is None:
                continue
            col = self._apc_checkpoint_column_for_meta(batch_idx, meta)
            if col is None or col <= start or col >= end:
                continue
            next_col = col if next_col is None else min(next_col, col)
        return next_col

    def _row_real_tokens_processed(self, batch_idx: int) -> int:
        meta = self._apc_meta[batch_idx]
        prefix_len = int(meta.get("prefix_len", 0) or 0)
        if self._right_pad_per_row is not None:
            suffix_done = min(
                self._suffix_lens[batch_idx],
                max(0, self._processed_prompt_columns),
            )
            return prefix_len + suffix_done
        real_done = (
            self._processed_prompt_columns - self._left_padding_per_row[batch_idx]
        )
        return prefix_len + min(self._suffix_lens[batch_idx], max(0, real_done))

    def _store_apc_exact_checkpoints(self) -> None:
        if self._apc_manager is None or self._apc_mode != "exact":
            return
        for batch_idx, meta in enumerate(self._apc_meta):
            if meta is None or meta.get("checkpoint_done"):
                continue
            checkpoint_len = int(meta.get("checkpoint_len") or 0)
            if checkpoint_len <= 0:
                continue
            if self._row_real_tokens_processed(batch_idx) != checkpoint_len:
                continue
            prompt_cache = _apc.extract_prompt_cache_from_batch(
                self.prompt_cache,
                batch_idx,
            )
            if prompt_cache is None:
                continue
            self._apc_manager.store_exact_cache(
                meta["full_input_ids"][:checkpoint_len],
                prompt_cache,
                extra_hash=meta.get("extra_hash", 0),
            )
            meta["checkpoint_done"] = True

    def _prompt_kwargs_for_step(self, n: Optional[int] = None) -> dict:
        if n is None or not self._prompt_length_aware_keys:
            return self._prompt_kwargs
        out = dict(self._prompt_kwargs)
        for k in self._prompt_length_aware_keys:
            out[k] = out[k][:, :n, ...]
        return out

    def prompt_step(self) -> int:
        """Process one chunk of the prompt. Returns tokens processed."""
        if not self.needs_processing():
            return 0

        step = self.prefill_step_size or self._inputs_embeds.shape[1]
        n = min(step, self._inputs_embeds.shape[1] - 1)
        checkpoint_col = self._next_apc_checkpoint_column()
        if checkpoint_col is not None:
            n = min(n, checkpoint_col - self._processed_prompt_columns)
        if n <= 0:
            return 0
        prompt_kwargs = self._prompt_kwargs_for_step(n)
        self.model(
            self._input_ids[:, :n],
            cache=self.prompt_cache,
            inputs_embeds=self._inputs_embeds[:, :n],
            n_to_process=n,
            **prompt_kwargs,
        )
        mx.eval([c.state for c in self.prompt_cache])
        self._processed_prompt_columns += n
        self._store_apc_exact_checkpoints()
        self._inputs_embeds = self._inputs_embeds[:, n:]
        self._input_ids = self._input_ids[:, n:]
        for k in self._prompt_length_aware_keys:
            self._prompt_kwargs[k] = self._prompt_kwargs[k][:, n:, ...]
        mx.clear_cache()
        return n

    def generate(
        self, sampler, stop_criteria, compute_logprobs=True, top_logprobs_k=0
    ) -> GenerationBatch:
        """Process final tokens and transition to GenerationBatch."""
        output = self.model(
            self._input_ids,
            cache=self.prompt_cache,
            inputs_embeds=self._inputs_embeds,
            **self._prompt_kwargs,
        )
        logits = output.logits if hasattr(output, "logits") else output
        if self._right_pad_per_row is not None and any(self._right_pad_per_row):
            # Per-row last *real* token sits at index (seq - 1 - right_pad[i]).
            seq = logits.shape[1]
            last_idx = mx.array(
                [seq - 1 - p for p in self._right_pad_per_row], dtype=mx.int32
            )[:, None, None]
            last_idx = mx.broadcast_to(last_idx, (logits.shape[0], 1, logits.shape[-1]))
            logits = mx.take_along_axis(logits, last_idx, axis=1).squeeze(1)
        else:
            logits = logits[:, -1, :]
        if self.logits_processors and any(self.logits_processors):
            processed_logits = []
            for i in range(logits.shape[0]):
                sample_logits = logits[i : i + 1]
                processors = self.logits_processors[i] or []
                for processor in processors:
                    sample_logits = processor(
                        mx.array(self._token_context[i]), sample_logits
                    )
                processed_logits.append(sample_logits)
            logits = mx.concatenate(processed_logits, axis=0)

        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        first_tokens = sampler(logprobs)

        # Roll any right-padding into left-padding so the cache decoded by
        # GenerationBatch sees a canonical layout.
        if self._right_pad_per_row is not None and any(self._right_pad_per_row):
            for c in self.prompt_cache:
                finalize = getattr(c, "finalize", None)
                if not callable(finalize):
                    self._apc_harvest_enabled = False
                    self._release_apc_meta_blocks()
                    raise RuntimeError(
                        "APC mixed prefill requires a prompt cache with finalize()"
                    )
                finalize()
        if logger.isEnabledFor(logging.DEBUG) and os.environ.get("APC_DEBUG"):
            c0 = self.prompt_cache[0] if self.prompt_cache else None
            if c0 is not None:
                off = getattr(c0, "offset", None)
                lp = getattr(c0, "left_padding", None)
                logger.warning(
                    "post-prefill cache[0]: _idx=%s offset=%s left_padding=%s right_pad_per_row=%s suffix_lens=%s",
                    getattr(c0, "_idx", None),
                    off.tolist() if hasattr(off, "tolist") else off,
                    lp.tolist() if hasattr(lp, "tolist") else lp,
                    self._right_pad_per_row,
                    self._suffix_lens,
                )

        gen_batch = GenerationBatch(
            model=self.model,
            uids=list(self.uids),
            inputs=first_tokens,
            prompt_cache=self.prompt_cache,
            sampler=sampler,
            stop_criteria=stop_criteria,
            max_tokens=list(self.max_tokens),
            top_logprobs_k=top_logprobs_k,
            token_context=[list(ctx) for ctx in self._token_context],
            logits_processors=list(self.logits_processors),
        )
        gen_batch.compute_logprobs = compute_logprobs

        if compute_logprobs:
            gen_batch._next_lps = logprobs[
                mx.arange(first_tokens.shape[0]), first_tokens
            ]

        # Prime top-K buffers so the first token can emit top_logprobs too.
        if top_logprobs_k > 0:
            k = top_logprobs_k
            sort_idx = mx.argsort(logprobs, axis=-1)
            top_idx = sort_idx[..., -k:][..., ::-1].astype(mx.int32)
            top_lp = mx.take_along_axis(logprobs, top_idx, axis=-1)
            gen_batch._next_top_idx = top_idx
            gen_batch._next_top_lp = top_lp

        language_model = getattr(self.model, "language_model", self.model)
        rope_deltas = self._capture_rope_deltas(language_model, len(gen_batch.uids))
        if rope_deltas is not None:
            # Normalize to shape (B, 1) so extend/filter stay consistent.
            if rope_deltas.ndim == 0:
                rope_deltas = rope_deltas.reshape(1, 1)
            elif rope_deltas.ndim == 1:
                rope_deltas = rope_deltas[:, None]
            # When a warm-start batch reuses the model's cached _rope_deltas
            # (computed during a previous prefill with a smaller batch), the
            # batch dim won't match this prompt batch's row count. Realign
            # so extend()/filter() down the line stay consistent with the
            # generation batch's row count.
            target_b = first_tokens.shape[0]
            if rope_deltas.shape[0] != target_b:
                if rope_deltas.shape[0] == 1:
                    rope_deltas = mx.broadcast_to(
                        rope_deltas, (target_b, rope_deltas.shape[1])
                    )
                elif rope_deltas.shape[0] < target_b:
                    pad = target_b - rope_deltas.shape[0]
                    rope_deltas = mx.concatenate(
                        [
                            rope_deltas,
                            mx.broadcast_to(
                                rope_deltas[-1:],
                                (pad, rope_deltas.shape[1]),
                            ),
                        ],
                        axis=0,
                    )
                else:
                    rope_deltas = rope_deltas[:target_b]
            gen_batch._rope_deltas = rope_deltas

        # APC: harvest the post-prefill K/V into hashed blocks. Done after the
        # final prefill forward but before the cache references are released
        # so the block tensors snapshot the prompt prefix.
        if (
            self._apc_manager is not None
            and self._apc_meta
            and self._apc_harvest_enabled
        ):
            try:
                for batch_idx, meta in enumerate(self._apc_meta):
                    if meta is None:
                        continue
                    if self._apc_mode == "exact":
                        prompt_cache = _apc.extract_prompt_cache_from_batch(
                            self.prompt_cache,
                            batch_idx,
                        )
                        if prompt_cache is not None:
                            self._apc_manager.store_exact_cache(
                                meta["full_input_ids"],
                                prompt_cache,
                                extra_hash=meta.get("extra_hash", 0),
                            )
                        self._apc_manager.release(meta.get("apc_blocks", []))
                    else:
                        new_blocks = _apc.harvest_blocks_from_batch_cache(
                            self._apc_manager,
                            self.prompt_cache,
                            batch_idx,
                            meta["full_input_ids"],
                            extra_hash=meta.get("extra_hash", 0),
                            skip_first_n_tokens=meta.get("prefix_len", 0),
                        )
                        self._apc_manager.release(
                            meta.get("apc_blocks", []) + new_blocks
                        )
            except Exception as e:
                logger.warning("APC harvest failed during batched prefill: %s", e)
                # Best effort — release any acquired prefix blocks.
                for meta in self._apc_meta:
                    if meta is not None:
                        self._apc_manager.release(meta.get("apc_blocks", []))

        self.uids = []
        self.prompt_cache = []
        self._token_context = []
        self.logits_processors = []
        self._apc_meta = []
        return gen_batch

    @property
    def total_prompt_tokens(self):
        return self._total_prompt_tokens

    @staticmethod
    def _capture_rope_deltas(language_model, B: int):
        if not hasattr(language_model, "_rope_deltas"):
            return None
        rope_deltas = language_model._rope_deltas
        if rope_deltas is None:
            return mx.zeros((B, 1), dtype=mx.int32)
        if rope_deltas.ndim == 0:
            rope_deltas = rope_deltas.reshape(1, 1)
        elif rope_deltas.ndim == 1:
            rope_deltas = rope_deltas[:, None]
        # Falcon OCR emits a singleton meant to broadcast across rows.
        if rope_deltas.shape[0] == 1 and B > 1:
            rope_deltas = mx.broadcast_to(rope_deltas, (B, 1))
        if rope_deltas.shape[0] != B:
            raise RuntimeError(
                f"_rope_deltas shape {rope_deltas.shape} does not match prefill batch size {B}"
            )
        return rope_deltas

