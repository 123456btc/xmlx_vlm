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

from .prompt_batch import PromptProcessingBatch
from .generation_batch import GenerationBatch
from .padding import (
    _left_pad_prompts, _right_pad_prompts, _split_prompt_kwargs_per_row,
    _prompt_kwarg_row, _is_sequence_aligned_prompt_kwarg, _pad_sequence_aligned_prompt_kwarg,
    _merge_prefill_prompt_kwargs, APC_PRIVATE_PROMPT_KEYS
)

class BatchGenerator:
    """
    Continuous batching with separate prompt processing and generation phases.

    next() returns (prompt_responses, generation_responses) where:
    - prompt_responses is currently always [] (reserved for progress tracking)
    - generation_responses is a list of GenerationBatch.Response objects
    """

    def __init__(
        self,
        model,
        processor,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stop_tokens: Optional[set] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        completion_batch_size: int = DEFAULT_COMPLETION_BATCH_SIZE,
        prefill_batch_size: int = DEFAULT_PREFILL_BATCH_SIZE,
        prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
        prompt_cache=None,
        kv_bits=None,
        kv_bits_per_layer=None,
        kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
        quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
        compute_logprobs: bool = True,
        top_logprobs_k: int = 0,
        logits_processors: Optional[
            List[Callable[[mx.array, mx.array], mx.array]]
        ] = None,
        stream=None,
        apc_manager: Optional["_apc.APCManager"] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.processor = processor
        self.kv_bits = kv_bits
        self.kv_bits_per_layer = kv_bits_per_layer
        self.kv_group_size = kv_group_size
        self.kv_quant_scheme = kv_quant_scheme
        self.quantized_kv_start = quantized_kv_start
        self.compute_logprobs = compute_logprobs
        self.top_logprobs_k = top_logprobs_k
        self.logits_processors = logits_processors or []
        # APC: opt-out for KV-quantized caches. Plain KV models use block APC;
        # mixed/custom cache models use exact prompt-cache snapshots.
        self.apc_mode = None
        if apc_manager is not None and kv_bits is not None:
            apc_manager = None
        if apc_manager is not None:
            self.apc_mode = _apc.model_apc_mode(model)
            if self.apc_mode is None:
                apc_manager = None
        self.apc_manager = apc_manager
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = completion_batch_size

        self._stream = stream or generation_stream

        self.tokenizer.stopping_criteria.add_eos_token_ids(stop_tokens)

        self._generation_batch = GenerationBatch.empty(
            self.model,
            self.sampler,
            self.tokenizer.stopping_criteria,
            compute_logprobs=self.compute_logprobs,
            top_logprobs_k=self.top_logprobs_k,
        )
        self._prompt_batch: Optional[PromptProcessingBatch] = None
        self._unprocessed_sequences = []

        self._prompt_tokens_counter = 0
        self._prompt_time_counter = 0
        self._gen_tokens_counter = 0
        self._steps_counter = 0

        self._wire_stack = contextlib.ExitStack()
        self._wire_stack.enter_context(wired_limit(model, [self._stream]))

    def _get_PromptProcessingBatch(self):
        import sys
        gen_mod = sys.modules.get("xmlx_vlm.generate")
        return getattr(gen_mod, "PromptProcessingBatch", PromptProcessingBatch) if gen_mod else PromptProcessingBatch

    # ---------------- APC integration helpers ----------------
    # Keys that are APC-only metadata; stripped from ``prompt_kwargs`` before
    # the merged kwargs are passed to the language model forward.
    _APC_PRIVATE_KEYS = APC_PRIVATE_PROMPT_KEYS

    def _apc_extra_hash(self, prompt_kwargs: dict) -> int:
        """Salt for the APC hash chain."""
        if self.apc_manager is None:
            return 0
        if prompt_kwargs is None:
            prompt_kwargs = {}
        img = prompt_kwargs.get("_apc_image_hash")
        if img is None:
            pixel_values = prompt_kwargs.get("pixel_values")
            img = _apc.hash_image_payload(pixel_values=pixel_values, image_ref=None)
        tenant = prompt_kwargs.get("_apc_tenant")
        return _apc.tenant_scoped_hash(tenant, img)

    def _apc_media_token_ids(self) -> set[int]:
        return _apc.multimodal_token_ids_from_config(self.model.config)

    def _apc_safe_prefix_lookup_min(self, ids_list: List[int]) -> int:
        safe_min = _apc.media_safe_prefix_min(ids_list, self._apc_media_token_ids())
        return max(0, safe_min - 1)

    def _apc_suffix_is_text_only(self, ids_list: List[int], prefix_len: int) -> bool:
        return _apc.prefix_leaves_text_only_suffix(
            ids_list,
            prefix_len,
            self._apc_media_token_ids(),
        )

    def _apc_prefix_has_media_tokens(
        self, ids_list: List[int], prefix_len: int
    ) -> bool:
        return _apc.prefix_contains_media_tokens(
            ids_list,
            prefix_len,
            self._apc_media_token_ids(),
        )

    def _apc_exact_checkpoint_len(self, ids_list: List[int]) -> int:
        if self.apc_manager is None or getattr(self, "apc_mode", "block") != "exact":
            return 0
        return _apc.adjust_prefix_to_text_suffix_boundary(
            ids_list,
            len(ids_list) - self.apc_manager.exact_cache_guard_tokens,
            self._apc_media_token_ids(),
            max_prefix_tokens=len(ids_list) - 1,
        )

    def _apc_pick_for(self, sequence) -> Optional[dict]:
        """Look up an APC prefix for ``sequence``. Returns dict with matched
        blocks + suffix metadata when there is a usable hit, else None.
        """
        if self.apc_manager is None:
            return None
        uid, ids_list, max_toks, prompt_kwargs, lps = sequence
        if not ids_list or len(ids_list) < 2:
            return None
        safe_lookup_min = self._apc_safe_prefix_lookup_min(ids_list)
        extra_hash = self._apc_extra_hash(prompt_kwargs or {})
        apc_mode = getattr(self, "apc_mode", "block")
        if apc_mode == "exact":
            exact_cache, exact_prefix_len, _exact_logits = self.apc_manager.lookup_exact_cache(
                ids_list,
                extra_hash=extra_hash,
                min_prefix_tokens=safe_lookup_min,
            )
            if (
                exact_cache is not None
                and exact_prefix_len > 0
                and exact_prefix_len < len(ids_list)
            ):
                if not self._apc_suffix_is_text_only(ids_list, exact_prefix_len):
                    return None
                return {
                    "matched_blocks": [],
                    "warm_cache": exact_cache,
                    "prefix_len": exact_prefix_len,
                    "extra_hash": extra_hash,
                    "full_input_ids": list(ids_list),
                }
            return None
        matched, prefix_len = self.apc_manager.lookup_prefix(
            ids_list, extra_hash=extra_hash
        )
        if prefix_len > 0 and self._apc_prefix_has_media_tokens(ids_list, prefix_len):
            self.apc_manager.release(matched)
            matched = []
            prefix_len = 0
        exact_cache = None
        exact_prefix_len = 0
        if prefix_len < len(ids_list):
            exact_cache, exact_prefix_len, _exact_logits = self.apc_manager.lookup_exact_cache(
                ids_list,
                extra_hash=extra_hash,
                min_prefix_tokens=max(prefix_len, safe_lookup_min),
            )
        warm_cache = None
        disk_prefix_len = 0
        if max(prefix_len, exact_prefix_len) < len(ids_list):
            warm_cache, disk_prefix_len = self.apc_manager.lookup_prefix_disk_cache(
                ids_list,
                extra_hash=extra_hash,
                min_prefix_tokens=max(prefix_len, exact_prefix_len, safe_lookup_min),
                allow_memory_overlap=max(prefix_len, exact_prefix_len) > 0,
            )
        if disk_prefix_len > max(
            prefix_len, exact_prefix_len
        ) and disk_prefix_len < len(ids_list):
            if matched:
                self.apc_manager.release(matched)
            if not self._apc_suffix_is_text_only(ids_list, disk_prefix_len):
                return None
            return {
                "matched_blocks": [],
                "warm_cache": warm_cache,
                "prefix_len": disk_prefix_len,
                "extra_hash": extra_hash,
                "full_input_ids": list(ids_list),
            }
        if exact_prefix_len > prefix_len and exact_prefix_len < len(ids_list):
            if matched:
                self.apc_manager.release(matched)
            if not self._apc_suffix_is_text_only(ids_list, exact_prefix_len):
                return None
            return {
                "matched_blocks": [],
                "warm_cache": exact_cache,
                "prefix_len": exact_prefix_len,
                "extra_hash": extra_hash,
                "full_input_ids": list(ids_list),
            }
        if prefix_len > 0 and prefix_len < len(ids_list):
            if not self._apc_suffix_is_text_only(ids_list, prefix_len):
                self.apc_manager.release(matched)
                return None
            return {
                "matched_blocks": matched,
                "prefix_len": prefix_len,
                "extra_hash": extra_hash,
                "full_input_ids": list(ids_list),
            }
        if matched:
            self.apc_manager.release(matched)
        return None

    def _build_mixed_prompt_batch(
        self, sequences: List[tuple]
    ) -> Optional["PromptProcessingBatch"]:
        """Build a multi-row PromptProcessingBatch admitting ``sequences``.

        Each row is independently looked up in APC. Warm rows have their
        suffixes prefilled against pre-populated K/V; cold rows prefill from
        scratch in the same batch. Right-padding aligns RoPE positions
        across rows with different prefix/suffix lengths.

        Returns ``None`` if APC is disabled (in which case the caller should
        use the cold-only path).
        """
        if self.apc_manager is None:
            return None

        picks: List[Optional[dict]] = [self._apc_pick_for(s) for s in sequences]
        any_warm = any(p is not None for p in picks)
        if not any_warm:
            return None  # caller falls back to cold-only path

        uids = [s[0] for s in sequences]
        full_ids = [list(s[1]) for s in sequences]
        max_tokens_list = [s[2] for s in sequences]
        prompt_kwargs_list = [s[3] for s in sequences]
        logits_processors = [s[4] for s in sequences]

        # Per-row prefix length and suffix tokens
        prefix_lens = [p["prefix_len"] if p else 0 for p in picks]
        suffix_ids_list = [full_ids[i][prefix_lens[i] :] for i in range(len(sequences))]
        suffix_lens = [len(s) for s in suffix_ids_list]

        max_suffix_len = max(suffix_lens)
        right_pad_per_row = [max_suffix_len - s for s in suffix_lens]

        # Source inputs_embeds: every row's prompt_kwargs holds the full-prompt
        # embeddings. Slice to suffix per-row, right-pad to max_suffix_len, stack.
        suffix_embeds_per_row: List[mx.array] = []
        for i, kw in enumerate(prompt_kwargs_list):
            if kw is None or kw.get("inputs_embeds") is None:
                raise ValueError("APC mixed prefill requires precomputed inputs_embeds")
            full = kw["inputs_embeds"]  # [1, full_len, D]
            suff = full[:, prefix_lens[i] :, :]
            pad = right_pad_per_row[i]
            if pad > 0:
                pad_emb = mx.zeros(
                    (suff.shape[0], pad, suff.shape[-1]), dtype=suff.dtype
                )
                suff = mx.concatenate([suff, pad_emb], axis=1)
            suffix_embeds_per_row.append(suff)
        inputs_embeds = mx.concatenate(suffix_embeds_per_row, axis=0)

        # Merge prompt-side kwargs (excluding inputs_embeds, which we've just
        # rebuilt). Per-batch tensors get concatenated across rows; scalars
        # take the first row's value (matches the existing cold-only path).
        # APC-private keys (e.g. tenant salt) are dropped — they're consumed
        # in _apc_extra_hash, never forwarded to the model.
        merged_kwargs: dict = {}
        per_row_keys: dict = {}
        batch_size = len(prompt_kwargs_list)
        for i, kw in enumerate(prompt_kwargs_list):
            if not kw:
                continue
            full_len = len(full_ids[i])
            prefix_len = prefix_lens[i]
            right_pad = right_pad_per_row[i]
            for k, v in kw.items():
                if k == "inputs_embeds" or k in self._APC_PRIVATE_KEYS:
                    continue
                if isinstance(v, mx.array) and v.ndim > 0 and v.shape[0] >= 1:
                    row_v = _prompt_kwarg_row(v, i, batch_size)
                    if _is_sequence_aligned_prompt_kwarg(k, row_v, full_len):
                        row_v = row_v[:, prefix_len:, ...]
                        row_v = _pad_sequence_aligned_prompt_kwarg(
                            row_v,
                            max_suffix_len,
                            left=False,
                        )
                    per_row_keys.setdefault(k, []).append(row_v)
                elif k not in merged_kwargs:
                    merged_kwargs[k] = v
        for k, vs in per_row_keys.items():
            merged_kwargs[k] = mx.concatenate(vs, axis=0)

        apc_mode = getattr(self, "apc_mode", "block")
        if apc_mode == "exact":
            row_caches = [
                p["warm_cache"] if p is not None else self.model.make_cache()
                for p in picks
            ]
            warm_cache, _ = _apc.make_warm_batch_exact_cache_multi(
                row_caches,
                prefix_lens,
            )
            if warm_cache is None:
                return None
        else:
            # Build the multi-row warm cache (zeros for cold rows, K/V for warm).
            num_layers = (
                len(self.model.make_cache())
                if hasattr(self.model, "make_cache")
                else len(self.model.layers)
            )
            warm_cache, _ = _apc.make_warm_batch_kv_cache_multi(
                picks, num_layers=num_layers
            )

        apc_meta = [
            {
                "full_input_ids": full_ids[i],
                "prefix_len": prefix_lens[i],
                "extra_hash": (
                    picks[i]["extra_hash"]
                    if picks[i]
                    else self._apc_extra_hash(prompt_kwargs_list[i] or {})
                ),
                "apc_blocks": picks[i].get("matched_blocks", []) if picks[i] else [],
                "checkpoint_len": self._apc_exact_checkpoint_len(full_ids[i]),
            }
            for i in range(len(sequences))
        ]

        PromptProcessingBatchClass = self._get_PromptProcessingBatch()
        return PromptProcessingBatchClass(
            model=self.model,
            uids=uids,
            input_ids=suffix_ids_list,
            max_tokens=max_tokens_list,
            inputs_embeds=inputs_embeds,
            prompt_kwargs=merged_kwargs,
            logits_processors=logits_processors,
            prefill_step_size=self.prefill_step_size,
            kv_bits=self.kv_bits,
            kv_group_size=self.kv_group_size,
            kv_quant_scheme=self.kv_quant_scheme,
            kv_bits_per_layer=getattr(self, "kv_bits_per_layer", None),
            warm_cache=warm_cache,
            apc_meta=apc_meta,
            apc_manager=self.apc_manager,
            right_pad_per_row=right_pad_per_row,
            suffix_lens=suffix_lens,
            apc_mode=apc_mode,
        )

    def _build_apc_meta_for_cold(
        self,
        input_ids_list: List[List[int]],
        prompt_kwargs_list: List[Optional[dict]],
    ) -> Optional[List[Optional[dict]]]:
        """Build per-row harvest metadata for a cold-prefill batch so the
        produced K/V are added to APC after prefill.
        """
        if self.apc_manager is None:
            return None
        meta: List[Optional[dict]] = []
        for ids_list, kw in zip(input_ids_list, prompt_kwargs_list):
            extra_hash = self._apc_extra_hash(kw or {})
            meta.append(
                {
                    "full_input_ids": list(ids_list),
                    "prefix_len": 0,
                    "extra_hash": extra_hash,
                    "apc_blocks": [],
                    "checkpoint_len": self._apc_exact_checkpoint_len(list(ids_list)),
                }
            )
        return meta

    @property
    def stream(self):
        return self._stream

    def close(self):
        if self._wire_stack is not None:
            self._wire_stack.close()
            self._wire_stack = None

    def __del__(self):
        self.close()

    def insert(
        self,
        prompts,
        max_tokens: Union[List[int], int, None] = None,
        prompt_kwargs: Optional[List[dict]] = None,
        logits_processors: Optional[
            List[Optional[List[Callable[[mx.array, mx.array], mx.array]]]]
        ] = None,
    ):
        uids = []

        if max_tokens is None or isinstance(max_tokens, int):
            max_tokens = [max_tokens or self.max_tokens] * len(prompts)

        if prompt_kwargs is None:
            prompt_kwargs = [{}] * len(prompts)
        if logits_processors is None:
            logits_processors = [self.logits_processors] * len(prompts)
        elif len(logits_processors) != len(prompts):
            raise ValueError("Insufficient number of logits_processors provided")

        for p, m, kw, lp in zip(prompts, max_tokens, prompt_kwargs, logits_processors):
            self._unprocessed_sequences.append((self.uid_count, p, m, kw, lp))
            uids.append(self.uid_count)
            self.uid_count += 1
        # Sort in ascending order of length
        self._unprocessed_sequences = sorted(
            self._unprocessed_sequences, key=lambda x: len(x[1])
        )
        return uids

    def remove(self, uid) -> bool:
        """Remove a sequence from the batch by uid."""
        with mx.stream(self._stream):
            # Waiting in the queue.
            for i, (seq_uid, _, _, _, _) in enumerate(self._unprocessed_sequences):
                if seq_uid == uid:
                    self._unprocessed_sequences.pop(i)
                    return True

            # Being prefilled
            if self._prompt_batch is not None and uid in self._prompt_batch.uids:
                if len(self._prompt_batch.uids) == 1:
                    self._prompt_batch.uids = []
                    self._prompt_batch.prompt_cache = []
                    self._prompt_batch = None
                    mx.clear_cache()
                    return True

            # Already decoding.
            if uid in self._generation_batch.uids:
                idx = self._generation_batch.uids.index(uid)
                keep = [i for i in range(len(self._generation_batch.uids)) if i != idx]
                self._generation_batch.filter(keep)
                return True

            return False

    @property
    def unprocessed_prompts(self):
        """Backward-compatible alias for server flush logic."""
        return self._unprocessed_sequences

    @property
    def has_pending_prompts(self):
        """True if there are prompts waiting or being processed."""
        return len(self._unprocessed_sequences) > 0 or self._prompt_batch is not None

    @property
    def has_work(self):
        """True if there is any remaining work."""
        return (
            len(self._generation_batch) > 0
            or self._prompt_batch is not None
            or len(self._unprocessed_sequences) > 0
        )

    def stats(self):
        """Return accumulated batch statistics."""
        stats = BatchStats()
        stats.prompt_tokens = self._prompt_tokens_counter
        stats.prompt_time = self._prompt_time_counter
        stats.prompt_tps = (
            self._prompt_tokens_counter / self._prompt_time_counter
            if self._prompt_time_counter > 0
            else 0
        )
        stats.generation_tokens = self._gen_tokens_counter
        stats.peak_memory = mx.get_peak_memory() / 1e9
        return stats

    def _next(self, **kwargs):
        generation_responses = []
        prompt_responses = []

        # Decode-first: always emit a generation step before touching prefill.
        if len(self._generation_batch) > 0:
            generation_responses = self._generation_batch.next()
            self._gen_tokens_counter += len(generation_responses)
            self._steps_counter += 1
            if self._steps_counter % 512 == 0:
                mx.clear_cache()

        if len(self._generation_batch) >= self.completion_batch_size:
            return prompt_responses, generation_responses

        if self._prompt_batch is not None:
            if self._prompt_batch.needs_processing():
                tic = time.perf_counter()
                n = self._prompt_batch.prompt_step()
                self._prompt_time_counter += time.perf_counter() - tic
                self._prompt_tokens_counter += n
                return prompt_responses, generation_responses

            tic = time.perf_counter()
            gen_batch = self._prompt_batch.generate(
                self.sampler,
                self.tokenizer.stopping_criteria,
                compute_logprobs=self.compute_logprobs,
                top_logprobs_k=self.top_logprobs_k,
            )
            self._prompt_time_counter += time.perf_counter() - tic
            self._generation_batch.extend(gen_batch)
            self._prompt_batch = None
            mx.clear_cache()
            return prompt_responses, generation_responses

        num_active = len(self._generation_batch)
        num_to_add = self.completion_batch_size - num_active
        if self._unprocessed_sequences and num_to_add >= self.prefill_batch_size:
            # Take up to prefill_batch_size pending sequences. If APC is on
            # and at least one of them has a prefix hit, build a mixed
            # warm/cold PromptProcessingBatch with right-padded suffixes so
            # warm and cold rows prefill in a single forward pass.
            n = min(self.prefill_batch_size, len(self._unprocessed_sequences))
            sequences = self._unprocessed_sequences[:n]
            if logger.isEnabledFor(logging.DEBUG) and os.environ.get("APC_DEBUG"):
                logger.warning(
                    "APC admit n=%d (pending=%d)",
                    n,
                    len(self._unprocessed_sequences),
                )
            mixed = self._build_mixed_prompt_batch(sequences)
            if mixed is not None:
                self._unprocessed_sequences = self._unprocessed_sequences[n:]
                self._prompt_batch = mixed
                self._prompt_tokens_counter += self._prompt_batch.total_prompt_tokens
                if self._prompt_batch.needs_processing():
                    tic = time.perf_counter()
                    nstep = self._prompt_batch.prompt_step()
                    self._prompt_time_counter += time.perf_counter() - tic
                else:
                    tic = time.perf_counter()
                    gen_batch = self._prompt_batch.generate(
                        self.sampler,
                        self.tokenizer.stopping_criteria,
                        compute_logprobs=self.compute_logprobs,
                        top_logprobs_k=self.top_logprobs_k,
                    )
                    self._prompt_time_counter += time.perf_counter() - tic
                    self._generation_batch.extend(gen_batch)
                    self._prompt_batch = None
                    mx.clear_cache()
                return prompt_responses, generation_responses

            self._unprocessed_sequences = self._unprocessed_sequences[n:]

            uids = [s[0] for s in sequences]
            input_ids = [s[1] for s in sequences]
            max_tokens_list = [s[2] for s in sequences]
            prompt_kwargs_list = [s[3] for s in sequences]
            logits_processors = [s[4] for s in sequences]

            inputs_embeds, merged_kwargs = _merge_prefill_prompt_kwargs(
                prompt_kwargs_list, input_ids
            )

            # APC: also harvest cold-prefill prefixes so future requests hit.
            apc_meta = self._build_apc_meta_for_cold(input_ids, prompt_kwargs_list)

            PromptProcessingBatchClass = self._get_PromptProcessingBatch()
            self._prompt_batch = PromptProcessingBatchClass(
                model=self.model,
                uids=uids,
                input_ids=input_ids,
                max_tokens=max_tokens_list,
                inputs_embeds=inputs_embeds,
                prompt_kwargs=merged_kwargs,
                logits_processors=logits_processors,
                prefill_step_size=self.prefill_step_size,
                kv_bits=self.kv_bits,
                kv_group_size=self.kv_group_size,
                kv_quant_scheme=self.kv_quant_scheme,
                kv_bits_per_layer=getattr(self, "kv_bits_per_layer", None),
                apc_meta=apc_meta,
                apc_manager=self.apc_manager,
                apc_mode=self.apc_mode,
            )
            self._prompt_tokens_counter += self._prompt_batch.total_prompt_tokens

            if self._prompt_batch.needs_processing():
                tic = time.perf_counter()
                n = self._prompt_batch.prompt_step()
                self._prompt_time_counter += time.perf_counter() - tic
            else:
                tic = time.perf_counter()
                gen_batch = self._prompt_batch.generate(
                    self.sampler,
                    self.tokenizer.stopping_criteria,
                    compute_logprobs=self.compute_logprobs,
                    top_logprobs_k=self.top_logprobs_k,
                )
                self._prompt_time_counter += time.perf_counter() - tic
                self._generation_batch.extend(gen_batch)
                self._prompt_batch = None
                mx.clear_cache()

            return prompt_responses, generation_responses

        return prompt_responses, generation_responses

    def next(self, **kwargs):
        with mx.stream(self._stream):
            return self._next(**kwargs)

def batch_generate(
    model,
    processor,
    images: Union[str, List[str]] = None,
    audios: Union[str, List[str]] = None,
    prompts: List[str] = None,
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    group_by_shape: bool = True,
    track_image_sizes: bool = True,
    **kwargs,
):
    """
    Generate responses for the given batch of prompts with variable-sized images.

    This function implements the transformers-style approach to batching:
    1. Group images with the same shape for efficient batch processing
    2. Process each group as a batch (no padding waste within groups)
    3. Track original image sizes for proper attention masking
    4. Restore results to original batch order

    Key insight: Instead of padding all images to the same spatial dimensions
    (which wastes computation and may hurt accuracy), we group same-sized
    images together so there's zero padding within each group.

    Args:
       model (nn.Module): The language model.
       processor (PreTrainedTokenizer): The tokenizer/processor.
       images (Union[str, List[str]]): Images (paths, URLs, or PIL images).
       audios (Union[str, List[str]]): Audio files (not yet supported for batching).
       prompts (List[str]): The input prompts.
       max_tokens (Union[int, List[int]]): Maximum number of output tokens. This
          can be per prompt if a list is provided.
       verbose (bool): If ``True``, print tokens and timing information.
       group_by_shape (bool): If ``True``, group same-shaped images for efficient
          batch processing.
       track_image_sizes (bool): If ``True``, track and return original image sizes.
       kwargs: The remaining options get passed to :obj:`BatchGenerator`.
          See :obj:`BatchGenerator` for more details.

    Returns:
        BatchResponse with generated texts, statistics, and optionally image_sizes.
    """
    from PIL import Image

    from ...utils import process_image
    import sys
    gen_mod = sys.modules.get("xmlx_vlm.generate")
    dyn_generate_batch = getattr(gen_mod, "_generate_batch", _generate_batch) if gen_mod else _generate_batch

    processor.detokenizer.reset()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Handle single image case
    if isinstance(images, str):
        images = [images]

    # Handle no images case
    if images is None:
        texts, stats = dyn_generate_batch(
            model, processor, prompts, None, max_tokens, verbose, **kwargs
        )
        return BatchResponse(texts, stats)

    # Load and preprocess images
    image_processor = (
        processor.image_processor if hasattr(processor, "image_processor") else None
    )

    processed_images = []
    image_sizes_original = []
    for img in images:
        if isinstance(img, str):
            pil_img = process_image(img, None, image_processor)
        elif isinstance(img, Image.Image):
            pil_img = img
        else:
            pil_img = img
        processed_images.append(pil_img)
        # Track original size
        if hasattr(pil_img, "height"):
            image_sizes_original.append((pil_img.height, pil_img.width))
        else:
            image_sizes_original.append((0, 0))

    # Group images by shape for efficient processing (no padding within groups)
    if group_by_shape and len(processed_images) > 1:
        grouped_images, grouped_indices = group_images_by_shape(processed_images)

        if verbose:
            print(f"[batch_generate] Found {len(grouped_images)} unique image shapes")
    else:
        # Single image or grouping disabled - treat as one group
        shape = (
            (processed_images[0].height, processed_images[0].width)
            if processed_images
            else (0, 0)
        )
        grouped_images = {shape: processed_images}
        grouped_indices = {shape: list(range(len(processed_images)))}

    # Process each shape group
    all_texts = [None] * len(prompts)
    all_image_sizes = [None] * len(prompts)
    total_stats = BatchStats()

    for shape, indices in grouped_indices.items():
        # Get images and prompts for this shape group
        group_images = [processed_images[i] for i in indices]
        group_prompts = [prompts[i] for i in indices]
        group_sizes = [image_sizes_original[i] for i in indices]

        # Handle per-sample max_tokens
        if isinstance(max_tokens, list):
            group_max_tokens = [max_tokens[i] for i in indices]
        else:
            group_max_tokens = max_tokens

        group_kwargs = dict(kwargs)
        logits_processors = group_kwargs.get("logits_processors")
        if logits_processors is not None and isinstance(logits_processors, list):
            if not logits_processors or all(callable(p) for p in logits_processors):
                group_kwargs["logits_processors"] = logits_processors
            else:
                group_kwargs["logits_processors"] = [
                    logits_processors[i] for i in indices
                ]

        # Process the entire group at once (same shape = no padding needed)
        chunk_texts, chunk_stats = dyn_generate_batch(
            model,
            processor,
            group_prompts,
            group_images,
            group_max_tokens,
            **group_kwargs,
        )

        # Store results in original order
        for j, orig_idx in enumerate(indices):
            all_texts[orig_idx] = chunk_texts[j]
            all_image_sizes[orig_idx] = group_sizes[j]

        # Accumulate stats
        total_stats.prompt_tokens += chunk_stats.prompt_tokens
        total_stats.prompt_time += chunk_stats.prompt_time
        total_stats.generation_tokens += chunk_stats.generation_tokens
        total_stats.generation_time += chunk_stats.generation_time

    mx.clear_cache()

    # Compute final stats
    if total_stats.prompt_time > 0:
        total_stats.prompt_tps = total_stats.prompt_tokens / total_stats.prompt_time
    if total_stats.generation_time > 0:
        total_stats.generation_tps = (
            total_stats.generation_tokens / total_stats.generation_time
        )
    total_stats.peak_memory = mx.get_peak_memory() / 1e9

    if verbose:
        print(f"[batch_generate] Finished processing {len(prompts)} samples")
        print(
            f"[batch_generate] Prompt: {total_stats.prompt_tokens} tokens, {total_stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {total_stats.generation_tokens} tokens, "
            f"{total_stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {total_stats.peak_memory:.3f} GB")

    response = BatchResponse(all_texts, total_stats)
    if track_image_sizes:
        response.image_sizes = all_image_sizes
    return response

def _clone_or_share_logits_processor(processor):
    if hasattr(processor, "clone"):
        return processor.clone()
    warnings.warn(
        "Sharing logits processor across batch entries because it does not "
        "implement clone(). Stateful logits processors should implement clone() "
        "to avoid shared state across sequences.",
        RuntimeWarning,
        stacklevel=2,
    )
    return processor

def _generate_batch(
    model,
    processor,
    prompts: List[str],
    images: List = None,
    max_tokens: Union[int, List[int]] = 100,
    verbose: bool = False,
    **kwargs,
) -> Tuple[List[str], BatchStats]:

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    batch_size = len(prompts)
    logits_processors = kwargs.pop("logits_processors", None)

    num_images_list = [
        1 if i < (len(images) if images is not None else 0) else 0
        for i in range(len(prompts))
    ]
    import sys
    gen_mod = sys.modules.get("xmlx_vlm.generate")
    dyn_apply_chat_template = getattr(gen_mod, "apply_chat_template", apply_chat_template) if gen_mod else apply_chat_template

    formatted_prompts = [
        dyn_apply_chat_template(
            processor,
            model.config,
            p,
            num_images=num_images_list[i],
        )
        for i, p in enumerate(prompts)
    ]

    add_special_tokens = (
        getattr(processor, "chat_template", None) is None
        if model.config.model_type in ["gemma3", "gemma3n", "gemma4"]
        else True
    )

    resize_shape = normalize_resize_shape(kwargs.pop("resize_shape", None))
    image_token_index = getattr(model.config, "image_token_index", None)

    dyn_prepare_inputs = getattr(gen_mod, "prepare_inputs", prepare_inputs) if gen_mod else prepare_inputs
    inputs = dyn_prepare_inputs(
        processor,
        images=images,
        audio=None,
        prompts=formatted_prompts,
        image_token_index=image_token_index,
        resize_shape=resize_shape,
        add_special_tokens=add_special_tokens,
        pad_to_uniform_size=False,  # Since images are pre-grouped by shape, they're already uniform size
    )
    input_ids = inputs.get("input_ids", None)
    pixel_values = inputs.get("pixel_values", None)
    mask = inputs.get("attention_mask", None)

    data_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }

    if getattr(model, "no_chunked_prefill", False):
        kwargs.pop("prefill_step_size", None)
        kwargs["prefill_step_size"] = None

    # Use batch_size for prefill and completion to ensure consistent processing
    gen = BatchGenerator(
        model.language_model,
        processor,
        prefill_batch_size=batch_size,
        completion_batch_size=batch_size,
        compute_logprobs=False,
        **kwargs,
    )

    embedding_output = model.get_input_embeddings(
        input_ids, pixel_values, mask=mask, **data_kwargs
    )

    gen_kwargs = {**data_kwargs, **embedding_output.to_dict()}

    if logits_processors and all(
        callable(processor) for processor in logits_processors
    ):
        logits_processors = [
            [_clone_or_share_logits_processor(p) for p in logits_processors]
            for _ in range(batch_size)
        ]

    uids = gen.insert(
        input_ids.tolist(),
        max_tokens,
        prompt_kwargs=_split_prompt_kwargs_per_row(gen_kwargs, batch_size),
        logits_processors=logits_processors,
    )
    results = {uid: [] for uid in uids}

    tic = time.perf_counter()
    while gen.has_work:
        _, generation_responses = gen.next()
        for r in generation_responses:
            if r.finish_reason != "stop":
                results[r.uid].append(r.token)
    total_time = time.perf_counter() - tic

    gen.close()

    detokenizer = processor.detokenizer
    texts = []
    for uid in uids:
        detokenizer.reset()
        for t in results[uid]:
            detokenizer.add_token(t)
        detokenizer.finalize()
        texts.append(detokenizer.text)

    stats = gen.stats()
    stats.generation_time = total_time - stats.prompt_time
    if stats.generation_time > 0:
        stats.generation_tps = stats.generation_tokens / stats.generation_time
    return texts, stats

