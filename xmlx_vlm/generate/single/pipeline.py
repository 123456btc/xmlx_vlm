from __future__ import annotations
import contextlib
import functools
import logging
import os
import time
import warnings
from collections.abc import Sequence
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from mlx_lm.generate import maybe_quantize_kv_cache as mlx_maybe_quantize_kv_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from ... import apc as _apc
from ... import diffusion_generate
from ...models import cache
from ...prompt_utils import apply_chat_template
from ...tokenizer_utils import make_streaming_detokenizer
from ...turboquant import TurboQuantKVCache, turboquant_enabled
from ...utils import StoppingCriteria, ThinkingBudgetCriteria, prepare_inputs

from ..types import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_SEED,
    DEFAULT_TOP_K,
    DEFAULT_MIN_P,
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_THINKING_START_TOKEN,
    DEFAULT_THINKING_END_TOKEN,
    DEFAULT_PREFILL_STEP_SIZE,
    GenerationResult,
    PromptCacheState,
    generation_stream,
)

logger = logging.getLogger('xmlx_vlm.generate')

from .utils import (
    normalize_resize_shape, maybe_quantize_kv_cache, wired_limit,
    _prime_cached_prefix_rope_state, _apply_rep_penalty, generation_stream
)
from .speculative import _speculative_walk, _speculative_walk_batch
from .mtp import _mtp_rounds, _mtp_rounds_batch, _batch_cache_left_padding
from .dflash import _dflash_rounds, _dflash_rounds_batch

def generate_step(
    input_ids: mx.array,
    model: nn.Module,
    pixel_values,
    mask,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = DEFAULT_REPETITION_CONTEXT_SIZE,
    top_p: float = DEFAULT_TOP_P,
    min_p: float = DEFAULT_MIN_P,
    top_k: int = DEFAULT_TOP_K,
    logit_bias: Optional[Dict[int, float]] = None,
    prompt_cache: Optional[List[Any]] = None,
    max_kv_size: Optional[int] = None,
    kv_bits: Optional[float] = None,
    kv_group_size: int = DEFAULT_KV_GROUP_SIZE,
    kv_quant_scheme: str = DEFAULT_KV_QUANT_SCHEME,
    quantized_kv_start: int = DEFAULT_QUANTIZED_KV_START,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
    draft_model: Optional[nn.Module] = None,
    draft_kind: str = "dflash",
    draft_block_size: Optional[int] = None,
    prompt_cache_checkpoint: Optional[Callable[[int, List[Any]], None]] = None,
    prompt_cache_checkpoint_len: Optional[int] = None,
    **kwargs,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        input_ids (mx.array): The input prompt token ids.
        model (nn.Module): The model to use for generation.
        pixel_values: The pixel values for vision models (optional).
        mask: The attention mask (optional).
        max_tokens (int): Maximum number of tokens to generate.
        temperature (float): The temperature for sampling, if 0 the argmax is used.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty.
        top_p (float, optional): Nucleus sampling, higher means model considers
          more less likely words.
        min_p (float, optional): Minimum probability threshold relative to the
          highest-probability token.
        top_k (int, optional): Restrict sampling to the top-k tokens.
        logit_bias (dictionary, optional): Additive logit bias.
        prompt_cache (list, optional): Pre-existing KV cache for the prompt.
        max_kv_size (int, optional): Maximum KV cache size.
        kv_bits (float, optional): Number of bits for KV cache quantization.
        kv_group_size (int): Group size for uniform KV cache quantization.
        kv_quant_scheme (str): KV cache quantization backend.
        quantized_kv_start (int): Start index for quantized KV cache.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits.
        prefill_step_size (int): Number of tokens to process per prefill step.
          Chunked prefill processes prompts in smaller chunks to reduce peak
          memory usage.
        draft_model (nn.Module, optional): A drafter for speculative decoding.
          When set, the decode loop is replaced by the drafter's speculative
          loop (e.g. DFlash block-diffusion). VLM prefill with image/audio
          is supported via the same ``get_input_embeddings`` path the normal
          decoder uses; decode itself is text-only. ``temperature`` and
          ``sampler`` are respected; ``logprobs`` is always ``None`` on the
          speculative path.
        draft_block_size (int, optional): Override the drafter's configured
          block size.

    Yields:
        Generator[Tuple[mx.array, mx.array], None, None]: A generator producing
          one token and a vector of log probabilities.
    """

    import sys
    gen_mod = sys.modules.get("xmlx_vlm.generate")
    dyn_maybe_quantize_kv_cache = getattr(gen_mod, "maybe_quantize_kv_cache", maybe_quantize_kv_cache) if gen_mod else maybe_quantize_kv_cache

    quantize_cache_fn = functools.partial(
        dyn_maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
        kv_quant_scheme=kv_quant_scheme,
    )

    dyn_make_sampler = getattr(gen_mod, "make_sampler", make_sampler) if gen_mod else make_sampler
    dyn_make_logits_processors = getattr(gen_mod, "make_logits_processors", make_logits_processors) if gen_mod else make_logits_processors

    if sampler is None:
        sampler = dyn_make_sampler(
            temp=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
        )

    processors = dyn_make_logits_processors(
        logit_bias, repetition_penalty, repetition_context_size
    )
    if logits_processors is not None:
        processors.extend(logits_processors)

    y = input_ids
    tokens = mx.array([], dtype=input_ids.dtype)

    thinking_budget_criteria = kwargs.pop("thinking_budget_criteria", None)
    enable_specprefill = kwargs.pop("enable_specprefill", False)
    specprefill_draft_model = kwargs.pop("specprefill_draft_model", None)
    specprefill_keep_pct = kwargs.pop("specprefill_keep_pct", 0.3)
    specprefill_chunk_size = kwargs.pop("specprefill_chunk_size", 32)
    specprefill_n_lookahead = kwargs.pop("specprefill_n_lookahead", 8)
    specprefill_threshold = kwargs.pop("specprefill_threshold", 512)

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model.language_model,
            max_kv_size=max_kv_size,
        )

    # Speculative decoding setup
    last_outputs = None
    if draft_model is not None:
        if draft_kind == "mtp":
            # MTP drafter consumes target's last-layer hidden + shared K/V
            # (per layer-type) rather than per-layer hidden captures.
            kwargs["return_hidden"] = True
            kwargs["return_shared_kv"] = True
        else:
            kwargs["capture_layer_ids"] = list(draft_model.config.target_layer_ids)
        prefill_step_size = None
        # Reset stale mRoPE state from any previous generation.
        lm = model.language_model if hasattr(model, "language_model") else model
        if hasattr(lm, "_position_ids"):
            lm._position_ids = None
        if hasattr(lm, "_rope_deltas"):
            lm._rope_deltas = None

    def _step(y, inputs_embeds=None):
        nonlocal tokens, kwargs, last_outputs

        with mx.stream(generation_stream):
            if "decoder_input_ids" in kwargs:
                outputs = model.language_model(
                    cache=prompt_cache,
                    **kwargs,
                )
            else:
                outputs = model.language_model(
                    y,
                    inputs_embeds=inputs_embeds,
                    cache=prompt_cache,
                    **kwargs,
                )

            last_outputs = outputs
            logits = outputs.logits[:, -1, :]

            if len(processors) > 0 and len(y) > 0:
                tokens = mx.concat([tokens, y.flatten()])

                for processor in processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits)
            y = sampler(logprobs)

            if outputs.cross_attention_states is not None:
                kwargs = {"cross_attention_states": outputs.cross_attention_states}
            elif outputs.encoder_outputs is not None:
                kwargs = {"encoder_outputs": outputs.encoder_outputs}
            else:
                kwargs = {}

            return y, logprobs.squeeze(0) if logprobs.shape[0] == 1 else logprobs

    try:
        with mx.stream(generation_stream):
            # Get input embeddings (handles both multimodal and text-only)
            embedding_output = model.get_input_embeddings(
                input_ids, pixel_values, mask=mask, **kwargs
            )

            inputs_embeds = embedding_output.inputs_embeds

            kwargs.update(
                {
                    k: v
                    for k, v in embedding_output.to_dict().items()
                    if k != "inputs_embeds" and v is not None
                }
            )

            prefill_draft_model = specprefill_draft_model or draft_model
            run_specprefill = (
                enable_specprefill
                and input_ids.shape[1] > specprefill_threshold
                and pixel_values is None
                and prefill_draft_model is not None
            )

            if run_specprefill:
                # Load draft model if passed as string
                if isinstance(prefill_draft_model, str):
                    if not hasattr(generate_step, "_draft_model_cache"):
                        generate_step._draft_model_cache = {}
                    if prefill_draft_model not in generate_step._draft_model_cache:
                        from ...utils import load_model, get_model_path
                        draft_path = get_model_path(prefill_draft_model)
                        generate_step._draft_model_cache[prefill_draft_model] = load_model(draft_path)
                    prefill_draft_model = generate_step._draft_model_cache[prefill_draft_model]

                from ...specprefill import score_tokens, select_chunks, sparse_prefill
                tokens_list = input_ids.flatten().tolist()
                draft_lm = getattr(prefill_draft_model, "language_model", prefill_draft_model)
                importance = score_tokens(
                    draft_lm,
                    tokens_list,
                    n_lookahead=specprefill_n_lookahead,
                )

                selected_indices = select_chunks(
                    importance,
                    keep_pct=specprefill_keep_pct,
                    chunk_size=specprefill_chunk_size,
                )

                # Ensure the last token is included
                M = len(tokens_list)
                sel_list = selected_indices.tolist()
                if (M - 1) not in sel_list:
                    sel_list.append(M - 1)
                    sel_list.sort()
                    selected_indices = mx.array(sel_list)

                target_lm = getattr(model, "language_model", model)
                sparse_logits = sparse_prefill(
                    target_lm,
                    mx.array(tokens_list),
                    selected_indices,
                    prompt_cache,
                )

                logits = sparse_logits[:, -1, :]

                if len(processors) > 0:
                    tokens = mx.concat([tokens, input_ids.flatten()])
                    for processor in processors:
                        logits = processor(tokens, logits)

                quantize_cache_fn(prompt_cache)
                logprobs = logits - mx.logsumexp(logits)
                y = sampler(logprobs)

                # Clear kwargs to prevent passing inputs to subsequent steps
                kwargs = {}
            else:
                if getattr(model, "no_chunked_prefill", False):
                    prefill_step_size = None
                checkpoint_len = (
                    int(prompt_cache_checkpoint_len)
                    if prompt_cache_checkpoint is not None
                    and prompt_cache_checkpoint_len is not None
                    else None
                )
                checkpoint_done = False
                should_chunk = (
                    prefill_step_size is not None and inputs_embeds.shape[1] > prefill_step_size
                ) or (
                    checkpoint_len is not None and 0 < checkpoint_len < inputs_embeds.shape[1]
                )
                if prefill_step_size is not None and should_chunk:
                    # Chunked prefill with embeddings
                    total_tokens = inputs_embeds.shape[1]
                    processed_tokens = 0
                    with tqdm(total=total_tokens, desc="Prefill", unit="tok") as pbar:
                        while inputs_embeds.shape[1] > 1:
                            n_to_process = min(prefill_step_size, inputs_embeds.shape[1] - 1)
                            if (
                                checkpoint_len is not None
                                and not checkpoint_done
                                and processed_tokens < checkpoint_len
                                and processed_tokens + n_to_process > checkpoint_len
                            ):
                                n_to_process = checkpoint_len - processed_tokens
                            model.language_model(
                                inputs=input_ids[:, :n_to_process],
                                inputs_embeds=inputs_embeds[:, :n_to_process],
                                cache=prompt_cache,
                                n_to_process=n_to_process,
                                **kwargs,
                            )
                            quantize_cache_fn(prompt_cache)
                            mx.eval([c.state for c in prompt_cache])
                            processed_tokens += n_to_process
                            if (
                                checkpoint_len is not None
                                and not checkpoint_done
                                and processed_tokens == checkpoint_len
                            ):
                                prompt_cache_checkpoint(processed_tokens, prompt_cache)
                                checkpoint_done = True
                            inputs_embeds = inputs_embeds[:, n_to_process:]
                            input_ids = input_ids[:, n_to_process:]
                            mx.clear_cache()
                            pbar.update(n_to_process)

                    input_ids = input_ids[:, -1:]

                y, logprobs = _step(input_ids, inputs_embeds=inputs_embeds)

        mx.async_eval(y)

        # Speculative decoding
        if draft_model is not None:
            B = input_ids.shape[0]
            if draft_kind == "mtp":
                shared_kv_states = last_outputs.shared_kv_states
                hidden = last_outputs.hidden_states[-1]
                if B == 1:
                    mx.eval(y)
                    yield y.item(), logprobs
                    yield from _mtp_rounds(
                        model,
                        draft_model,
                        prompt_cache,
                        hidden,
                        shared_kv_states,
                        first_bonus=y.item(),
                        max_tokens=max_tokens,
                        sampler=sampler,
                        draft_block_size=draft_block_size,
                        token_dtype=input_ids.dtype,
                    )
                else:
                    mx.eval(y)
                    # ``y`` is shape (B,) from sampler — no squeeze needed.
                    first_bonus = y if y.ndim == 1 else y.reshape(-1)
                    yield first_bonus.tolist(), logprobs
                    # Surface EOS token IDs from the model config so per-row
                    # stops are detected inside the round loop.
                    eos = getattr(model.config, "eos_token_id", None)
                    if isinstance(eos, int):
                        eos_set = {eos}
                    elif eos is None:
                        eos_set = None
                    else:
                        eos_set = set(int(x) for x in eos)
                    yield from _mtp_rounds_batch(
                        model,
                        draft_model,
                        prompt_cache,
                        hidden,
                        shared_kv_states,
                        first_bonus=first_bonus,
                        max_tokens=max_tokens,
                        sampler=sampler,
                        draft_block_size=draft_block_size,
                        token_dtype=input_ids.dtype,
                        eos_token_ids=eos_set,
                    )
                return

            if draft_kind != "dflash":
                raise ValueError(
                    f"Unknown draft_kind {draft_kind!r}. Supported: " "['dflash', 'mtp']"
                )
            hidden = mx.concatenate(last_outputs.hidden_states, axis=-1)
            if B == 1:
                mx.eval(y)
                yield y.item(), logprobs
                yield from _dflash_rounds(
                    model,
                    draft_model,
                    prompt_cache,
                    hidden,
                    first_bonus=y.item(),
                    max_tokens=max_tokens,
                    sampler=sampler,
                    draft_block_size=draft_block_size,
                    token_dtype=input_ids.dtype,
                    processors=tuple(processors),
                    tokens_ctx=tokens if tokens.size > 0 else None,
                )
            else:
                mx.eval(y)
                first_bonus = y.squeeze(-1)
                yield first_bonus.tolist(), logprobs
                yield from _dflash_rounds_batch(
                    model,
                    draft_model,
                    prompt_cache,
                    hidden,
                    first_bonus=first_bonus,
                    max_tokens=max_tokens,
                    sampler=sampler,
                    draft_block_size=draft_block_size,
                    token_dtype=input_ids.dtype,
                )
            return

        n = 0
        while True:
            if n != max_tokens:
                next_y, next_logprobs = _step(y[None])
                mx.async_eval(next_y)
            if n == 0:
                mx.eval(y)
            if n == max_tokens:
                break

            yield y.item(), logprobs
            if n % 256 == 0:
                mx.clear_cache()

            if thinking_budget_criteria is not None:
                next_y = thinking_budget_criteria.apply_forced_token(next_y)
            y, logprobs = next_y, next_logprobs
            n += 1
    finally:
        from ...specprefill import cleanup_rope
        cleanup_rope(model.language_model if hasattr(model, "language_model") else model)

def stream_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    audio: Union[str, List[str]] = None,
    video: Union[str, List[str]] = None,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        model (nn.Module): The model to use for generation.
        processor (PreTrainedTokenizer): The tokenizer/processor.
        prompt (str): The input prompt text.
        image (Union[str, List[str]], optional): Image path(s) or URL(s).
        audio (Union[str, List[str]], optional): Audio file path(s).
        prefill_step_size (int, optional): Number of tokens to process per prefill
          step. When set, enables chunked prefill which processes long prompts in
          smaller chunks to reduce peak memory usage.
        kwargs: Additional options passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        Generator[GenerationResult]: A generator producing GenerationResult objects
          containing the generated text, tokens, and statistics.
    """
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Set up thinking budget criteria if requested
    thinking_budget = kwargs.pop("thinking_budget", None)
    thinking_end_token = kwargs.pop("thinking_end_token", DEFAULT_THINKING_END_TOKEN)
    thinking_start_token = kwargs.pop(
        "thinking_start_token", DEFAULT_THINKING_START_TOKEN
    )
    enable_thinking = kwargs.pop("enable_thinking", False)

    # Skip special tokens
    skip_special_tokens = kwargs.pop("skip_special_tokens", False)
    skip_special_token_ids = (
        set(tokenizer.all_special_ids)
        if skip_special_tokens and hasattr(tokenizer, "all_special_ids")
        else []
    )

    add_special_tokens = (
        getattr(processor, "chat_template", None) is None
        if model.config.model_type in ["gemma3", "gemma3n", "gemma4"]
        else True
    )

    resize_shape = normalize_resize_shape(kwargs.pop("resize_shape", None))
    image_token_index = getattr(model.config, "image_token_index", None)
    vision_cache = kwargs.pop("vision_cache", None)
    prompt_cache_state = kwargs.pop("prompt_cache_state", None)
    apc_manager: Optional[_apc.APCManager] = kwargs.pop("apc_manager", None)
    apc_tenant: Optional[str] = kwargs.pop("apc_tenant", None)

    if kwargs.get("input_ids", None) is not None:
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values", None)
        mask = kwargs.pop("mask", None)
    else:
        import sys
        gen_mod = sys.modules.get("xmlx_vlm.generate")
        dyn_prepare_inputs = getattr(gen_mod, "prepare_inputs", prepare_inputs) if gen_mod else prepare_inputs
        inputs = dyn_prepare_inputs(
            processor,
            images=image,
            audio=audio,
            videos=video,
            prompts=prompt,
            image_token_index=image_token_index,
            resize_shape=resize_shape,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )
        input_ids = inputs.get("input_ids", None)
        pixel_values = inputs.get("pixel_values", None)
        mask = inputs.get("attention_mask", None)
        data_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        kwargs.update(data_kwargs)

    # Diffusion model dispatch: block-diffusion models (e.g. DiffusionGemma)
    # run their own denoising loop instead of the autoregressive step generator.
    if diffusion_generate.is_diffusion_model(model):
        yield from diffusion_generate.stream_diffusion_generate_from_kwargs(
            model,
            processor,
            tokenizer,
            input_ids,
            pixel_values,
            mask,
            skip_special_token_ids,
            kwargs,
            prompt_cache_state=prompt_cache_state,
        )
        return

    # Vision feature caching: reuse cached image features across turns
    if vision_cache is not None and image is not None and pixel_values is not None:
        cached = vision_cache.get(image)
        if cached is not None:
            kwargs["cached_image_features"] = cached
        elif hasattr(model, "encode_image"):
            features = model.encode_image(pixel_values)
            mx.eval(features)
            vision_cache.put(image, features)
            kwargs["cached_image_features"] = features

    # Prompt cache reuse: skip common prefix from previous turn
    reused_prefix_len = 0
    full_input_ids_list = input_ids.flatten().tolist()
    apc_blocks_in_use: List[_apc.APCBlock] = []
    apc_extra_hash = 0
    apc_mode: Optional[str] = None

    if apc_manager is not None:
        apc_mode = _apc.model_apc_mode(model.language_model)
        if apc_mode is None:
            apc_manager = None

    if apc_manager is not None:
        image_hash = _apc.hash_image_payload(pixel_values=pixel_values, image_ref=image)
        apc_extra_hash = _apc.tenant_scoped_hash(apc_tenant, image_hash)

    # Media-aware APC helpers: a reusable prefix is only safe when the suffix
    # is text-only, because restored KV still carries full image/video features.
    multimodal_token_ids = _apc.multimodal_token_ids_from_config(model.config)
    apc_safe_prefix_min = _apc.media_safe_prefix_min(
        full_input_ids_list, multimodal_token_ids
    )
    apc_safe_prefix_lookup_min = max(0, apc_safe_prefix_min - 1)

    def _apc_suffix_is_text_only(prefix_len: int) -> bool:
        return _apc.prefix_leaves_text_only_suffix(
            full_input_ids_list, prefix_len, multimodal_token_ids
        )

    def _apc_prefix_has_media_tokens(prefix_len: int) -> bool:
        return _apc.prefix_contains_media_tokens(
            full_input_ids_list, prefix_len, multimodal_token_ids
        )

    if prompt_cache_state is not None and prompt_cache_state.cache is not None:
        prefix_len = prompt_cache_state.find_prefix_length(full_input_ids_list)
        if prefix_len > 0 and prefix_len < input_ids.shape[1]:
            if _apc_suffix_is_text_only(
                prefix_len
            ) and _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs):
                reused_prefix_len = prefix_len
                # Trim to only new tokens
                input_ids = input_ids[:, prefix_len:]
                # Suffix is text-only, so vision features are not needed.
                pixel_values = None
                kwargs.pop("cached_image_features", None)
                # Reuse the saved KV cache (trimmed to prefix length)
                kv_cache = prompt_cache_state.cache
                # Trim cache to prefix_len in case it includes generated tokens
                for c in kv_cache:
                    if hasattr(c, "keys") and c.keys is not None:
                        cached_len = c.keys.shape[2]
                        if cached_len > prefix_len:
                            c.keys = c.keys[:, :, :prefix_len, :]
                            c.values = c.values[:, :, :prefix_len, :]
                            if hasattr(c, "offset"):
                                c.offset = prefix_len
                kwargs["prompt_cache"] = kv_cache

    # APC: cross-request, hash-based prefix lookup. Only consulted if a per-turn
    # PromptCacheState didn't already produce a hit.
    if apc_manager is not None and reused_prefix_len == 0:
        if apc_mode == "exact":
            exact_prompt_cache, exact_prefix_len, _exact_logits = apc_manager.lookup_exact_cache(
                full_input_ids_list,
                extra_hash=apc_extra_hash,
                min_prefix_tokens=apc_safe_prefix_lookup_min,
            )
            if (
                exact_prompt_cache is not None
                and exact_prefix_len > 0
                and exact_prefix_len < input_ids.shape[1]
                and _apc_suffix_is_text_only(exact_prefix_len)
                and _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs)
            ):
                reused_prefix_len = exact_prefix_len
                input_ids = input_ids[:, exact_prefix_len:]
                pixel_values = None
                kwargs.pop("cached_image_features", None)
                kwargs["prompt_cache"] = exact_prompt_cache
        else:
            matched_blocks, prefix_len = apc_manager.lookup_prefix(
                full_input_ids_list, extra_hash=apc_extra_hash
            )
            if prefix_len > 0 and _apc_prefix_has_media_tokens(prefix_len):
                apc_manager.release(matched_blocks)
                matched_blocks = []
                prefix_len = 0
            exact_prompt_cache = None
            exact_prefix_len = 0
            if prefix_len < input_ids.shape[1]:
                exact_prompt_cache, exact_prefix_len, _exact_logits = apc_manager.lookup_exact_cache(
                    full_input_ids_list,
                    extra_hash=apc_extra_hash,
                    min_prefix_tokens=max(prefix_len, apc_safe_prefix_lookup_min),
                )
            disk_prompt_cache = None
            disk_prefix_len = 0
            if max(prefix_len, exact_prefix_len) < input_ids.shape[1]:
                disk_prompt_cache, disk_prefix_len = (
                    apc_manager.lookup_prefix_disk_cache(
                        full_input_ids_list,
                        extra_hash=apc_extra_hash,
                        min_prefix_tokens=max(
                            prefix_len, exact_prefix_len, apc_safe_prefix_lookup_min
                        ),
                        allow_memory_overlap=max(prefix_len, exact_prefix_len) > 0,
                    )
                )
            if (
                disk_prefix_len > max(prefix_len, exact_prefix_len)
                and disk_prefix_len < input_ids.shape[1]
            ):
                if matched_blocks:
                    apc_manager.release(matched_blocks)
                if _apc_suffix_is_text_only(
                    disk_prefix_len
                ) and _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs):
                    reused_prefix_len = disk_prefix_len
                    input_ids = input_ids[:, disk_prefix_len:]
                    pixel_values = None
                    kwargs.pop("cached_image_features", None)
                    kwargs["prompt_cache"] = disk_prompt_cache
            elif (
                exact_prefix_len > prefix_len and exact_prefix_len < input_ids.shape[1]
            ):
                if matched_blocks:
                    apc_manager.release(matched_blocks)
                if _apc_suffix_is_text_only(
                    exact_prefix_len
                ) and _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs):
                    reused_prefix_len = exact_prefix_len
                    input_ids = input_ids[:, exact_prefix_len:]
                    pixel_values = None
                    kwargs.pop("cached_image_features", None)
                    kwargs["prompt_cache"] = exact_prompt_cache
            elif prefix_len > 0 and prefix_len < input_ids.shape[1]:
                if _apc_suffix_is_text_only(
                    prefix_len
                ) and _prime_cached_prefix_rope_state(model, input_ids, mask, kwargs):
                    apc_blocks_in_use = matched_blocks
                    reused_prefix_len = prefix_len
                    input_ids = input_ids[:, prefix_len:]
                    pixel_values = None
                    kwargs.pop("cached_image_features", None)
                    kwargs["prompt_cache"] = _apc.make_warm_kv_cache(
                        matched_blocks,
                        min_capacity_tokens=prefix_len + input_ids.shape[1] + 1,
                    )
                else:
                    apc_manager.release(matched_blocks)
            elif matched_blocks:
                # Full match (no new tokens to compute) — release; fall through to normal path
                apc_manager.release(matched_blocks)

    if thinking_budget is not None:
        thinking_start_token_id = tokenizer.encode(
            thinking_start_token, add_special_tokens=False
        )[-1]
        enable_thinking = enable_thinking and (
            thinking_start_token_id in input_ids.flatten().tolist()
        )
        tokenizer.thinking_budget_criteria = ThinkingBudgetCriteria(
            tokenizer=tokenizer,
            thinking_budget=thinking_budget,
            thinking_end_token=thinking_end_token,
            thinking_start_token=thinking_start_token,
            enable_thinking=enable_thinking,
        )
        kwargs["thinking_budget_criteria"] = tokenizer.thinking_budget_criteria
    else:
        tokenizer.thinking_budget_criteria = None

    # Ensure we have a prompt_cache we can track for reuse.
    if "prompt_cache" not in kwargs:
        kwargs["prompt_cache"] = cache.make_prompt_cache(
            model.language_model,
            max_kv_size=kwargs.get("max_kv_size", None),
        )
    tracked_cache = kwargs["prompt_cache"]

    total_prompt_tokens = reused_prefix_len + input_ids.size

    with wired_limit(model, [generation_stream]):
        detokenizer = make_streaming_detokenizer(processor)
        thinking_criteria = getattr(tokenizer, "thinking_budget_criteria", None)
        exact_checkpoint_len = None
        exact_checkpoint = None
        if apc_manager is not None and apc_mode == "exact" and reused_prefix_len == 0:
            exact_checkpoint_len = _apc.adjust_prefix_to_text_suffix_boundary(
                full_input_ids_list,
                len(full_input_ids_list) - apc_manager.exact_cache_guard_tokens,
                multimodal_token_ids,
                max_prefix_tokens=len(full_input_ids_list) - 1,
            )
            if exact_checkpoint_len <= 0:
                exact_checkpoint_len = None

            def exact_checkpoint(prefix_len: int, prompt_cache: List[Any]) -> None:
                apc_manager.store_exact_cache(
                    full_input_ids_list[:prefix_len],
                    prompt_cache,
                    extra_hash=apc_extra_hash,
                )

        gen = generate_step(
            input_ids,
            model,
            pixel_values,
            mask,
            prompt_cache_checkpoint=exact_checkpoint,
            prompt_cache_checkpoint_len=exact_checkpoint_len,
            **kwargs,
        )
        tic = time.perf_counter()

        generated_tokens = []
        for n, (token, logprobs) in enumerate(gen):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = total_prompt_tokens / prompt_time
                tic = time.perf_counter()
                if (
                    apc_manager is not None
                    and apc_mode == "exact"
                    and reused_prefix_len == 0
                ):
                    try:
                        # Save KV for the full input prefix together with the
                        # first-token log-softmax (ds4-style logits snapshot).
                        # ``logprobs`` here is the full vocab-size log-softmax
                        # vector produced by generate_step right after prefill.
                        apc_manager.store_exact_cache(
                            full_input_ids_list,
                            tracked_cache,
                            extra_hash=apc_extra_hash,
                            logits=logprobs,
                        )
                    except Exception as e:
                        logger.warning("APC exact-cache store failed: %s", e)

            generated_tokens.append(token)

            # Check thinking budget and force token if needed
            if thinking_criteria is not None:
                thinking_criteria(token)

            # Stop generation if the token is in the eos_token_ids
            if tokenizer.stopping_criteria(token):
                break

            detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)

            # Yield the last segment if streaming
            yield GenerationResult(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                prompt_tokens=total_prompt_tokens,
                generation_tokens=n + 1,
                total_tokens=total_prompt_tokens + n + 1,
                prompt_tps=prompt_tps,
                generation_tps=(n + 1) / (time.perf_counter() - tic),
                peak_memory=mx.get_peak_memory() / 1e9,
            )

        detokenizer.finalize()
        yield GenerationResult(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            prompt_tokens=total_prompt_tokens,
            generation_tokens=n + 1,
            total_tokens=total_prompt_tokens + n + 1,
            prompt_tps=prompt_tps,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
        )

        # Save cache state for potential reuse on next turn
        all_ids: Optional[List[int]] = None
        if prompt_cache_state is not None:
            all_ids = full_input_ids_list + [
                t.item() if hasattr(t, "item") else t for t in generated_tokens
            ]
            prompt_cache_state.update(all_ids, tracked_cache)

        # ── ds4-style post-generation session save ──────────────────────────
        # Save the full KV state (input + all generated tokens) so the NEXT
        # request in the same conversation can restore from here instead of
        # re-prefilling the entire context.  This is the critical gap vs ds4:
        # the mid-generation APC store (n==0 above) only covers the input
        # prefix; this one covers input + assistant response.
        #
        # Key: full_input_ids_list + generated_tokens
        # Value: tracked_cache (KV populated for every one of those tokens)
        # Logits: the last token's log-softmax (distribution for the *next*
        #         token after this session ends, e.g. start of next user turn).
        if apc_manager is not None and apc_mode == "exact" and generated_tokens:
            try:
                if all_ids is None:
                    all_ids = full_input_ids_list + [
                        t.item() if hasattr(t, "item") else t
                        for t in generated_tokens
                    ]
                apc_manager.store_exact_cache(
                    all_ids,
                    tracked_cache,
                    extra_hash=apc_extra_hash,
                    logits=logprobs,  # last-token log-softmax from the loop
                )
            except Exception as e:
                logger.warning("APC post-gen session save failed: %s", e)

        # APC: harvest new blocks from the post-generation KV state.
        if apc_manager is not None and apc_mode == "block":
            try:
                if all_ids is None:
                    all_ids = full_input_ids_list + [
                        t.item() if hasattr(t, "item") else t for t in generated_tokens
                    ]
                # Snapshot keys/values up to the live offset for each layer.
                layer_keys: List[mx.array] = []
                layer_values: List[mx.array] = []
                ok = True
                for c in tracked_cache:
                    k = getattr(c, "keys", None)
                    v = getattr(c, "values", None)
                    off = getattr(c, "offset", None)
                    if k is None or v is None or off is None:
                        ok = False
                        break
                    layer_keys.append(k[..., :off, :])
                    layer_values.append(v[..., :off, :])
                if ok and layer_keys:
                    new_blocks = apc_manager.store_kv_blocks(
                        all_ids,
                        layer_keys,
                        layer_values,
                        extra_hash=apc_extra_hash,
                        skip_first_n_tokens=reused_prefix_len,
                    )
                    apc_manager.release(apc_blocks_in_use + new_blocks)
                else:
                    apc_manager.release(apc_blocks_in_use)
            except Exception as e:
                logger.warning("APC store failed: %s", e)
                apc_manager.release(apc_blocks_in_use)

        # Cleanup after generation
        mx.clear_cache()

def generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    audio: Union[str, List[str]] = None,
    video: Union[str, List[str]] = None,
    verbose: bool = False,
    **kwargs,
) -> GenerationResult:
    """
    Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temperature (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
       verbose (bool): If ``True``, print tokens and timing information
           (default ``False``).
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       repetition_penalty (float, optional): The penalty factor for repeating tokens.
       repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.
    """

    if verbose:
        print("=" * 10)
        files = []
        if image is not None:
            files.extend(image)
        if audio is not None:
            files.extend(audio)
        if video is not None:
            files.extend(video if isinstance(video, list) else [video])

        print(f"Files: {files}", "\n")

        print("Prompt:", prompt)

    text = ""
    last_response = None

    eos_tokens = kwargs.get("eos_tokens", None)
    stopping_criteria = kwargs.get("stopping_criteria", None)

    # Get the tokenizer
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Add custom EOS tokens to the stopping criteria
    if eos_tokens is not None:
        tokenizer.stopping_criteria.add_eos_token_ids(eos_tokens)

    # Use custom stopping criteria
    elif stopping_criteria is not None:
        if isinstance(stopping_criteria, StoppingCriteria) or callable(
            stopping_criteria
        ):
            tokenizer.stopping_criteria = stopping_criteria
        else:
            raise ValueError(
                "stopping_criteria must be an instance of StoppingCriteria or a callable"
            )
    else:
        tokenizer.stopping_criteria.reset(model.config.eos_token_id)

    for response in stream_generate(
        model, processor, prompt, image, audio, video, **kwargs
    ):
        if verbose:
            print(response.text, end="", flush=True)
        text += response.text
        last_response = response

    if verbose:
        print("\n" + "=" * 10)
        if len(text) == 0:
            print("No text generated for this prompt")
            return GenerationResult(
                text=text,
                token=None,
                logprobs=None,
                prompt_tokens=0,
                generation_tokens=0,
                total_tokens=0,
                prompt_tps=0.0,
                generation_tps=0.0,
                peak_memory=mx.get_peak_memory() / 1e9,
            )
        print(
            f"Prompt: {last_response.prompt_tokens} tokens, "
            f"{last_response.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {last_response.generation_tokens} tokens, "
            f"{last_response.generation_tps:.3f} tokens-per-sec"
        )
        print(f"Peak memory: {last_response.peak_memory:.3f} GB")

    return GenerationResult(
        text=text,
        token=last_response.token,
        logprobs=last_response.logprobs,
        prompt_tokens=last_response.prompt_tokens,
        generation_tokens=last_response.generation_tokens,
        total_tokens=last_response.total_tokens,
        prompt_tps=last_response.prompt_tps,
        generation_tps=last_response.generation_tps,
        peak_memory=last_response.peak_memory,
    )

