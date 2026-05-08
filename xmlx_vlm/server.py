import argparse
import asyncio
import gc
import json
import logging
import os
import re
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from queue import Empty as QueueEmpty
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, Iterator, List, Literal, Optional, Tuple, Union

logger = logging.getLogger("xmlx_vlm.server")

import mlx.core as mx

# Environment toggle for per-request cache clearing. Defaults to False because
# continuous batching relies on KV-cache/APC reuse and generate.py already
# clears periodically every 256–512 tokens.
_CLEAR_CACHE_PER_REQUEST = os.environ.get("XMLX_VLM_CLEAR_CACHE_PER_REQUEST", "false").lower() in ("1", "true", "yes")


def _maybe_clear_cache():
    """Clear MLX cache + GC only when the user explicitly opts in."""
    if _CLEAR_CACHE_PER_REQUEST:
        mx.clear_cache()
        gc.collect()
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from huggingface_hub import scan_cache_dir
from . import apc as _apc
from .server_schemas import (
    AnthropicMessageContent,
    AnthropicMessageRequest,
    AnthropicMessageResponse,
    AnthropicUsage,
    ChatChoice,
    ChatLogprobContent,
    ChatLogprobs,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChoice,
    ChatStreamChunk,
    ContentPartOutputText,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    MCPExecuteRequest,
    MessageItem,
    ModelsResponse,
    OpenAIRequest,
    OpenAIResponse,
    RerankDocument,
    RerankRequest,
    RerankResponse,
    RerankResult,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputTextDeltaEvent,
    ResponseOutputTextDoneEvent,
    TopLogprob,
    UsageStats,
    get_server_max_tokens,
)
from .generate import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    BatchGenerator,
    _dflash_rounds_batch,
    _make_cache,
    _merge_prefill_prompt_kwargs,
    _mtp_rounds_batch,
    generate,
    normalize_resize_shape,
    stream_generate,
)
from .prompt_utils import apply_chat_template
from .sample_utils import top_p_sampling
from .structured import build_json_schema_logits_processor
from .constrained import ThinkingAwareLogitsProcessor
from .generate import DEFAULT_THINKING_END_TOKEN, DEFAULT_THINKING_START_TOKEN
from .tokenizer_utils import make_streaming_detokenizer
from .tool_parsers import _infer_tool_parser_from_processor, load_tool_module
from .tool_logits_bias import ToolLogitsBiasProcessor
from .embedding_engine import EmbeddingEngine
from .rerank_engine import RerankEngine
from .metrics import metrics
from .optimizations import detect_hardware
from .prompt_warmup import load_warmup_file, warm_prompts
from .utils import load, prepare_inputs
from .version import __version__
from .vision_cache import VisionFeatureCache
from .mcp import get_manager
from .reasoning import get_parser as _get_reasoning_parser_cls
from .moe_topk import apply_moe_top_k_override
from .patches import apply_all_patches

DEFAULT_SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 8080
DEFAULT_TOKEN_QUEUE_TIMEOUT = 600.0
DEFAULT_FIRST_TOKEN_TIMEOUT = None  # No timeout for first token (prefill) by default
DEFAULT_ENABLE_THINKING = False
DEFAULT_ENABLE_TOOL_LOGITS_BIAS = False


def _get_speculative_rounds_batch(draft_kind: str):
    if draft_kind == "mtp":
        return _mtp_rounds_batch
    if draft_kind == "dflash":
        return _dflash_rounds_batch
    raise ValueError(f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'mtp']")


def _speculative_prefill_kwargs(draft_kind: str, drafter) -> dict:
    if draft_kind == "mtp":
        return {"return_hidden": True, "return_shared_kv": True}
    if draft_kind == "dflash":
        return {"capture_layer_ids": list(drafter.config.target_layer_ids)}
    raise ValueError(f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'mtp']")


def _speculative_hidden_state(draft_kind: str, outputs):
    if draft_kind == "mtp":
        return outputs.hidden_states[-1]
    if draft_kind == "dflash":
        return mx.concatenate(outputs.hidden_states, axis=-1)
    raise ValueError(f"Unknown draft_kind {draft_kind!r}. Supported: ['dflash', 'mtp']")


def _get_draft_block_size_from_env():
    draft_block_size_str = os.environ.get("XMLX_VLM_DRAFT_BLOCK_SIZE")
    return int(draft_block_size_str) if draft_block_size_str else None


def get_prefill_step_size():
    return int(os.environ.get("PREFILL_STEP_SIZE", DEFAULT_PREFILL_STEP_SIZE))




def _parse_timeout_env(env_name: str, default: Optional[float]) -> Optional[float]:
    raw_timeout = os.environ.get(env_name, "")
    if raw_timeout == "":
        return default
    try:
        timeout = float(raw_timeout)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to %ss.",
            env_name,
            raw_timeout,
            default,
        )
        return default
    if timeout <= 0:
        return None
    return timeout


def get_token_queue_timeout():
    return _parse_timeout_env("XMLX_VLM_TOKEN_QUEUE_TIMEOUT", DEFAULT_TOKEN_QUEUE_TIMEOUT)


def get_first_token_timeout():
    return _parse_timeout_env("XMLX_VLM_FIRST_TOKEN_TIMEOUT", DEFAULT_FIRST_TOKEN_TIMEOUT)


_ENABLE_TOOL_LOGITS_BIAS: Optional[bool] = None

def get_server_enable_tool_logits_bias():
    global _ENABLE_TOOL_LOGITS_BIAS
    if _ENABLE_TOOL_LOGITS_BIAS is not None:
        return _ENABLE_TOOL_LOGITS_BIAS
    env = os.environ.get("XMLX_VLM_ENABLE_TOOL_LOGITS_BIAS", "")
    return env.lower() in ("1", "true", "yes")

def set_server_enable_tool_logits_bias(value: bool):
    global _ENABLE_TOOL_LOGITS_BIAS
    _ENABLE_TOOL_LOGITS_BIAS = value

def get_server_enable_thinking():
    raw = os.environ.get("XMLX_VLM_ENABLE_THINKING")
    if raw is None:
        return DEFAULT_ENABLE_THINKING
    return raw.lower() in ("1", "true", "yes", "on")


def get_quantized_kv_bits(model: str):
    kv_bits = float(os.environ.get("KV_BITS", 0))
    if kv_bits == 0:
        return None
    if "qat" in model:
        print(f"Model {model} is quantization aware, KV cache will not be quantized.")
        return None
    return kv_bits


def get_kv_group_size():
    return int(os.environ.get("KV_GROUP_SIZE", DEFAULT_KV_GROUP_SIZE))


def get_kv_quant_scheme():
    return os.environ.get("KV_QUANT_SCHEME", DEFAULT_KV_QUANT_SCHEME)


def get_max_kv_size(model: str):
    max_kv_tokens = int(os.environ.get("MAX_KV_SIZE", 0))
    if max_kv_tokens == 0:
        return None
    if get_quantized_kv_bits(model) is not None:
        print(f"Model {model} uses QuantizedKVCache, can't set max KV size.")
        return None
    return max_kv_tokens


def get_quantized_kv_start():
    return int(os.environ.get("QUANTIZED_KV_START", DEFAULT_QUANTIZED_KV_START))


def get_top_logprobs_k():
    """Max per-token top_logprobs honored by the server (0 = disabled).

    Set via TOP_LOGPROBS_K env var. OpenAI caps this at 20. When 0, requests
    with top_logprobs>0 still succeed but the top_logprobs list stays empty.
    """
    k = int(os.environ.get("TOP_LOGPROBS_K", 0))
    return max(0, min(k, 20))


# =============================================================================
# ResponseGenerator - Concurrent Request Handling with Threaded Batching
# =============================================================================


@dataclass
class GenerationArguments:
    """Arguments for a generation request."""

    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = 0
    min_p: float = 0.0
    seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    logit_bias: Optional[dict] = None
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    thinking_budget: Optional[Union[int, str]] = None
    thinking_start_token: Optional[str] = None
    thinking_end_token: Optional[str] = None
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None
    # Per-tenant salt for APC. When set, it's mixed into ``extra_hash`` so
    # cached blocks from one tenant can't be reused (or detected via timing)
    # by another. None = no salt = single-tenant behaviour.
    tenant_id: Optional[str] = None
    enable_tool_logits_bias: bool = DEFAULT_ENABLE_TOOL_LOGITS_BIAS

    def to_generate_kwargs(self) -> dict:
        """Convert to kwargs dict for generate()/stream_generate()."""
        kw = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "enable_thinking": self.enable_thinking,
        }
        if self.repetition_penalty is not None:
            kw["repetition_penalty"] = self.repetition_penalty
        if self.logit_bias is not None:
            kw["logit_bias"] = self.logit_bias
        if self.thinking_budget is not None:
            kw["thinking_budget"] = self.thinking_budget
        if self.thinking_start_token is not None:
            kw["thinking_start_token"] = self.thinking_start_token
        if self.thinking_end_token is not None:
            kw["thinking_end_token"] = self.thinking_end_token
        if self.logits_processors is not None:
            kw["logits_processors"] = self.logits_processors
        if self.tenant_id is not None:
            kw["apc_tenant"] = self.tenant_id
        return kw

    def maybe_add_tool_logits_bias(self, tokenizer, tools_present: bool) -> None:
        """Add ToolLogitsBiasProcessor if enabled and tools are present."""
        if not self.enable_tool_logits_bias or not tools_present:
            return
        if self.logits_processors is None:
            self.logits_processors = []
        self.logits_processors.append(ToolLogitsBiasProcessor(tokenizer))

    def to_template_kwargs(self) -> dict:
        """Convert to kwargs for apply_chat_template()."""
        kw = {"enable_thinking": self.enable_thinking}
        if self.thinking_budget is not None:
            kw["thinking_budget"] = self.thinking_budget
        if self.thinking_start_token is not None:
            kw["thinking_start_token"] = self.thinking_start_token
        if self.thinking_end_token is not None:
            kw["thinking_end_token"] = self.thinking_end_token
        return kw


@dataclass
class GenerationContext:
    """Context returned when a request is queued."""

    uid: int
    prompt_tokens: int


@dataclass
class StreamingToken:
    """A single token response during streaming generation."""

    text: str
    token: int
    logprobs: float
    finish_reason: Optional[str]
    peak_memory: float = 0.0
    top_logprobs: Optional[List[Tuple[int, float]]] = None


class ResponseGenerator:
    """
    Continuous batching for concurrent requests via a single GPU thread.

    A dedicated thread owns all GPU work (BatchGenerator). FastAPI async
    handlers submit requests to a queue and read tokens back from
    per-request queues. Multiple requests are batched together for
    higher throughput — same pattern as mlx-lm's server.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        vision_cache=None,
        kv_bits=None,
        kv_group_size=DEFAULT_KV_GROUP_SIZE,
        kv_quant_scheme=DEFAULT_KV_QUANT_SCHEME,
        quantized_kv_start=DEFAULT_QUANTIZED_KV_START,
        top_logprobs_k=0,
        apc_manager: Optional["_apc.APCManager"] = None,
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.model = None
        self.processor = None
        self.config = None
        self.stop_tokens = set()
        self.vision_cache = vision_cache
        self.draft_model = None
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.kv_quant_scheme = kv_quant_scheme
        self.quantized_kv_start = quantized_kv_start
        self.top_logprobs_k = top_logprobs_k
        self.apc_manager = apc_manager
        self.tokenizer = None
        self.requests: Queue = Queue()
        self._stop = False
        self._ready = Event()
        self._load_error: Optional[Exception] = None
        self._cancelled: set = set()
        self._cancel_lock = Lock()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop_and_join(self):
        self._stop = True
        self.requests.put(None)
        self._thread.join(timeout=5.0)

    def wait_until_ready(self, timeout: Optional[float] = None):
        if not self._ready.wait(timeout):
            raise RuntimeError("Timed out waiting for generation thread to load model.")
        if self._load_error is not None:
            raise self._load_error
        return self.model, self.processor, self.config

    def _cancel(self, uid):
        with self._cancel_lock:
            self._cancelled.add(uid)

    def _drain_cancellations(self) -> set:
        with self._cancel_lock:
            pending, self._cancelled = self._cancelled, set()
            return pending

    def _initialize_model(self):
        model, processor, config = load_model_resources(
            self.model_path, self.adapter_path
        )

        stop_tokens = set()
        if hasattr(config, "eos_token_id"):
            if isinstance(config.eos_token_id, list):
                stop_tokens.update(config.eos_token_id)
            elif config.eos_token_id is not None:
                stop_tokens.add(config.eos_token_id)

        draft_model = None
        draft_kind = os.environ.get("XMLX_VLM_DRAFT_KIND")
        draft_model_path = os.environ.get("XMLX_VLM_DRAFT_MODEL")
        if draft_model_path:
            from .speculative.drafters import load_drafter

            print(
                f"Loading speculative drafter ({draft_kind or 'auto'}): "
                f"{draft_model_path}"
            )
            draft_model, resolved_kind = load_drafter(draft_model_path, kind=draft_kind)
            if draft_kind is None:
                print(f"  → auto-detected --draft-kind={resolved_kind!r}.")
            elif resolved_kind != draft_kind:
                print(
                    f"  → drafter requires --draft-kind={resolved_kind!r}; "
                    f"using {resolved_kind!r} instead of {draft_kind!r}."
                )
            draft_kind = resolved_kind
            print("Drafter ready — speculative decoding enabled.")

        self.model = model
        self.processor = processor
        self.config = config
        self.stop_tokens = stop_tokens
        self.draft_model = draft_model
        self.draft_kind = draft_kind
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )

    def generate(
        self,
        prompt: str,
        images: Optional[List] = None,
        audio: Optional[List] = None,
        args: Optional[GenerationArguments] = None,
    ) -> Tuple[GenerationContext, Iterator[StreamingToken]]:
        self.wait_until_ready()
        args = args or GenerationArguments(max_tokens=get_server_max_tokens())
        if self.draft_model is not None and args.logits_processors is not None:
            raise ValueError(
                "Structured response_format is not supported with speculative decoding."
            )
        rqueue: Queue = Queue()

        # CPU preprocessing (tokenize, load images) on caller thread.
        # GPU work (vision encoder) deferred to GPU thread.
        raw_inputs = self._cpu_preprocess(prompt, images, audio)
        prompt_tokens = (
            raw_inputs["input_ids"].size
            if hasattr(raw_inputs["input_ids"], "size")
            else len(raw_inputs["input_ids"])
        )

        self.requests.put((rqueue, raw_inputs, prompt_tokens, args, images))

        # Block until the GPU thread sends back the context
        ctx = rqueue.get()
        if isinstance(ctx, Exception):
            raise ctx

        uid = ctx.uid

        def token_iterator():
            # Mark ended before yielding the final token so a consumer that
            # closes immediately after seeing finish_reason isn't treated
            # as a client abort.
            ended = False
            first_token_timeout = get_first_token_timeout()
            queue_timeout = get_token_queue_timeout()
            is_first_token = True
            try:
                while True:
                    current_timeout = first_token_timeout if is_first_token else queue_timeout
                    try:
                        item = rqueue.get(timeout=current_timeout)
                    except QueueEmpty as exc:
                        timeout_label = (
                            "without a timeout"
                            if current_timeout is None
                            else f"for {current_timeout:g}s"
                        )
                        if is_first_token:
                            raise RuntimeError(
                                "Timed out waiting "
                                f"{timeout_label} for the first generated token. "
                                "Increase XMLX_VLM_FIRST_TOKEN_TIMEOUT for long "
                                "prefills, or reduce the prompt size."
                            ) from exc
                        raise RuntimeError(
                            "Timed out waiting "
                            f"{timeout_label} for the next generated token. "
                            "Increase XMLX_VLM_TOKEN_QUEUE_TIMEOUT for long "
                            "generation steps, or reduce the prompt size."
                        ) from exc
                    if item is None:
                        ended = True
                        break
                    if isinstance(item, Exception):
                        ended = True
                        raise item
                    if getattr(item, "finish_reason", None):
                        ended = True
                    yield item
                    if ended:
                        break
                    is_first_token = False
            finally:
                if not ended:
                    self._cancel(uid)

        return ctx, token_iterator()

    def _cpu_preprocess(self, prompt, images=None, audio=None) -> dict:
        """CPU-only: tokenize text, load/resize images. Thread-safe."""
        add_special_tokens = (
            getattr(self.processor, "chat_template", None) is None
            if self.model.config.model_type in ["gemma3", "gemma3n", "gemma4"]
            else True
        )
        image_token_index = getattr(self.model.config, "image_token_index", None)
        return prepare_inputs(
            self.processor,
            images=images,
            audio=audio,
            prompts=prompt,
            image_token_index=image_token_index,
            add_special_tokens=add_special_tokens,
        )

    # -- internals --

    def _make_sampler(self, args: GenerationArguments) -> Optional[Callable]:
        if args.temperature == 0:
            return None

        def sampler(logprobs: mx.array) -> mx.array:
            if args.top_p > 0 and args.top_p < 1.0:
                return top_p_sampling(logprobs, args.top_p, args.temperature)
            else:
                return mx.random.categorical(logprobs * (1 / args.temperature))

        return sampler

    def _gpu_embed(self, raw_inputs: dict, images=None) -> Tuple[mx.array, dict]:
        """GPU-only: run vision encoder if needed. Must run on GPU thread."""
        input_ids = raw_inputs.get("input_ids")
        pixel_values = raw_inputs.get("pixel_values")
        mask = raw_inputs.get("attention_mask")
        data_kwargs = {
            k: v
            for k, v in raw_inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        # Pass vision cache for image feature caching
        if (
            pixel_values is not None
            and self.vision_cache is not None
            and images is not None
        ):
            data_kwargs["vision_cache"] = self.vision_cache
            data_kwargs["_image_key"] = images

        # Always call get_input_embeddings — BatchGenerator requires inputs_embeds
        embed = self.model.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **data_kwargs
        )
        # Remove cache kwargs before passing to BatchGenerator
        data_kwargs.pop("vision_cache", None)
        data_kwargs.pop("_image_key", None)
        gen_kwargs = {**data_kwargs, **embed.to_dict()}
        if images is not None:
            gen_kwargs["_apc_image_hash"] = _apc.hash_image_payload(image_ref=images)
        elif pixel_values is not None:
            gen_kwargs["_apc_image_hash"] = _apc.hash_image_payload(
                pixel_values=pixel_values
            )
        return input_ids, gen_kwargs

    def _run(self):
        """Single GPU thread: owns BatchGenerator, runs tight next() loop."""
        try:
            self._initialize_model()
        except Exception as e:
            self._load_error = e
            self._ready.set()
            print(f"Error loading model in generation thread: {e}")
            traceback.print_exc()
            return

        self._ready.set()

        if self.draft_model is not None:
            self._run_speculative()
            return

        generation_stream = mx.default_stream(mx.default_device())

        batch_gen = None
        # uid -> {rqueue, tokens, gen_kwargs}
        active: dict = {}

        while not self._stop:
            try:
                # Poll the request queue — non-blocking when generating, short
                # blocking wait when idle so we don't spin.
                new_items = []
                if active:
                    try:
                        item = self.requests.get_nowait()
                        if item is None:
                            if self._stop:
                                break
                        else:
                            new_items.append(item)
                    except QueueEmpty:
                        pass
                else:
                    try:
                        item = self.requests.get(timeout=0.1)
                        if item is None:
                            if self._stop:
                                break
                        else:
                            new_items.append(item)
                    except QueueEmpty:
                        pass

                while True:
                    try:
                        item = self.requests.get_nowait()
                        if item is not None:
                            new_items.append(item)
                    except QueueEmpty:
                        break

                # Drop abandoned requests before doing more work.
                cancelled = self._drain_cancellations()
                if cancelled and batch_gen is not None:
                    for uid in cancelled:
                        if uid in active:
                            batch_gen.remove(uid)
                            info = active.pop(uid)
                            try:
                                info["rqueue"].put(None)
                            except Exception:
                                pass

                for rqueue, raw_inputs, prompt_tokens, args, images in new_items:
                    if batch_gen is None:
                        batch_gen = BatchGenerator(
                            self.model.language_model,
                            self.processor,
                            stop_tokens=self.stop_tokens,
                            sampler=self._make_sampler(args),
                            kv_bits=self.kv_bits,
                            kv_group_size=self.kv_group_size,
                            kv_quant_scheme=self.kv_quant_scheme,
                            quantized_kv_start=self.quantized_kv_start,
                            top_logprobs_k=self.top_logprobs_k,
                            stream=generation_stream,
                            apc_manager=self.apc_manager,
                        )

                    # Vision encoder runs on the GPU thread; text tokenization
                    # already happened on the caller thread.
                    input_ids, gen_kwargs = self._gpu_embed(raw_inputs, images)
                    has_embeds = bool(gen_kwargs.get("inputs_embeds") is not None)
                    # Per-tenant APC salt: keep this out of the model forward
                    # by namespacing under "_apc_tenant"; BatchGenerator strips
                    # it before merging kwargs for the language model.
                    if getattr(args, "tenant_id", None):
                        gen_kwargs["_apc_tenant"] = args.tenant_id

                    # Drain pending text-only prompts before inserting an
                    # embed-bearing request — multi-row PromptProcessingBatch
                    # admission expects all rows to carry inputs_embeds (the
                    # mixed APC path concatenates them per-row).
                    if has_embeds and any(
                        not (s[3] and s[3].get("inputs_embeds") is not None)
                        for s in batch_gen.unprocessed_prompts
                    ):
                        self._flush(batch_gen, active)

                    try:
                        (uid,) = batch_gen.insert(
                            [input_ids.squeeze(0).tolist()],
                            max_tokens=args.max_tokens,
                            prompt_kwargs=[gen_kwargs],
                            logits_processors=[args.logits_processors],
                        )
                    except Exception as e:
                        rqueue.put(e)
                        continue

                    rqueue.put(GenerationContext(uid=uid, prompt_tokens=prompt_tokens))
                    active[uid] = {
                        "rqueue": rqueue,
                        "detokenizer": make_streaming_detokenizer(self.processor),
                        "gen_kwargs": gen_kwargs if has_embeds else None,
                    }

                if not active or batch_gen is None:
                    continue

                self._step(batch_gen, active)

            except Exception as e:
                logger.exception("Error in generation thread")
                for info in list(active.values()):
                    try:
                        info["rqueue"].put(e)
                        info["rqueue"].put(None)
                    except Exception:
                        pass
                active.clear()
                batch_gen = None
                mx.clear_cache()
                gc.collect()

    def _run_speculative(self):
        """GPU thread loop with DFlash or Gemma 4 MTP speculative decoding.

        Collects incoming requests, prefills them as a batch with the
        per-family hooks (``capture_layer_ids`` for DFlash; ``return_hidden``
        + ``return_shared_kv`` for MTP), then runs the matching round-loop
        for decode. Finished sequences are filtered out automatically by
        the round-loop's ``stop_check`` callback.
        """
        from mlx_lm.sample_utils import make_sampler as _make_sampler

        generation_stream = mx.default_stream(mx.default_device())

        lm = self.model.language_model
        drafter = self.draft_model
        draft_kind = self.draft_kind
        is_mtp = draft_kind == "mtp"
        rounds_batch = _get_speculative_rounds_batch(draft_kind)
        prefill_kwargs = _speculative_prefill_kwargs(draft_kind, drafter)
        eos_set = set(self.stop_tokens) if is_mtp else None
        sampler = _make_sampler(temp=0)
        draft_block_size = _get_draft_block_size_from_env()

        while not self._stop:
            try:
                # --- Phase 1: collect pending requests ---
                pending = []
                timeout = 0.1
                try:
                    item = self.requests.get(timeout=timeout)
                    if item is None and self._stop:
                        break
                    if item is not None:
                        pending.append(item)
                except QueueEmpty:
                    pass
                while True:
                    try:
                        item = self.requests.get_nowait()
                        if item is not None:
                            pending.append(item)
                    except QueueEmpty:
                        break

                if not pending:
                    continue

                # --- Phase 2: prefill new batch ---
                uids = []
                rqueues = {}
                token_lists = {}
                stream_infos = {}
                max_tokens_map = {}
                all_input_ids = []
                prompt_kwargs_list = []

                if hasattr(lm, "_position_ids"):
                    lm._position_ids = None
                if hasattr(lm, "_rope_deltas"):
                    lm._rope_deltas = None

                for rqueue, raw_inputs, prompt_tokens, args, images in pending:
                    input_ids, gen_kwargs = self._gpu_embed(raw_inputs, images)
                    uid = id(rqueue)
                    uids.append(uid)
                    rqueues[uid] = rqueue
                    token_lists[uid] = []
                    stream_infos[uid] = {
                        "detokenizer": make_streaming_detokenizer(self.processor)
                    }
                    max_tokens_map[uid] = args.max_tokens
                    all_input_ids.append(input_ids.squeeze(0).tolist())
                    prompt_kwargs_list.append(gen_kwargs)
                    rqueue.put(GenerationContext(uid=uid, prompt_tokens=prompt_tokens))
                    sampler = self._make_sampler(args) or _make_sampler(temp=0)

                B = len(uids)
                max_len = max(len(ids) for ids in all_input_ids)
                left_padding = [max_len - len(ids) for ids in all_input_ids]
                padded = [
                    [0] * left_padding[i] + ids for i, ids in enumerate(all_input_ids)
                ]
                input_mx = mx.array(padded, dtype=mx.int32)

                inputs_embeds_mx, prompt_kwargs = _merge_prefill_prompt_kwargs(
                    prompt_kwargs_list, all_input_ids
                )

                prompt_cache = _make_cache(lm, left_padding)

                lm_call_kwargs = {**prefill_kwargs, **prompt_kwargs}
                lm_call_kwargs["inputs_embeds"] = inputs_embeds_mx

                with mx.stream(generation_stream):
                    out = lm(input_mx, cache=prompt_cache, **lm_call_kwargs)
                hidden = _speculative_hidden_state(draft_kind, out)
                shared_kv_states = out.shared_kv_states if is_mtp else None
                first_bonus = sampler(out.logits[:, -1:]).squeeze(-1)
                mx.eval(first_bonus, hidden)

                finished_uids = set()

                # Send first bonus tokens to clients
                fb_list = first_bonus.tolist()
                for j, uid in enumerate(uids):
                    tok = int(fb_list[j])
                    token_lists[uid].append(tok)
                    is_stop = tok in self.stop_tokens
                    is_max = len(token_lists[uid]) >= max_tokens_map[uid]
                    finish = "stop" if is_stop else "length" if is_max else None
                    text = self._stream_text(stream_infos[uid], tok, finish)
                    rqueues[uid].put(
                        StreamingToken(
                            text=text,
                            token=tok,
                            logprobs=0.0,
                            finish_reason=finish,
                            peak_memory=mx.get_peak_memory() / 1e9,
                        )
                    )
                    if finish is not None:
                        rqueues[uid].put(None)
                        finished_uids.add(uid)

                if len(finished_uids) == len(uids):
                    continue

                # --- Phase 3: speculative decode rounds ---
                max_tok = max(max_tokens_map[u] for u in uids)

                def stop_check(seq_idx, token_id):
                    uid = uids[seq_idx]
                    if uid in finished_uids:
                        return True
                    if token_id in self.stop_tokens:
                        return True
                    if len(token_lists[uid]) >= max_tokens_map[uid]:
                        return True
                    return False

                rounds_kwargs = dict(
                    first_bonus=first_bonus,
                    max_tokens=max_tok,
                    sampler=sampler,
                    draft_block_size=draft_block_size,
                    token_dtype=mx.int32,
                    stop_check=stop_check,
                )
                if is_mtp:
                    rounds_iter = rounds_batch(
                        self.model,
                        drafter,
                        prompt_cache,
                        hidden,
                        shared_kv_states,
                        eos_token_ids=eos_set,
                        **rounds_kwargs,
                    )
                else:
                    rounds_iter = rounds_batch(
                        self.model,
                        drafter,
                        prompt_cache,
                        hidden,
                        **rounds_kwargs,
                    )
                for tok_list, _ in rounds_iter:
                    for j, tok in enumerate(tok_list):
                        if tok is None:
                            continue
                        uid = uids[j]
                        if uid in finished_uids:
                            continue

                        token_lists[uid].append(tok)
                        tokens = token_lists[uid]

                        is_stop = tok in self.stop_tokens
                        is_max = len(tokens) >= max_tokens_map[uid]
                        finish = "stop" if is_stop else "length" if is_max else None
                        text = self._stream_text(stream_infos[uid], tok, finish)

                        rqueues[uid].put(
                            StreamingToken(
                                text=text,
                                token=tok,
                                logprobs=0.0,
                                finish_reason=finish,
                                peak_memory=mx.get_peak_memory() / 1e9,
                            )
                        )

                        if finish is not None:
                            rqueues[uid].put(None)
                            finished_uids.add(uid)

                # Log acceptance stats
                al = drafter.accept_lens
                if al:
                    mean_a = sum(al) / len(al)
                    print(
                        f"[{'MTP' if is_mtp else 'DFlash'}] batch={B} "
                        f"tokens={sum(len(token_lists[u]) for u in uids)} "
                        f"accept={mean_a:.2f} rounds={len(al)}"
                    )

                # Finalize any remaining
                for uid in uids:
                    if uid not in finished_uids:
                        stream_infos[uid]["detokenizer"].finalize()
                        text = stream_infos[uid]["detokenizer"].last_segment
                        rqueues[uid].put(
                            StreamingToken(
                                text=text,
                                token=0,
                                logprobs=0.0,
                                finish_reason="length",
                                peak_memory=mx.get_peak_memory() / 1e9,
                            )
                        )
                        rqueues[uid].put(None)

            except Exception as e:
                print(f"Error in speculative generation thread: {e}")
                traceback.print_exc()

    def _step(self, batch_gen, active, gen_kwargs=None):
        """One batch generation step: prefill + decode."""
        kwargs = gen_kwargs or {}
        _, responses = batch_gen.next(**kwargs)
        if not responses:
            return

        for r in responses:
            if r.uid not in active:
                continue

            info = active[r.uid]
            rqueue = info["rqueue"]

            tok = r.token
            if hasattr(tok, "item"):
                tok = tok.item()

            text = self._stream_text(info, tok, r.finish_reason)

            lp = r.token_logprob

            rqueue.put(
                StreamingToken(
                    text=text,
                    token=tok,
                    logprobs=lp,
                    finish_reason=r.finish_reason,
                    peak_memory=mx.get_peak_memory() / 1e9 if r.finish_reason else 0,
                    top_logprobs=getattr(r, "top_logprobs", None),
                )
            )

            if r.finish_reason is not None:
                rqueue.put(None)
                del active[r.uid]

    def _stream_text(self, info: dict, token: int, finish_reason: Optional[str]) -> str:
        """Convert one generated token into a streaming text segment."""
        detokenizer = info["detokenizer"]
        if finish_reason == "stop":
            detokenizer.finalize()
        else:
            detokenizer.add_token(token)
            if finish_reason is not None:
                detokenizer.finalize()
        return detokenizer.last_segment

    def _flush(self, batch_gen, active):
        """Drain all pending text-only prompts before inserting an image request."""
        while batch_gen.has_pending_prompts:
            self._step(batch_gen, active)


def suppress_tool_call_content(
    full_output: str,
    in_tool_call: bool,
    tc_start: Optional[str],
    delta_content: Optional[str],
) -> Tuple[bool, Optional[str]]:
    """Suppress tool-call markup from streamed delta.content.

    Returns updated (in_tool_call, delta_content).
    """
    if not tc_start:
        return in_tool_call, delta_content
    if not in_tool_call:
        if tc_start in full_output:
            return True, None
        if any(full_output.endswith(tc_start[:j]) for j in range(1, len(tc_start))):
            return False, None
    else:
        return True, None
    return in_tool_call, delta_content


def process_tool_calls(model_output: str, tool_module, tools):
    """Parse tool calls from model output using the appropriate tool parser."""
    called_tools = []
    remaining = model_output

    if tool_module.tool_call_start in model_output:
        if tool_module.tool_call_end == "":
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?(?:\n|$)", re.DOTALL
            )
        else:
            pattern = re.compile(
                f"{re.escape(tool_module.tool_call_start)}.*?{re.escape(tool_module.tool_call_end)}",
                re.DOTALL,
            )

        matches = re.findall(pattern, model_output)
        if matches:
            remaining = re.sub(pattern, " ", model_output).strip()
            for i, match in enumerate(matches):
                call = (
                    match.strip()
                    .removeprefix(tool_module.tool_call_start)
                    .removesuffix(tool_module.tool_call_end)
                )
                try:
                    tool_call = tool_module.parse_tool_call(call, tools)
                    args = tool_call["arguments"]
                    called_tools.append(
                        {
                            "type": "function",
                            "index": i,
                            "id": str(uuid.uuid4()),
                            "function": {
                                "name": tool_call["name"].strip(),
                                "arguments": (
                                    args
                                    if isinstance(args, str)
                                    else json.dumps(args, ensure_ascii=False)
                                ),
                            },
                        }
                    )
                except Exception as exc:
                    logger.warning("Invalid tool call: %s | error: %s", call, exc)
                    # Append a visible marker to remaining_text so the model
                    # (and client) knows parsing failed instead of silently
                    # dropping the call.
                    err_marker = f"[Tool call parsing failed: {exc}]"
                    remaining = f"{remaining}\n{err_marker}".strip()
    return dict(calls=called_tools, remaining_text=remaining)


def _build_gen_args(
    request, processor=None, tenant_id: Optional[str] = None
) -> GenerationArguments:
    """Build GenerationArguments from an OpenAIRequest or ChatRequest."""
    max_tokens = getattr(request, "max_tokens", None)
    if max_tokens is None:
        max_tokens = getattr(request, "max_output_tokens", None)
    if max_tokens is None:
        max_tokens = get_server_max_tokens()
    logit_bias = getattr(request, "logit_bias", None)
    if logit_bias is not None and isinstance(logit_bias, dict):
        logit_bias = {int(k): v for k, v in logit_bias.items()}
    enable_thinking = _request_field_or_default(
        request,
        "enable_thinking",
        get_server_enable_thinking(),
    )
    enable_tool_logits_bias = _request_field_or_default(
        request,
        "enable_tool_logits_bias",
        get_server_enable_tool_logits_bias(),
    )
    # reasoning_effort maps to thinking_budget when the latter is unset
    thinking_budget = getattr(request, "thinking_budget", None)
    reasoning_effort = getattr(request, "reasoning_effort", None)
    if thinking_budget is None and reasoning_effort is not None:
        thinking_budget = reasoning_effort
    args = GenerationArguments(
        max_tokens=max_tokens,
        temperature=getattr(request, "temperature", DEFAULT_TEMPERATURE),
        top_p=getattr(request, "top_p", DEFAULT_TOP_P),
        top_k=getattr(request, "top_k", 0),
        min_p=getattr(request, "min_p", 0.0),
        repetition_penalty=getattr(request, "repetition_penalty", None),
        logit_bias=logit_bias,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        thinking_start_token=getattr(request, "thinking_start_token", None),
        thinking_end_token=getattr(request, "thinking_end_token", None),
        tenant_id=tenant_id,
        enable_tool_logits_bias=enable_tool_logits_bias,
    )
    if processor is not None:
        args.logits_processors = _build_structured_logits_processors(request, processor, args)
        # Jump-forward decoding: bias tool-call tokens when tools are present
        tools_present = bool(getattr(request, "tools", None))
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        args.maybe_add_tool_logits_bias(tokenizer, tools_present)
    return args


def _request_field_or_default(request, field_name: str, default):
    fields_set = getattr(request, "model_fields_set", None)
    if fields_set is not None and field_name not in fields_set:
        return default
    value = getattr(request, field_name, default)
    return default if value is None else value


def _read_tenant_id(http_request) -> Optional[str]:
    """Pull a per-tenant APC salt from the request headers.

    Honoured headers (in order): ``X-APC-Tenant``, ``X-Tenant-Id``.
    """
    if http_request is None or not hasattr(http_request, "headers"):
        return None
    h = http_request.headers
    return h.get("x-apc-tenant") or h.get("x-tenant-id") or None


def _as_plain_dict(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return value


def _extract_response_format_schema(request) -> Optional[Union[str, dict]]:
    response_format = _as_plain_dict(getattr(request, "response_format", None))

    text_config = _as_plain_dict(getattr(request, "text", None))
    if response_format is None and isinstance(text_config, dict):
        response_format = _as_plain_dict(text_config.get("format"))

    if response_format is None:
        return None

    format_type = response_format.get("type")
    if format_type in (None, "text"):
        return None
    if format_type != "json_schema":
        raise ValueError(f"Unsupported response_format type: {format_type!r}")

    json_schema = _as_plain_dict(response_format.get("json_schema"))
    if json_schema is None:
        # Responses API text.format places schema directly on the format object.
        json_schema = response_format

    schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
    if schema is None:
        raise ValueError("response_format json_schema must include a schema field")
    return schema


def _build_structured_logits_processors(request, processor, gen_args):
    schema = _extract_response_format_schema(request)
    if schema is None:
        return None

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    inner = build_json_schema_logits_processor(tokenizer, schema)

    # If thinking is enabled alongside structured output, wrap the schema
    # processor with a thinking-aware lifecycle manager so the model can
    # reason freely before the schema constraint kicks in.
    _budget_val = gen_args.thinking_budget
    _has_budget = False
    if isinstance(_budget_val, int):
        _has_budget = _budget_val > 0
    elif isinstance(_budget_val, str):
        _has_budget = _budget_val.lower() not in ("off", "disabled", "none", "")
    if gen_args.enable_thinking or _has_budget:
        start_token = gen_args.thinking_start_token or DEFAULT_THINKING_START_TOKEN
        end_token = gen_args.thinking_end_token or DEFAULT_THINKING_END_TOKEN
        try:
            start_token_ids = tokenizer.encode(start_token, add_special_tokens=False)
            end_token_ids = tokenizer.encode(end_token, add_special_tokens=False)
        except Exception:
            # Fallback: if tokenizer can't encode the tag strings,
            # return the plain inner processor without thinking awareness.
            return [inner]

        budget = gen_args.thinking_budget if gen_args.thinking_budget is not None else 0
        # When enable_thinking is True, the chat template injects <think>
        # into the prompt, so the first generated token is already inside
        # the thinking span.
        prompt_has_think_tag = bool(gen_args.enable_thinking)

        # Best-effort prompt token count for adaptive budget scaling
        prompt_token_count = 0
        try:
            messages = getattr(request, "messages", None) or getattr(request, "input", None)
            if messages is not None:
                if isinstance(messages, str):
                    prompt_token_count = len(tokenizer.encode(messages))
                elif isinstance(messages, list):
                    texts = []
                    for m in messages:
                        content = getattr(m, "content", None)
                        if isinstance(content, str):
                            texts.append(content)
                        elif isinstance(content, list):
                            texts.append("".join(str(c) for c in content))
                    prompt_token_count = len(tokenizer.encode("\n".join(texts)))
        except Exception:
            pass

        thinking_processor = ThinkingAwareLogitsProcessor(
            start_token_ids=list(start_token_ids),
            end_token_ids=list(end_token_ids),
            thinking_token_budget=budget,
            inner=inner,
            vocab_size=getattr(tokenizer, "vocab_size", 152064),
            prompt_has_think_tag=prompt_has_think_tag,
            prompt_token_count=prompt_token_count,
        )
        return [thinking_processor]

    return [inner]


def _infer_reasoning_parser(model, config) -> str | None:
    """Infer reasoning parser name from model config."""
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    # Try explicit model_type mapping first
    mapping = {
        "qwen3": "qwen3",
        "qwen3_5": "qwen3",
        "qwen3_moe": "qwen3",
        "qwen3_5_moe": "qwen3",
        "qwen3_vl": "qwen3",
        "qwen3_vl_moe": "qwen3",
        "deepseek_v3": "deepseek_r1",
        "gemma4": "gemma4",
        "glm4": "glm4",
        "gpt_oss": "gpt_oss",
        "harmony": "harmony",
    }
    if model_type in mapping:
        return mapping[model_type]
    # Fallback: heuristic from model path or config name
    name = getattr(config, "model_type", "") or getattr(config, "architectures", [""])[0]
    name = name.lower()
    if "qwen3" in name:
        return "qwen3"
    if "deepseek" in name:
        return "deepseek_r1"
    if "gemma4" in name or "gemma-4" in name:
        return "gemma4"
    if "glm4" in name or "glm-4" in name:
        return "glm4"
    return None


def _count_thinking_tag_tokens(text: str) -> int:
    """Count tokens consumed by thinking tags (excluded from completion_tokens)."""
    count = 0
    # <|channel>thought (2 tokens) + <channel|> (1 token) + EOS (1 token)
    if "<|channel>thought" in text and "<channel|>" in text:
        count = 4
    elif "<think>" in text and "</think>" in text:
        count = 2  # <think> and </think> are 1 token each typically
    return count


def _split_thinking(text: str, parser=None) -> Tuple[Optional[str], str]:
    """Split thinking tags from content. Returns (reasoning, content).

    If a reasoning parser is provided, uses the parser for extraction.
    Otherwise falls back to heuristic string splitting.
    """
    if parser is not None:
        try:
            return parser.extract_reasoning(text)
        except Exception:
            pass  # Fall through to heuristic

    # Handle <|channel>thought...<channel|> format (gemma4)
    if "<|channel>thought" in text or (
        "<channel|>" in text and text.lstrip().startswith("thought")
    ):
        parts = text.split("<channel|>", 1)
        if len(parts) == 2:
            reasoning = (
                parts[0].replace("<|channel>thought", "").lstrip("thought").strip()
            )
            content = parts[1].strip()
            return reasoning or None, content
        reasoning = parts[0].replace("<|channel>thought", "").lstrip("thought").strip()
        return reasoning or None, ""
    # Handle <think>...</think> format (qwen3.5 etc)
    if "<think>" in text or "</think>" in text:
        parts = text.split("</think>", 1)
        if len(parts) == 2:
            reasoning = parts[0].replace("<think>", "").strip()
            content = parts[1].strip()
            return reasoning or None, content
        return parts[0].replace("<think>", "").strip(), ""
    return None, text


def _decode_token(tokenizer, token_id: int) -> Tuple[str, Optional[List[int]]]:
    """Decode a single token id to its string + UTF-8 bytes."""
    try:
        text = tokenizer.decode([int(token_id)])
    except Exception:
        text = ""
    try:
        token_bytes = list(text.encode("utf-8"))
    except Exception:
        token_bytes = None
    return text, token_bytes


def _make_logprob_content(
    tokenizer,
    token_id: int,
    logprob: float,
    top_logprobs: Optional[List[Tuple[int, float]]] = None,
    top_k: int = 0,
) -> "ChatLogprobContent":
    """Build an OpenAI-style logprob entry for a single token."""
    token_text, token_bytes = _decode_token(tokenizer, token_id)
    top_list: List[TopLogprob] = []
    if top_k > 0 and top_logprobs:
        for tid, lp in top_logprobs[:top_k]:
            t_text, t_bytes = _decode_token(tokenizer, tid)
            top_list.append(TopLogprob(token=t_text, logprob=float(lp), bytes=t_bytes))
    return ChatLogprobContent(
        token=token_text,
        logprob=float(logprob),
        bytes=token_bytes,
        top_logprobs=top_list,
    )


# Global response generator for continuous batching
response_generator: Optional[ResponseGenerator] = None

# Global APC manager (shared across requests for the loaded model)
apc_manager: Optional[_apc.APCManager] = None

# Optional embedding / rerank engines (lazy-loaded)
_embedding_engine: Optional[EmbeddingEngine] = None
_rerank_engine: Optional[RerankEngine] = None


# Loading/unloading utilities
model_cache = {}


@asynccontextmanager
async def lifespan(app):
    model_path = os.environ.pop("XMLX_VLM_PRELOAD_MODEL", None)
    if model_path:
        adapter_path = os.environ.pop("XMLX_VLM_PRELOAD_ADAPTER", None)
        logger.info("Pre-loading model: %s", model_path)
        model, processor, config = get_cached_model(model_path, adapter_path)
        kv_bits = os.environ.get("KV_BITS")
        kv_scheme = os.environ.get("KV_QUANT_SCHEME", "uniform")
        if kv_bits:
            logger.info("KV cache quantization: bits=%s scheme=%s", kv_bits, kv_scheme)
        logger.info("Model ready, continuous batching enabled.")

        # Prompt warmup for coding assistants
        warmup_path = os.environ.get("XMLX_VLM_WARMUP_PROMPTS")
        if warmup_path:
            try:
                prompts = load_warmup_file(warmup_path)
                warm_prompts(model, processor, prompts)
            except Exception as e:
                logger.warning("Prompt warmup failed: %s", e)
    # Connect external MCP servers if configured
    mcp_config_path = os.environ.pop("MLX_MCP_CONFIG", None)
    if mcp_config_path:
        try:
            mcp_manager = get_manager()
            await mcp_manager.connect_external_servers(mcp_config_path)
        except Exception as e:
            logger.warning("Failed to connect external MCP servers: %s", e)
    yield
    # Flush APC blocks to disk so prefix caches survive server restarts
    global apc_manager
    if apc_manager is not None:
        try:
            n_flushed = apc_manager.flush_to_disk()
            if n_flushed:
                logger.info("APC: flushed %d blocks to disk before shutdown", n_flushed)
        except Exception as e:
            logger.warning("APC flush on shutdown failed: %s", e)
    # Cleanup MCP connections on shutdown
    try:
        mcp_manager = get_manager()
        await mcp_manager.disconnect_all()
    except Exception:
        pass


app = FastAPI(
    title="MLX-VLM Inference API",
    description="API for using Vision Language Models (VLMs) and Omni Models (Vision, Audio and Video support) with MLX.",
    version=__version__,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_IMAGES = 10  # Maximum number of images to process at once

# API Key authentication (optional)
_API_KEY: str | None = None


def verify_api_key(request: Request):
    """FastAPI dependency to verify Bearer token API key."""
    # Read from env var because uvicorn.run("module:app") re-imports the module,
    # which resets module-level globals. Env vars survive across imports.
    key = os.environ.get("XMLX_VLM_API_KEY")
    if not key:
        return
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
    elif auth.startswith("bearer "):
        token = auth[7:]
    else:
        token = auth
    if token != key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def load_model_resources(model_path: str, adapter_path: Optional[str]):
    """
    Loads model, processor, and config based on paths.
    Handles potential loading errors.
    """
    try:
        print(f"Loading model from: {model_path}")
        if adapter_path:
            print(f"Loading adapter from: {adapter_path}")
        # Use the load function from utils.py which handles path resolution and loading
        trust_remote_code = (
            os.environ.get("MLX_TRUST_REMOTE_CODE", "false").lower() == "true"
        )
        model, processor = load(
            model_path, adapter_path, trust_remote_code=trust_remote_code
        )
        config = model.config
        # Apply runtime patches for model-specific fixes (MTP, BatchKVCache, etc.)
        apply_all_patches()
        print("Model and processor loaded successfully.")
        moe_top_k = os.environ.get("MLX_MOE_TOP_K")
        if moe_top_k:
            try:
                apply_moe_top_k_override(model, int(moe_top_k))
            except Exception as e:
                logger.warning("MoE top-k override failed: %s", e)
        return model, processor, config
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        traceback.print_exc()  # Print detailed traceback for debugging
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


_INHERIT_ADAPTER = object()


def get_cached_model(model_path: str, adapter_path=_INHERIT_ADAPTER):
    """
    Factory function to get or load the appropriate model resources from cache or by loading.
    Also creates/updates the ResponseGenerator for continuous batching.
    """
    global model_cache, response_generator, apc_manager

    if adapter_path is _INHERIT_ADAPTER:
        cached = model_cache.get("cache_key")
        adapter_path = cached[1] if cached and cached[0] == model_path else None

    cache_key = (model_path, adapter_path)

    # Return from cache if already loaded and matches the requested paths
    if model_cache.get("cache_key") == cache_key:
        print(f"Using cached model: {model_path}, Adapter: {adapter_path}")
        return model_cache["model"], model_cache["processor"], model_cache["config"]

    # If cache exists but doesn't match, clear it
    if model_cache:
        print("New model request, clearing existing cache...")
        unload_model_sync()  # Use a synchronous version for internal call

    vision_cache_size = int(os.environ.get("XMLX_VLM_VISION_CACHE_SIZE", "20"))
    vision_cache = VisionFeatureCache(max_size=vision_cache_size)

    # APC: build a shared block pool if opted in via env var.
    apc_manager = _apc.from_env(model_namespace=model_path)

    # KV cache quantization (uniform or TurboQuant)
    kv_bits = get_quantized_kv_bits(model_path)
    kv_group_size = get_kv_group_size()
    quantized_kv_start = get_quantized_kv_start()
    kv_quant_scheme = get_kv_quant_scheme()

    response_generator = ResponseGenerator(
        model_path=model_path,
        adapter_path=adapter_path,
        vision_cache=vision_cache,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        kv_quant_scheme=kv_quant_scheme,
        quantized_kv_start=quantized_kv_start,
        top_logprobs_k=get_top_logprobs_k(),
        apc_manager=apc_manager,
    )
    try:
        model, processor, config = response_generator.wait_until_ready()
    except Exception:
        response_generator.stop_and_join()
        response_generator = None
        vision_cache.clear()
        raise

    model_cache = {
        "cache_key": cache_key,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "model": model,
        "processor": processor,
        "config": config,
        "vision_cache": vision_cache,
    }

    return model, processor, config


# Synchronous unload function for internal use
def unload_model_sync():
    global model_cache, response_generator, apc_manager
    if not model_cache:
        return False

    print(
        f"Unloading model: {model_cache.get('model_path')}, Adapter: {model_cache.get('adapter_path')}"
    )

    # Stop the ResponseGenerator if running
    if response_generator is not None:
        print("Stopping ResponseGenerator...")
        response_generator.stop_and_join()
        response_generator = None

    # Drop APC blocks for the previous model
    if apc_manager is not None:
        apc_manager.clear()
        apc_manager = None

    # Clear vision cache before dropping references
    if "vision_cache" in model_cache:
        model_cache["vision_cache"].clear()
    model_cache = {}
    # Force garbage collection
    gc.collect()
    mx.clear_cache()
    print("Model unloaded and cache cleared.")
    return True


# OpenAI API Models

# Models for /responses endpoint



# OpenAI compatile endpoints


@app.post("/responses")
@app.post("/v1/responses", include_in_schema=False)
async def responses_endpoint(request: Request, _=Depends(verify_api_key)):
    """
    OpenAI-compatible endpoint for generating text based on a prompt and optional images.

    using client.responses.create method.

    example:

    from openai import OpenAI

    API_URL = "http://0.0.0.0:8000"
    API_KEY = 'any'

    def run_openai(prompt, img_url,system, stream=False, max_output_tokens=512, model="mlx-community/Qwen2.5-VL-3B-Instruct-8bit"):
        ''' Calls the OpenAI API
        '''

        client = OpenAI(base_url=f"{API_URL}", api_key=API_KEY)

        try :
            response = client.responses.create(
                model=model,
                input=[
                    {"role":"system",
                    "content": f"{system}"
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": f"{img_url}"},
                        ],
                    }
                ],
                max_output_tokens=max_output_tokens,
                stream=stream
            )
            if not stream:
                print(response.output[0].content[0].text)
                print(response.usage)
            else:
                for event in response:
                    # Process different event types if needed
                    if hasattr(event, 'delta') and event.delta:
                        print(event.delta, end="", flush=True)
                    elif event.type == 'response.completed':
                        print("\n--- Usage ---")
                        print(event.response.usage)

        except Exception as e:
            # building a response object to match the one returned when request is successful so that it can be processed in the same way
            return {"model - error":str(e),"content":{}, "model":model}

    """

    request_start = time.perf_counter()
    body = await request.json()
    openai_request = OpenAIRequest(**body)

    try:
        # Get model, processor, config - loading if necessary
        model, processor, config = get_cached_model(openai_request.model)

        # Initialize reasoning parser for thinking models
        reasoning_parser_name = _infer_reasoning_parser(model, config)
        reasoning_parser = None
        if reasoning_parser_name:
            try:
                parser_cls = _get_reasoning_parser_cls(reasoning_parser_name)
                reasoning_parser = parser_cls(
                    getattr(processor, "tokenizer", processor)
                )
                logger.debug("Using reasoning parser: %s", reasoning_parser_name)
            except Exception:
                logger.debug("Failed to load reasoning parser: %s", reasoning_parser_name)

        kwargs = {}

        chat_messages = []
        images = []
        audio = []
        instructions = None
        if openai_request.input:
            if isinstance(openai_request.input, str):
                # If input is a string, treat it as a single text message
                chat_messages.append({"role": "user", "content": openai_request.input})
            elif isinstance(openai_request.input, list):
                # If input is a list, treat it as a series of chat messages
                for message in openai_request.input:
                    if isinstance(message, ChatMessage):
                        if isinstance(message.content, str):
                            chat_messages.append(
                                {"role": message.role, "content": message.content}
                            )
                            if message.role == "system":
                                instructions = message.content
                        elif isinstance(message.content, list):
                            # Handle list of content items
                            for item in message.content:
                                if isinstance(item, dict):
                                    if item["type"] == "input_text":
                                        chat_messages.append(
                                            {
                                                "role": message.role,
                                                "content": item["text"],
                                            }
                                        )
                                        if message.role == "system":
                                            instructions = item["text"]
                                    # examples for multiple images (https://platform.openai.com/docs/guides/images?api-mode=responses)
                                    elif item["type"] == "input_image":
                                        images.append(item["image_url"])
                                    elif item["type"] == "input_audio":
                                        audio.append(item["input_audio"]["data"])
                                    else:
                                        print(
                                            f"invalid input item type: {item['type']}"
                                        )
                                        raise HTTPException(
                                            status_code=400,
                                            detail="Invalid input item type.",
                                        )
                                else:
                                    print(
                                        f"Invalid message content item format: {item}"
                                    )
                                    raise HTTPException(
                                        status_code=400,
                                        detail="Missing type in input item.",
                                    )
                        else:
                            print("Invalid message content format.")
                            raise HTTPException(
                                status_code=400, detail="Invalid input format."
                            )
                    else:
                        print("not a ChatMessage")
                        raise HTTPException(
                            status_code=400, detail="Invalid input format."
                        )
            else:
                print("neither string not list")
                raise HTTPException(status_code=400, detail="Invalid input format.")

        else:
            print("no input")
            raise HTTPException(status_code=400, detail="Missing input.")

        try:
            gen_args = _build_gen_args(
                openai_request, processor, tenant_id=_read_tenant_id(request)
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        formatted_prompt = apply_chat_template(
            processor,
            config,
            chat_messages,
            num_images=len(images),
            **gen_args.to_template_kwargs(),
        )

        logger.debug(
            "responses request: model=%s images=%d max_tokens=%s temp=%s stream=%s",
            openai_request.model,
            len(images),
            gen_args.max_tokens,
            gen_args.temperature,
            openai_request.stream,
        )

        generated_at = datetime.now().timestamp()
        response_id = f"resp_{uuid.uuid4().hex}"
        message_id = f"msg_{uuid.uuid4().hex}"

        if openai_request.stream:
            # Streaming response
            async def stream_generator():
                token_iterator = None
                token_iter = None  # For ResponseGenerator cleanup
                try:
                    # Create base response object (to match the openai pipeline)
                    base_response = OpenAIResponse(
                        id=response_id,
                        object="response",
                        created_at=int(generated_at),
                        status="in_progress",
                        instructions=instructions,
                        max_output_tokens=openai_request.max_output_tokens,
                        model=openai_request.model,
                        output=[],
                        output_text="",
                        temperature=openai_request.temperature,
                        top_p=openai_request.top_p,
                        usage={
                            "input_tokens": 0,  # get prompt tokens
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                    )

                    # Send response.created event  (to match the openai pipeline)
                    yield f"event: response.created\ndata: {ResponseCreatedEvent(type='response.created', response=base_response).model_dump_json()}\n\n"

                    # Send response.in_progress event  (to match the openai pipeline)
                    yield f"event: response.in_progress\ndata: {ResponseInProgressEvent(type='response.in_progress', response=base_response).model_dump_json()}\n\n"

                    # Send response.output_item.added event  (to match the openai pipeline)
                    message_item = MessageItem(
                        id=message_id,
                        type="message",
                        status="in_progress",
                        role="assistant",
                        content=[],
                    )
                    yield f"event: response.output_item.added\ndata: {ResponseOutputItemAddedEvent(type='response.output_item.added', output_index=0, item=message_item).model_dump_json()}\n\n"

                    # Send response.content_part.added event
                    content_part = ContentPartOutputText(
                        type="output_text", text="", annotations=[]
                    )
                    yield f"event: response.content_part.added\ndata: {ResponseContentPartAddedEvent(type='response.content_part.added', item_id=message_id, output_index=0, content_index=0, part=content_part).model_dump_json()}\n\n"

                    # Stream text deltas using ResponseGenerator (continuous batching)
                    full_text = ""
                    usage_stats = {"input_tokens": 0, "output_tokens": 0}

                    if response_generator is not None:
                        # generate() blocks on _cpu_preprocess + queue.get;
                        # offload so concurrent handlers preprocess in parallel.
                        ctx, token_iter = await asyncio.to_thread(
                            response_generator.generate,
                            formatted_prompt,
                            images if images else None,
                            audio if audio else None,
                            gen_args,
                        )

                        output_tokens = 0

                        def _next_token_resp_stream():
                            try:
                                return next(token_iter)
                            except StopIteration:
                                return None

                        while True:
                            token = await asyncio.to_thread(_next_token_resp_stream)
                            if token is None:
                                break
                            output_tokens += 1
                            delta = token.text
                            full_text += delta
                            usage_stats = {
                                "input_tokens": ctx.prompt_tokens,
                                "output_tokens": output_tokens,
                            }

                            yield f"event: response.output_text.delta\ndata: {ResponseOutputTextDeltaEvent(type='response.output_text.delta', item_id=message_id, output_index=0, content_index=0, delta=delta).model_dump_json()}\n\n"
                            await asyncio.sleep(0.01)

                            if token.finish_reason:
                                break
                    else:
                        # Fallback to stream_generate
                        token_iterator = stream_generate(
                            model=model,
                            processor=processor,
                            prompt=formatted_prompt,
                            image=images,
                            audio=audio,
                            temperature=openai_request.temperature,
                            max_tokens=gen_args.max_tokens,
                            top_p=openai_request.top_p,
                            vision_cache=model_cache.get("vision_cache"),
                            logits_processors=gen_args.logits_processors,
                            apc_manager=apc_manager,
                            apc_tenant=gen_args.tenant_id,
                            **kwargs,
                        )

                        for chunk in token_iterator:
                            if chunk is None or not hasattr(chunk, "text"):
                                continue

                            delta = chunk.text
                            full_text += delta
                            usage_stats = {
                                "input_tokens": chunk.prompt_tokens,
                                "output_tokens": chunk.generation_tokens,
                            }

                            yield f"event: response.output_text.delta\ndata: {ResponseOutputTextDeltaEvent(type='response.output_text.delta', item_id=message_id, output_index=0, content_index=0, delta=delta).model_dump_json()}\n\n"
                            await asyncio.sleep(0.01)

                    # Split thinking from content for final events
                    _, clean_text = _split_thinking(full_text, reasoning_parser)

                    # Send response.output_text.done event (to match the openai pipeline)
                    yield f"event: response.output_text.done\ndata: {ResponseOutputTextDoneEvent(type='response.output_text.done', item_id=message_id, output_index=0, content_index=0, text=clean_text).model_dump_json()}\n\n"

                    # Send response.content_part.done event (to match the openai pipeline)
                    final_content_part = ContentPartOutputText(
                        type="output_text", text=clean_text, annotations=[]
                    )
                    yield f"event: response.content_part.done\ndata: {ResponseContentPartDoneEvent(type='response.content_part.done', item_id=message_id, output_index=0, content_index=0, part=final_content_part).model_dump_json()}\n\n"

                    # Send response.output_item.done event (to match the openai pipeline)
                    final_message_item = MessageItem(
                        id=message_id,
                        type="message",
                        status="completed",
                        role="assistant",
                        content=[final_content_part],
                    )
                    yield f"event: response.output_item.done\ndata: {ResponseOutputItemDoneEvent(type='response.output_item.done', output_index=0, item=final_message_item).model_dump_json()}\n\n"

                    # Send response.completed event (to match the openai pipeline)
                    completed_response = base_response.model_copy(
                        update={
                            "status": "completed",
                            "output": [final_message_item],
                            "usage": {
                                "input_tokens": usage_stats["input_tokens"],
                                "output_tokens": usage_stats["output_tokens"],
                                "total_tokens": usage_stats["input_tokens"]
                                + usage_stats["output_tokens"],
                            },
                        }
                    )
                    yield f"event: response.completed\ndata: {ResponseCompletedEvent(type='response.completed', response=completed_response).model_dump_json()}\n\n"

                except Exception as e:
                    print(f"Error during stream generation: {e}")
                    traceback.print_exc()
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"

                finally:
                    if token_iter is not None:
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                    _maybe_clear_cache()
                    print("Stream finished, cache cleared.")

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            # Non-streaming response
            try:
                full_text = ""
                prompt_tokens = 0
                output_tokens = 0

                if response_generator is not None:

                    def _blocking_resp():
                        ctx_, ti = response_generator.generate(
                            prompt=formatted_prompt,
                            images=images if images else None,
                            audio=audio if audio else None,
                            args=gen_args,
                        )
                        text = ""
                        ot = 0
                        for tok in ti:
                            text += tok.text
                            ot += 1
                            if tok.finish_reason:
                                break
                        try:
                            ti.close()
                        except Exception:
                            pass
                        return ctx_.prompt_tokens, text, ot

                    prompt_tokens, full_text, output_tokens = await asyncio.to_thread(
                        _blocking_resp
                    )
                else:
                    result = generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        audio=audio,
                        verbose=logger.isEnabledFor(logging.DEBUG),
                        vision_cache=model_cache.get("vision_cache"),
                        apc_manager=apc_manager,
                        **gen_args.to_generate_kwargs(),
                        **kwargs,
                    )
                    full_text = result.text
                    prompt_tokens = result.prompt_tokens
                    output_tokens = result.generation_tokens

                _maybe_clear_cache()

                reasoning, content = _split_thinking(full_text, reasoning_parser)

                response = OpenAIResponse(
                    id=response_id,
                    object="response",
                    created_at=int(generated_at),
                    status="completed",
                    instructions=instructions,
                    max_output_tokens=openai_request.max_output_tokens,
                    model=openai_request.model,
                    output=[
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": content,
                                }
                            ],
                            "reasoning": reasoning,
                        }
                    ],
                    output_text=content,
                    temperature=openai_request.temperature,
                    top_p=openai_request.top_p,
                    usage={
                        "input_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": prompt_tokens + output_tokens,
                    },
                )

                elapsed = time.perf_counter() - request_start
                logger.debug(
                    "responses done: prompt_tokens=%d output_tokens=%d "
                    "total_time=%.2fs",
                    prompt_tokens,
                    output_tokens,
                    elapsed,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    resp_text = content or ""
                    logger.debug(
                        "  response: %s",
                        resp_text[:200] + ("..." if len(resp_text) > 200 else ""),
                    )

                return response

            except Exception as e:
                print(f"Error during generation: {e}")
                traceback.print_exc()
                _maybe_clear_cache()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(f"Unexpected error in /responses endpoint: {e}")
        traceback.print_exc()
        _maybe_clear_cache()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.post("/chat/completions", response_model=None)
@app.post("/v1/chat/completions", response_model=None, include_in_schema=False)
async def chat_completions_endpoint(request: ChatRequest, http_request: Request, _=Depends(verify_api_key)):
    """
    Generate text based on a prompt and optional images.
    Prompt must be a list of chat messages, including system, user, and assistant messages.
    System message will be ignored if not already in the prompt.
    Can operate in streaming or non-streaming mode.
    """

    request_start = time.perf_counter()
    try:
        adapter_path = (
            request.adapter_path
            if "adapter_path" in request.model_fields_set
            else _INHERIT_ADAPTER
        )
        model, processor, config = get_cached_model(request.model, adapter_path)

        # Initialize reasoning parser for thinking models
        reasoning_parser_name = _infer_reasoning_parser(model, config)
        reasoning_parser = None
        if reasoning_parser_name:
            try:
                parser_cls = _get_reasoning_parser_cls(reasoning_parser_name)
                reasoning_parser = parser_cls(
                    getattr(processor, "tokenizer", processor)
                )
                logger.debug("Using reasoning parser: %s", reasoning_parser_name)
            except Exception:
                logger.debug("Failed to load reasoning parser: %s", reasoning_parser_name)

        kwargs = {}

        if request.resize_shape is not None:
            if len(request.resize_shape) not in [1, 2]:
                raise HTTPException(
                    status_code=400,
                    detail="resize_shape must contain exactly two integers (height, width)",
                )
            kwargs["resize_shape"] = (
                (request.resize_shape[0],) * 2
                if len(request.resize_shape) == 1
                else tuple(request.resize_shape)
            )

        images = []
        audio = []
        processed_messages = []
        for message in request.messages:
            msg = {"role": message.role}

            if isinstance(message.content, str):
                msg["content"] = message.content
            elif isinstance(message.content, list):
                text_content = ""
                for item in message.content:
                    if isinstance(item, dict):
                        if message.role == "user":
                            if item["type"] == "input_image":
                                images.append(item["image_url"])
                            elif item["type"] == "image_url":
                                images.append(item["image_url"]["url"])
                            elif item["type"] == "input_audio":
                                audio.append(item["input_audio"]["data"])
                        if item["type"] in ("text", "input_text"):
                            text_content = item.get("text", "")
                msg["content"] = text_content
            else:
                msg["content"] = message.content

            # Preserve tool-calling metadata.
            # Ensure arguments are dicts (not JSON strings) for Jinja templates
            # that iterate them with |items (e.g. Qwen3.5).
            if message.tool_calls is not None:
                normalized_calls = []
                for tc in message.tool_calls:
                    tc = dict(tc) if isinstance(tc, dict) else tc
                    if isinstance(tc, dict) and "function" in tc:
                        fn = dict(tc["function"])
                        args = fn.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                fn["arguments"] = json.loads(args)
                            except (json.JSONDecodeError, TypeError):
                                fn["arguments"] = {}
                        tc["function"] = fn
                    normalized_calls.append(tc)
                msg["tool_calls"] = normalized_calls
            if message.tool_call_id is not None:
                msg["tool_call_id"] = message.tool_call_id
            if message.name is not None:
                msg["name"] = message.name

            # Guard against empty tool results that confuse local models into
            # hallucination loops. Provide an explicit sentinel so the model
            # knows the tool executed but produced no usable output.
            if msg.get("role") == "tool":
                content = msg.get("content")
                if content is None or (isinstance(content, str) and not content.strip()):
                    tool_name = msg.get("name", "unknown")
                    msg["content"] = (
                        f"[Tool '{tool_name}' returned empty or missing output. "
                        f"If this was unexpected, verify the arguments and try again.]"
                    )

            processed_messages.append(msg)

        # Detect tool parser from chat template
        tools = getattr(request, "tools", None)
        mcp_enabled = getattr(request, "mcp", False)
        mcp_manager = get_manager()
        if mcp_enabled and not tools:
            tools = mcp_manager.schemas

        tool_parser_type = _infer_tool_parser_from_processor(processor)
        tool_module = load_tool_module(tool_parser_type) if tool_parser_type else None

        try:
            gen_args = _build_gen_args(
                request, processor, tenant_id=_read_tenant_id(http_request)
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

        formatted_prompt = apply_chat_template(
            processor,
            config,
            processed_messages,
            num_images=len(images),
            num_audios=len(audio),
            tools=tools,
            **gen_args.to_template_kwargs(),
        )

        logger.debug(
            "chat/completions request: model=%s images=%d audio=%d "
            "max_tokens=%s temp=%s stream=%s",
            request.model,
            len(images),
            len(audio),
            gen_args.max_tokens,
            gen_args.temperature,
            request.stream,
        )

        if request.stream:
            # Streaming response using ResponseGenerator for continuous batching
            async def stream_generator():
                global response_generator
                token_iterator = None
                token_iter = None  # For ResponseGenerator cleanup
                try:
                    # Use ResponseGenerator if available, otherwise fall back to stream_generate
                    if response_generator is not None:
                        # generate() does blocking Queue.get — run off event loop
                        ctx, token_iter = await asyncio.to_thread(
                            response_generator.generate,
                            formatted_prompt,
                            images if images else None,
                            audio if audio else None,
                            gen_args,
                        )

                        output_tokens = 0
                        request_id = f"chatcmpl-{uuid.uuid4()}"
                        full_output = ""  # raw output for tool call parsing
                        # Track tool-call state to suppress markup from content
                        in_tool_call = False
                        tc_start = tool_module.tool_call_start if tool_module else None
                        tc_end = tool_module.tool_call_end if tool_module else None
                        # Initialize reasoning parser for streaming extraction
                        if reasoning_parser is not None:
                            reasoning_parser.reset_state()

                        def _next_token():
                            try:
                                return next(token_iter)
                            except StopIteration:
                                return None

                        while True:
                            token = await asyncio.to_thread(_next_token)
                            if token is None:
                                break
                            output_tokens += 1
                            previous_text = full_output
                            full_output += token.text

                            # Detect thinking boundaries
                            delta_reasoning = None
                            delta_content = None

                            if reasoning_parser is not None:
                                msg = reasoning_parser.extract_reasoning_streaming(
                                    previous_text, full_output, token.text
                                )
                                if msg is not None:
                                    delta_reasoning = msg.reasoning
                                    delta_content = msg.content
                            else:
                                # Fallback heuristic for models without reasoning parser
                                delta_content = token.text

                            # Suppress tool-call markup from content
                            in_tool_call, delta_content = suppress_tool_call_content(
                                full_output, in_tool_call, tc_start, delta_content
                            )

                            chunk_logprobs = None
                            if request.logprobs and token.finish_reason != "stop":
                                req_top_k = int(request.top_logprobs or 0)
                                chunk_logprobs = ChatLogprobs(
                                    content=[
                                        _make_logprob_content(
                                            response_generator.tokenizer,
                                            token.token,
                                            token.logprobs,
                                            top_logprobs=token.top_logprobs,
                                            top_k=req_top_k,
                                        )
                                    ]
                                )

                            # Skip empty deltas (e.g. suppressed tool-call tokens)
                            has_payload = (
                                delta_content is not None
                                or delta_reasoning is not None
                                or token.finish_reason is not None
                                or chunk_logprobs is not None
                            )
                            if has_payload:
                                choices = [
                                    ChatStreamChoice(
                                        finish_reason=token.finish_reason,
                                        delta=ChatMessage(
                                            role="assistant",
                                            content=delta_content,
                                            reasoning=delta_reasoning,
                                        ),
                                        logprobs=chunk_logprobs,
                                    )
                                ]
                                chunk_data = ChatStreamChunk(
                                    id=request_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    usage={
                                        "prompt_tokens": ctx.prompt_tokens,
                                        "completion_tokens": output_tokens,
                                        "total_tokens": ctx.prompt_tokens
                                        + output_tokens,
                                    },
                                    choices=choices,
                                )

                                yield f"data: {chunk_data.model_dump_json()}\n\n"

                            if token.finish_reason:
                                break

                        # Flush any buffered reasoning at stream end
                        if reasoning_parser is not None:
                            final_msg = reasoning_parser.finalize_stream()
                            if final_msg is not None and (
                                final_msg.reasoning is not None
                                or final_msg.content is not None
                            ):
                                choices = [
                                    ChatStreamChoice(
                                        delta=ChatMessage(
                                            role="assistant",
                                            content=final_msg.content,
                                            reasoning=final_msg.reasoning,
                                        ),
                                    )
                                ]
                                chunk_data = ChatStreamChunk(
                                    id=request_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    choices=choices,
                                )
                                yield f"data: {chunk_data.model_dump_json()}\n\n"

                        # Parse tool calls from full output and emit final chunk
                        if tool_module is not None:
                            tc = process_tool_calls(full_output, tool_module, tools)
                            if tc["calls"]:
                                choices = [
                                    ChatStreamChoice(
                                        finish_reason="tool_calls",
                                        delta=ChatMessage(
                                            role="assistant",
                                            tool_calls=tc["calls"],
                                        ),
                                    )
                                ]
                                chunk_data = ChatStreamChunk(
                                    id=request_id,
                                    created=int(time.time()),
                                    model=request.model,
                                    choices=choices,
                                )
                                yield f"data: {chunk_data.model_dump_json()}\n\n"
                    else:
                        # Fallback to stream_generate
                        token_iterator = stream_generate(
                            model=model,
                            processor=processor,
                            prompt=formatted_prompt,
                            image=images,
                            audio=audio,
                            temperature=request.temperature,
                            max_tokens=gen_args.max_tokens,
                            top_p=request.top_p,
                            vision_cache=model_cache.get("vision_cache"),
                            logits_processors=gen_args.logits_processors,
                            apc_manager=apc_manager,
                            apc_tenant=gen_args.tenant_id,
                            **kwargs,
                        )

                        request_id = f"chatcmpl-{uuid.uuid4()}"
                        output_text = ""
                        for chunk in token_iterator:
                            if chunk is None or not hasattr(chunk, "text"):
                                continue

                            output_text += chunk.text

                            choices = [
                                ChatStreamChoice(
                                    delta=ChatMessage(
                                        role="assistant", content=chunk.text
                                    )
                                )
                            ]
                            chunk_data = ChatStreamChunk(
                                id=request_id,
                                created=int(time.time()),
                                model=request.model,
                                usage={
                                    "prompt_tokens": chunk.prompt_tokens,
                                    "completion_tokens": chunk.generation_tokens,
                                    "total_tokens": chunk.prompt_tokens
                                    + chunk.generation_tokens,
                                },
                                choices=choices,
                            )

                            yield f"data: {chunk_data.model_dump_json()}\n\n"
                            await asyncio.sleep(0.01)

                    # Signal stream end
                    yield "data: [DONE]\n\n"

                    elapsed = time.perf_counter() - request_start
                    logger.debug(
                        "chat/completions stream done: tokens=%d " "total_time=%.2fs",
                        output_tokens,
                        elapsed,
                    )

                except Exception as e:
                    print(f"Error during stream generation: {e}")
                    traceback.print_exc()
                    error_data = json.dumps({"error": str(e)})
                    yield f"data: {error_data}\n\n"

                finally:
                    # Close the token iterator to trigger cleanup (important for ResponseGenerator)
                    if token_iter is not None:
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                    _maybe_clear_cache()
                    print("Stream finished, cache cleared.")

            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            # Non-streaming response
            try:
                full_text = ""
                prompt_tokens = 0
                output_tokens = 0
                peak_memory = 0.0

                collected_logprobs: List[
                    Tuple[int, float, Optional[List[Tuple[int, float]]]]
                ] = []

                if response_generator is not None:

                    def _blocking_generate():
                        text = ""
                        pt = gt = 0
                        pm = 0.0
                        ctx, token_iter = response_generator.generate(
                            prompt=formatted_prompt,
                            images=images if images else None,
                            audio=audio if audio else None,
                            args=gen_args,
                        )
                        pt = ctx.prompt_tokens
                        for token in token_iter:
                            text += token.text
                            gt += 1
                            pm = token.peak_memory
                            if request.logprobs and token.finish_reason != "stop":
                                collected_logprobs.append(
                                    (token.token, token.logprobs, token.top_logprobs)
                                )
                            if token.finish_reason:
                                break
                        try:
                            token_iter.close()
                        except Exception:
                            pass
                        return text, pt, gt, pm

                    full_text, prompt_tokens, output_tokens, peak_memory = (
                        await asyncio.to_thread(_blocking_generate)
                    )
                else:
                    gen_result = generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        audio=audio,
                        verbose=logger.isEnabledFor(logging.DEBUG),
                        vision_cache=model_cache.get("vision_cache"),
                        apc_manager=apc_manager,
                        **gen_args.to_generate_kwargs(),
                        **kwargs,
                    )
                    full_text = gen_result.text
                    prompt_tokens = gen_result.prompt_tokens
                    output_tokens = gen_result.generation_tokens
                    peak_memory = gen_result.peak_memory

                _maybe_clear_cache()

                reasoning, content = _split_thinking(full_text)

                # Count raw generated tokens minus thinking tag tokens
                completion_tokens = output_tokens - _count_thinking_tag_tokens(
                    full_text
                )

                usage_stats = UsageStats(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    peak_memory=peak_memory,
                )

                # Parse tool calls from generated output
                parsed_tool_calls = None
                if tool_module is not None:
                    tc = process_tool_calls(
                        model_output=full_text,
                        tool_module=tool_module,
                        tools=tools,
                    )
                    if tc["calls"]:
                        parsed_tool_calls = tc["calls"]
                        # Clean thinking tags and control tokens from remaining text
                        _, clean_remaining = _split_thinking(tc["remaining_text"] or "", reasoning_parser)
                        if clean_remaining:
                            # Strip model control tokens
                            clean_remaining = re.sub(
                                r"<\|[^>]+\|>|<[^>]+>", "", clean_remaining
                            ).strip()
                        content = clean_remaining or None

                response_logprobs = None
                if request.logprobs and collected_logprobs:
                    tokenizer = (
                        processor.tokenizer
                        if hasattr(processor, "tokenizer")
                        else processor
                    )
                    req_top_k = int(request.top_logprobs or 0)
                    response_logprobs = ChatLogprobs(
                        content=[
                            _make_logprob_content(
                                tokenizer,
                                tid,
                                lp,
                                top_logprobs=top_lps,
                                top_k=req_top_k,
                            )
                            for tid, lp, top_lps in collected_logprobs
                        ]
                    )

                choices = [
                    ChatChoice(
                        finish_reason="tool_calls" if parsed_tool_calls else "stop",
                        message=ChatMessage(
                            role="assistant",
                            content=content if content else None,
                            reasoning=reasoning,
                            tool_calls=parsed_tool_calls,
                        ),
                        logprobs=response_logprobs,
                    )
                ]
                result = ChatResponse(
                    id=f"chatcmpl-{uuid.uuid4()}",
                    created=int(time.time()),
                    model=request.model,
                    usage=usage_stats,
                    choices=choices,
                )

                elapsed = time.perf_counter() - request_start
                logger.debug(
                    "chat/completions done: prompt_tokens=%d completion_tokens=%d "
                    "total_time=%.2fs peak_memory=%.2fGB",
                    prompt_tokens,
                    completion_tokens,
                    elapsed,
                    peak_memory,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    resp_text = content or ""
                    logger.debug(
                        "  response: %s",
                        resp_text[:200] + ("..." if len(resp_text) > 200 else ""),
                    )

                return result

            except Exception as e:
                print(f"Error during generation: {e}")
                traceback.print_exc()
                _maybe_clear_cache()
                raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions (like model loading failure)
        raise http_exc
    except Exception as e:
        # Catch unexpected errors
        print(f"Unexpected error in /generate endpoint: {e}")
        traceback.print_exc()
        _maybe_clear_cache()
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}"
        )


@app.post("/v1/messages", response_model=None)
async def anthropic_messages_endpoint(request: AnthropicMessageRequest, http_request: Request, _=Depends(verify_api_key)):
    """
    Anthropic-compatible /v1/messages endpoint.
    Converts the request to OpenAI ChatRequest format, reuses the existing
    generation logic, and converts the response back to Anthropic format.
    """
    # Convert Anthropic messages to OpenAI ChatMessage format
    chat_messages: List[ChatMessage] = []
    if request.system:
        chat_messages.append(ChatMessage(role="system", content=request.system))
    for msg in request.messages:
        content = msg.content
        if isinstance(content, list):
            content = "".join(c.text for c in content if c.type == "text")
        chat_messages.append(ChatMessage(role=msg.role, content=content))

    chat_request = ChatRequest(
        model=request.model,
        messages=chat_messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=request.stream or False,
    )

    # Reuse existing chat completions logic for non-streaming
    if not chat_request.stream:
        result = await chat_completions_endpoint(chat_request, http_request)
        chat_response = result
        content_text = chat_response.choices[0].message.content or ""
        return AnthropicMessageResponse(
            id=f"msg_{uuid.uuid4().hex[:24]}",
            type="message",
            role="assistant",
            model=request.model,
            content=[AnthropicMessageContent(type="text", text=content_text)],
            stop_reason="end_turn",
            usage=AnthropicUsage(
                input_tokens=chat_response.usage.prompt_tokens,
                output_tokens=chat_response.usage.completion_tokens,
            ),
        )

    # ── Streaming path: emit native Anthropic SSE ──────────────────────────
    model, processor, config = get_cached_model(request.model)
    gen_args = _build_gen_args(
        chat_request, processor, tenant_id=_read_tenant_id(http_request)
    )

    simple_messages = [{"role": m.role, "content": m.content} for m in chat_messages]
    formatted_prompt = apply_chat_template(
        processor,
        config,
        simple_messages,
        num_images=0,
        num_audios=0,
        **gen_args.to_template_kwargs(),
    )

    # Best-effort input token count for message_start usage
    try:
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        input_tokens = len(tokenizer.encode(formatted_prompt))
    except Exception:
        input_tokens = 0

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    model_name = request.model

    async def anthropic_stream_generator():
        output_tokens = 0
        full_text = ""
        token_iter = None

        try:
            # event: message_start
            yield f"event: message_start\ndata: {json.dumps({'type':'message_start','message':{'id':msg_id,'type':'message','role':'assistant','model':model_name,'content':[],'stop_reason':None,'stop_sequence':None,'usage':{'input_tokens':input_tokens,'output_tokens':0}}})}\n\n"

            # event: content_block_start
            yield f"event: content_block_start\ndata: {json.dumps({'type':'content_block_start','index':0,'content_block':{'type':'text','text':''}})}\n\n"

            # event: ping
            yield f"event: ping\ndata: {json.dumps({'type':'ping'})}\n\n"

            if response_generator is not None:
                ctx, token_iter = await asyncio.to_thread(
                    response_generator.generate,
                    formatted_prompt,
                    None,
                    None,
                    gen_args,
                )

                def _next_token():
                    try:
                        return next(token_iter)
                    except StopIteration:
                        return None

                while True:
                    token = await asyncio.to_thread(_next_token)
                    if token is None:
                        break
                    output_tokens += 1
                    full_text += token.text
                    yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':token.text}})}\n\n"
                    if token.finish_reason:
                        break
            else:
                token_iterator = stream_generate(
                    model=model,
                    processor=processor,
                    prompt=formatted_prompt,
                    image=None,
                    audio=None,
                    vision_cache=model_cache.get("vision_cache"),
                    apc_manager=apc_manager,
                    **gen_args.to_generate_kwargs(),
                )
                for chunk in token_iterator:
                    if chunk is None or not hasattr(chunk, "text"):
                        continue
                    output_tokens += 1
                    full_text += chunk.text
                    yield f"event: content_block_delta\ndata: {json.dumps({'type':'content_block_delta','index':0,'delta':{'type':'text_delta','text':chunk.text}})}\n\n"
                    if getattr(chunk, "finish_reason", None):
                        break

            # event: content_block_stop
            yield f"event: content_block_stop\ndata: {json.dumps({'type':'content_block_stop','index':0})}\n\n"

            # event: message_delta
            yield f"event: message_delta\ndata: {json.dumps({'type':'message_delta','delta':{'stop_reason':'end_turn','stop_sequence':None},'usage':{'output_tokens':output_tokens}})}\n\n"

            # event: message_stop
            yield f"event: message_stop\ndata: {json.dumps({'type':'message_stop'})}\n\n"

        except Exception as e:
            logger.exception("Error in Anthropic streaming")
            yield f"event: error\ndata: {json.dumps({'type':'error','error':{'type':'overloaded_error','message':str(e)}})}\n\n"

        finally:
            if token_iter is not None:
                try:
                    token_iter.close()
                except Exception:
                    pass
            _maybe_clear_cache()

    return StreamingResponse(
        anthropic_stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ─── MCP endpoints ──────────────────────────────────────────────────────────


@app.post("/v1/mcp/execute", response_model=None)
async def mcp_execute_endpoint(request: MCPExecuteRequest, _=Depends(verify_api_key)):
    """Execute MCP tool calls and return results."""
    mcp_manager = get_manager()
    results = await mcp_manager.execute(request.calls)
    return {"results": results}


@app.post("/v1/mcp/chat/completions", response_model=None)
async def mcp_chat_completions_endpoint(request: ChatRequest, http_request: Request, _=Depends(verify_api_key)):
    """
    Chat completions with automatic MCP tool execution.

    Works like /v1/chat/completions but automatically executes any MCP tool
    calls generated by the model and continues the conversation up to 3 rounds.
    Only supports non-streaming mode.
    """
    request.stream = False
    if not getattr(request, "mcp", False):
        # Force mcp on so that tools are injected
        object.__setattr__(request, "mcp", True)

    mcp_manager = get_manager()
    max_rounds = int(os.environ.get("MLX_MCP_MAX_ROUNDS", "3"))

    all_messages = list(request.messages)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_peak_memory = 0.0

    for round_idx in range(max_rounds):
        # Update messages in the request for this round
        request.messages = all_messages

        result = await chat_completions_endpoint(request, http_request)
        if isinstance(result, StreamingResponse):
            return result  # Should not happen because we force stream=False

        choice = result.choices[0]
        msg = choice.message

        total_prompt_tokens += result.usage.prompt_tokens
        total_completion_tokens += result.usage.completion_tokens
        total_peak_memory = max(total_peak_memory, result.usage.peak_memory)

        # Append assistant message
        assistant_msg = ChatMessage(
            role="assistant",
            content=msg.content,
            tool_calls=msg.tool_calls,
            reasoning=msg.reasoning,
        )
        all_messages.append(assistant_msg)

        if not msg.tool_calls:
            break

        # Execute MCP tool calls
        tool_results = await mcp_manager.execute(msg.tool_calls)
        for tr in tool_results:
            all_messages.append(
                ChatMessage(
                    role="tool",
                    tool_call_id=tr.get("tool_call_id", ""),
                    name=tr.get("name", ""),
                    content=tr.get("content", ""),
                )
            )
    else:
        # Max rounds reached — return what we have
        pass

    # Build final response with accumulated stats
    final_choice = result.choices[0]
    final_choice.message.reasoning = final_choice.message.reasoning
    return ChatResponse(
        id=result.id,
        created=result.created,
        model=result.model,
        usage=UsageStats(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            peak_memory=total_peak_memory,
        ),
        choices=[final_choice],
    )


@app.get("/v1/mcp/status", response_model=None)
async def mcp_status_endpoint(_=Depends(verify_api_key)):
    """Get status of all connected external MCP servers."""
    mcp_manager = get_manager()
    return {
        "builtin_tools": len(mcp_manager.schemas) - len(mcp_manager._external_schemas),
        "external_tools": len(mcp_manager._external_schemas),
        "servers": mcp_manager.get_status(),
    }


@app.get("/v1/mcp/tools", response_model=None)
async def mcp_tools_endpoint(_=Depends(verify_api_key)):
    """List all available MCP tools (builtin + external)."""
    mcp_manager = get_manager()
    return {"tools": mcp_manager.schemas}


# ─── Embeddings endpoint ────────────────────────────────────────────────────


@app.post("/v1/embeddings", response_model=None)
async def embeddings_endpoint(request: EmbeddingRequest, _=Depends(verify_api_key)):
    global _embedding_engine
    if _embedding_engine is None or _embedding_engine.model_name != request.model:
        _embedding_engine = EmbeddingEngine(request.model)
    texts = request.input if isinstance(request.input, list) else [request.input]
    vectors = _embedding_engine.embed(texts)
    prompt_tokens = _embedding_engine.count_tokens(texts)
    data = [
        EmbeddingData(embedding=vec, index=i)
        for i, vec in enumerate(vectors)
    ]
    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage=EmbeddingUsage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
    )


# ─── Rerank endpoint ────────────────────────────────────────────────────────


@app.post("/v1/rerank", response_model=None)
async def rerank_endpoint(request: RerankRequest, _=Depends(verify_api_key)):
    global _rerank_engine
    if _rerank_engine is None or _rerank_engine.model_name != request.model:
        _rerank_engine = RerankEngine(request.model)
    docs: List[str] = []
    for d in request.documents:
        if isinstance(d, str):
            docs.append(d)
        else:
            docs.append(d.text)
    scores, total_tokens = _rerank_engine.score_pairs(request.query, docs)
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    if request.top_n is not None:
        indexed = indexed[: request.top_n]
    results = [
        RerankResult(
            index=idx,
            relevance_score=score,
            document=RerankDocument(text=docs[idx]),
        )
        for idx, score in indexed
    ]
    return RerankResponse(
        results=results,
        model=request.model,
        usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )


# ─── Prometheus metrics endpoint ────────────────────────────────────────────

@app.get("/metrics", response_model=None)
async def metrics_endpoint():
    if not metrics.enabled:
        raise HTTPException(status_code=503, detail="Metrics not enabled")
    try:
        data, content_type = metrics.render_metrics()
        return Response(content=data, media_type=content_type)
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Metrics disabled")


@app.get("/models", response_model=ModelsResponse)
@app.get("/v1/models", response_model=ModelsResponse, include_in_schema=False)
def models_endpoint():
    """
    Return list of locally downloaded MLX models.
    """

    files = ["config.json", "model.safetensors.index.json", "tokenizer_config.json"]

    def probably_mlx_lm(repo):
        if repo.repo_type != "model":
            return False
        if "main" not in repo.refs:
            return False
        file_names = {f.file_path.name for f in repo.refs["main"].files}
        return all(f in file_names for f in files)

    # Scan the cache directory for downloaded mlx models
    hf_cache_info = scan_cache_dir()
    downloaded_models = [repo for repo in hf_cache_info.repos if probably_mlx_lm(repo)]

    # Create a list of available models
    models = [
        {"id": repo.repo_id, "object": "model", "created": int(repo.last_modified)}
        for repo in downloaded_models
    ]

    response = {"object": "list", "data": models}

    return response


# MLX_VLM API endpoints


@app.middleware("http")
async def add_server_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["Server"] = f"xmlx_vlm/{__version__}"
    return response


@app.get("/health")
async def health_check():
    """
    Check if the server is healthy and what model is loaded.
    """
    config = model_cache.get("config")
    text_config = getattr(config, "text_config", None)

    return {
        "status": "healthy",
        "loaded_model": model_cache.get("model_path", None),
        "loaded_adapter": model_cache.get("adapter_path", None),
        "loaded_context_size": getattr(text_config, "max_position_embeddings", None),
        "loaded_tool_parser": (
            _infer_tool_parser_from_processor(model_cache.get("processor"))
            if model_cache.get("processor")
            else None
        ),
        "continuous_batching_enabled": response_generator is not None,
        "apc_enabled": apc_manager is not None,
    }


@app.get("/v1/cache/stats")
@app.get("/cache/stats", include_in_schema=False)
async def apc_cache_stats():
    """Return Automatic Prefix Cache statistics (or ``enabled=false``)."""
    if apc_manager is None:
        return {"enabled": False}
    snap = apc_manager.stats_snapshot()
    snap["enabled"] = True
    return snap


@app.post("/v1/cache/reset")
@app.post("/cache/reset", include_in_schema=False)
async def apc_cache_reset():
    if apc_manager is None:
        return {"enabled": False}
    apc_manager.clear()
    return {"enabled": True, "status": "cleared"}


@app.post("/unload")
async def unload_model_endpoint():
    """
    Unload the currently loaded model from memory.
    """
    unloaded_info = {
        "model_name": model_cache.get("model_path", None),
        "adapter_name": model_cache.get("adapter_path", None),
    }

    if not unload_model_sync():  # Use the synchronous unload function
        return {"status": "no_model_loaded", "message": "No model is currently loaded"}

    return {
        "status": "success",
        "message": f"Model unloaded successfully",
        "unloaded": unloaded_info,
    }


def main():
    parser = argparse.ArgumentParser(description="MLX VLM Http Server.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the HTTP server (default:0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading models from Hugging Face Hub.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Pre-load a model at startup (e.g. mlx-community/Qwen2.5-VL-3B-Instruct-4bit).",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Adapter weights to load with the model.",
    )
    parser.add_argument(
        "--vision-cache-size",
        type=int,
        default=20,
        help="Max number of cached vision features (default: 20).",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help="Tokens per prefill step (default: %(default)s).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=get_server_max_tokens(),
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        default=DEFAULT_ENABLE_THINKING,
        help=(
            "Enable thinking mode by default for requests that do not set "
            "enable_thinking explicitly."
        ),
    )
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="Number of bits for KV cache quantization (e.g. 3.5 for TurboQuant).",
    )
    parser.add_argument(
        "--kv-quant-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=DEFAULT_KV_QUANT_SCHEME,
        help="KV cache quantization backend.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=DEFAULT_KV_GROUP_SIZE,
        help="Group size for uniform KV cache quantization.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV cache size in tokens.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index for quantized KV cache.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help=(
            "Speculative drafter path or HF id "
            "(e.g. z-lab/Qwen3.5-4B-DFlash, google/gemma-4-31B-it-assistant)."
        ),
    )
    parser.add_argument(
        "--draft-kind",
        type=str,
        default=None,
        choices=["dflash", "mtp"],
        help="Drafter family — 'dflash' or 'mtp' (Gemma 4). "
        "Default: auto-detected from the drafter's HF model_type.",
    )
    parser.add_argument(
        "--draft-block-size",
        type=int,
        default=None,
        help="Override the drafter's configured block size.",
    )
    parser.add_argument(
        "--top-logprobs-k",
        type=int,
        default=None,
        help=(
            "Server-side cap for per-token top_logprobs (0-20, default 0 = "
            "disabled). Maps to the TOP_LOGPROBS_K env var."
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable auto-reload for development.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication. If set, clients must provide it via Authorization: Bearer <key> header.",
    )
    parser.add_argument(
        "--apc-enabled",
        action="store_true",
        default=False,
        help="Enable Automatic Prefix Caching (APC) for KV cache reuse across requests.",
    )
    parser.add_argument(
        "--ssd-cache-dir",
        type=str,
        default=None,
        help="Directory for APC SSD-tiered cache (persists prefix cache to disk).",
    )
    parser.add_argument(
        "--ssd-cache-max-gb",
        type=float,
        default=0,
        help="Max SSD cache size in GB (0 = unbounded).",
    )
    parser.add_argument(
        "--moe-top-k",
        type=int,
        default=None,
        help="Override MoE top_k per token (e.g. 4 for Qwen3 MoE). Must be <= trained top_k.",
    )
    parser.add_argument(
        "--mcp-config",
        type=str,
        default=None,
        help="Path to MCP config file (JSON/YAML) for external MCP servers.",
    )
    parser.add_argument(
        "--enable-tool-logits-bias",
        action="store_true",
        default=DEFAULT_ENABLE_TOOL_LOGITS_BIAS,
        help=(
            "Bias logits toward structured tool-call tokens (e.g. <tool_call>, {, "
            '\"name\") to accelerate tool-call generation. Similar to Rapid-MLX\'s '
            "jump-forward decoding."
        ),
    )
    args = parser.parse_args()
    global _API_KEY
    _API_KEY = args.api_key
    if _API_KEY:
        os.environ["XMLX_VLM_API_KEY"] = _API_KEY
    if args.trust_remote_code:
        os.environ["MLX_TRUST_REMOTE_CODE"] = "true"
    if args.model:
        os.environ["XMLX_VLM_PRELOAD_MODEL"] = args.model
        if args.adapter_path:
            os.environ["XMLX_VLM_PRELOAD_ADAPTER"] = args.adapter_path
    os.environ["XMLX_VLM_VISION_CACHE_SIZE"] = str(args.vision_cache_size)
    if args.draft_model:
        os.environ["XMLX_VLM_DRAFT_MODEL"] = args.draft_model
        if args.draft_kind is not None:
            os.environ["XMLX_VLM_DRAFT_KIND"] = args.draft_kind
        if args.draft_block_size is not None:
            os.environ["XMLX_VLM_DRAFT_BLOCK_SIZE"] = str(args.draft_block_size)
    if args.prefill_step_size:
        os.environ["PREFILL_STEP_SIZE"] = str(args.prefill_step_size)
    os.environ["XMLX_VLM_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["XMLX_VLM_ENABLE_THINKING"] = "1" if args.enable_thinking else "0"
    if args.kv_bits is not None:
        os.environ["KV_BITS"] = str(args.kv_bits)
    os.environ["KV_GROUP_SIZE"] = str(args.kv_group_size)
    os.environ["KV_QUANT_SCHEME"] = args.kv_quant_scheme
    if args.max_kv_size is not None:
        os.environ["MAX_KV_SIZE"] = str(args.max_kv_size)
    os.environ["QUANTIZED_KV_START"] = str(args.quantized_kv_start)
    if args.top_logprobs_k is not None:
        os.environ["TOP_LOGPROBS_K"] = str(args.top_logprobs_k)
    if args.apc_enabled:
        os.environ["APC_ENABLED"] = "1"
    if args.ssd_cache_dir:
        os.environ["APC_DISK_PATH"] = args.ssd_cache_dir
    if args.ssd_cache_max_gb > 0:
        os.environ["APC_DISK_MAX_GB"] = str(args.ssd_cache_max_gb)
    if args.moe_top_k is not None:
        os.environ["MLX_MOE_TOP_K"] = str(args.moe_top_k)
    if args.mcp_config:
        os.environ["MLX_MCP_CONFIG"] = args.mcp_config
    os.environ["XMLX_VLM_ENABLE_TOOL_LOGITS_BIAS"] = "1" if args.enable_tool_logits_bias else "0"

    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.setLevel(log_level)

    uvicorn.run(
        "xmlx_vlm.server:app",
        host=args.host,
        port=args.port,
        workers=1,
        reload=args.reload,
        server_header=False,
    )


if __name__ == "__main__":
    main()
