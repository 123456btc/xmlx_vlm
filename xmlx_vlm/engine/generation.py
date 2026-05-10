"""
ResponseGenerator — continuous-batching GPU thread, plus streaming dataclasses
and speculative-decoding helpers.

Extracted from server.py Phase 2 refactor.

Design note: ``ResponseGenerator`` accepts a ``model_loader`` callable so it
has no direct dependency on server.py or FastAPI. The caller (server.py)
passes ``load_model_resources`` at construction time, keeping this module
import-clean and independently testable.
"""
import gc
import logging
import os
import traceback
from dataclasses import dataclass
from queue import Empty as QueueEmpty
from queue import Full as QueueFull
from queue import Queue
from threading import Event, Lock, Thread
from typing import Callable, Iterator, List, Optional, Tuple

import mlx.core as mx

from .. import apc as _apc
from ..config import (
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_QUANTIZED_KV_START,
    get_first_token_timeout,
    get_server_max_queue_depth,
    get_token_queue_timeout,
)
from ..generate import (
    BatchGenerator,
    _apply_rep_penalty,
    _dflash_rounds_batch,
    _make_cache,
    _merge_prefill_prompt_kwargs,
    _mtp_rounds_batch,
)
from ..sample_utils import top_p_sampling
from ..tokenizer_utils import make_streaming_detokenizer
from ..utils import prepare_inputs
from .arguments import GenerationArguments

logger = logging.getLogger("xmlx_vlm.engine.generation")


# ---------------------------------------------------------------------------
# Speculative-decoding helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Streaming dataclasses
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ResponseGenerator
# ---------------------------------------------------------------------------

class ResponseGenerator:
    """
    Continuous batching for concurrent requests via a single GPU thread.

    A dedicated thread owns all GPU work (BatchGenerator). FastAPI async
    handlers submit requests to a queue and read tokens back from
    per-request queues. Multiple requests are batched together for
    higher throughput — same pattern as mlx-lm's server.

    Parameters
    ----------
    model_loader:
        Callable ``(model_path, adapter_path) -> (model, processor, config)``
        that loads model weights.  Accepting it as a parameter keeps this
        class free of FastAPI / HTTPException imports.
    """

    def __init__(
        self,
        model_path: str,
        model_loader: Callable,
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
        self._model_loader = model_loader
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
        _depth = get_server_max_queue_depth()
        # maxsize=0 → unbounded (Queue semantics); >0 → bounded with backpressure
        self.requests: Queue = Queue(maxsize=_depth)
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
        model, processor, config = self._model_loader(
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
            from ..speculative.drafters import load_drafter

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
        from ..server_schemas import get_server_max_tokens

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

        try:
            self.requests.put_nowait((rqueue, raw_inputs, prompt_tokens, args, images))
        except QueueFull:
            depth = self.requests.maxsize
            raise RuntimeError(
                f"Server request queue is full ({depth} pending requests). "
                "Try again later, or increase XMLX_VLM_MAX_QUEUE_DEPTH."
            )

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

                # --- Build per-sequence repetition-penalty function ---
                _rep_penalty = (
                    getattr(args, "repetition_penalty", None)
                    or float(os.environ.get("XMLX_VLM_REPETITION_PENALTY", "1.0"))
                )
                _rep_ctx_size = int(os.environ.get("XMLX_VLM_REPETITION_CONTEXT_SIZE", "20"))

                if _rep_penalty and _rep_penalty != 1.0:
                    # Capture mutable references so the closure sees live token_lists
                    _tl = token_lists
                    _uids = uids
                    _p = float(_rep_penalty)
                    _cs = _rep_ctx_size

                    def rep_penalty_fn(active_orig_idx, logits):
                        """Apply per-sequence repetition penalty.

                        active_orig_idx maps active-slot j → original-batch index.
                        logits: [B_active, bs, V].
                        """
                        penalized = []
                        V = logits.shape[-1]
                        for ai, orig_j in enumerate(active_orig_idx):
                            uid_j = _uids[orig_j]
                            ctx = _tl[uid_j][-_cs:]
                            seq_l = logits[ai]  # [bs, V]
                            if ctx:
                                ctx_arr = mx.array(ctx, dtype=mx.int32)
                                mask = mx.zeros(V, dtype=mx.bool_).at[ctx_arr].set(True)
                                is_pos = seq_l > 0
                                seq_l = mx.where(
                                    mask[None, :],
                                    mx.where(is_pos, seq_l / _p, seq_l * _p),
                                    seq_l,
                                )
                            penalized.append(seq_l)
                        return mx.stack(penalized, axis=0)
                else:
                    rep_penalty_fn = None

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
                if not is_mtp and rep_penalty_fn is not None:
                    rounds_kwargs["rep_penalty_fn"] = rep_penalty_fn
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
