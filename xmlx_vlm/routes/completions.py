"""
OpenAI Chat Completions endpoint (/chat/completions, /v1/chat/completions).
"""
import logging
import time
import traceback
from typing import Any, List, Optional, Tuple, Union

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ..auth import verify_api_key
from ..model_store import _INHERIT_ADAPTER, get_store
import json
import re
import uuid
from datetime import datetime

from ..server_schemas import (
    ChatChoice,
    ChatLogprobContent,
    ChatLogprobs,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatStreamChoice,
    ChatStreamChunk,
    TopLogprob,
    UsageStats,
    get_server_max_tokens,
)
from ..generate import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    normalize_resize_shape,
    stream_generate,
)
from ..prompt_utils import apply_chat_template
from ..engine import (
    _build_gen_args,
    _read_tenant_id,
    _split_thinking,
    _count_thinking_tag_tokens,
    _decode_token,
    _make_logprob_content,
    suppress_tool_call_content,
    process_tool_calls,
    _infer_reasoning_parser,
)
from ..config import (
    _maybe_clear_cache,
    get_prefill_step_size,
    get_top_logprobs_k,
)
from ..tool_parsers import _infer_tool_parser_from_processor, load_tool_module
from ..reasoning import get_parser as _get_reasoning_parser_cls
from ..metrics import metrics

logger = logging.getLogger("xmlx_vlm.routes.completions")

import asyncio
from concurrent.futures import ThreadPoolExecutor
_COMPLETIONS_EXECUTOR = ThreadPoolExecutor(
    max_workers=256,
    thread_name_prefix="xmlx_vlm_completions",
)
import gc

import mlx.core as mx

from ..mcp import get_manager
from ..generate import generate

router = APIRouter()

@router.post("/chat/completions", response_model=None)
@router.post("/v1/chat/completions", response_model=None, include_in_schema=False)
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
        model, processor, config = get_store().get_or_load(request.model, adapter_path)

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
                token_iterator = None
                token_iter = None  # For ResponseGenerator cleanup
                try:
                    # Use ResponseGenerator if available, otherwise fall back to stream_generate
                    if get_store().response_generator is not None:
                        # generate() does blocking Queue.get — run off event loop
                        try:
                            ctx, token_iter = await asyncio.get_running_loop().run_in_executor(
                                _COMPLETIONS_EXECUTOR,
                                get_store().response_generator.generate,
                                formatted_prompt,
                                images if images else None,
                                audio if audio else None,
                                gen_args,
                            )
                        except RuntimeError as _qe:
                            _msg = str(_qe)
                            if "request queue is full" in _msg:
                                # Yield a well-formed SSE error then a 503-style
                                # terminal chunk so the client sees the problem.
                                yield (
                                    f"data: {{\"error\": {{\"message\": \"{_msg}\","
                                    f" \"type\": \"server_error\", \"code\": 503}}}}\n\n"
                                )
                                return
                            raise

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

                        # SSE keepalive: emit a comment every N seconds when no
                        # data chunk is yielded (e.g. during long thinking phases
                        # where delta_content is suppressed).  Prevents proxies
                        # and agent clients (Pi.dev, etc.) from closing an idle
                        # connection before the model finishes reasoning.
                        _SSE_KEEPALIVE_INTERVAL = 15.0  # seconds
                        _last_sse_time = time.monotonic()

                        while True:
                            token = await asyncio.get_running_loop().run_in_executor(
                                _COMPLETIONS_EXECUTOR, _next_token
                            )
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
                                full_output, in_tool_call, tc_start, delta_content,
                                tc_end=tc_end,
                            )

                            chunk_logprobs = None
                            if request.logprobs and token.finish_reason != "stop":
                                req_top_k = int(request.top_logprobs or 0)
                                chunk_logprobs = ChatLogprobs(
                                    content=[
                                        _make_logprob_content(
                                            get_store().response_generator.tokenizer,
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
                                _last_sse_time = time.monotonic()
                            else:
                                # No payload this token (thinking, suppressed markup).
                                # Emit an SSE comment to keep the connection alive.
                                _now = time.monotonic()
                                if _now - _last_sse_time >= _SSE_KEEPALIVE_INTERVAL:
                                    yield ": keepalive\n\n"
                                    _last_sse_time = _now

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
                            vision_cache=get_store().cache.get("vision_cache"),
                            logits_processors=gen_args.logits_processors,
                            apc_manager=get_store().apc_manager,
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

                if get_store().response_generator is not None:

                    def _blocking_generate():
                        text = ""
                        pt = gt = 0
                        pm = 0.0
                        ctx, token_iter = get_store().response_generator.generate(
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
                        await asyncio.get_running_loop().run_in_executor(
                            _COMPLETIONS_EXECUTOR, _blocking_generate
                        )
                    )
                else:
                    gen_result = generate(
                        model=model,
                        processor=processor,
                        prompt=formatted_prompt,
                        image=images,
                        audio=audio,
                        verbose=logger.isEnabledFor(logging.DEBUG),
                        vision_cache=get_store().cache.get("vision_cache"),
                        apc_manager=get_store().apc_manager,
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


