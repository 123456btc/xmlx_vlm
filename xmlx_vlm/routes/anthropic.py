"""
Anthropic Messages API endpoint (/v1/messages).
"""
import logging
import time
import traceback
from typing import Any, List, Optional, Tuple, Union

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from ..auth import verify_api_key
from ..model_store import _INHERIT_ADAPTER, get_store
import uuid
from datetime import datetime

from ..server_schemas import (
    AnthropicMessageContent,
    AnthropicMessageRequest,
    AnthropicMessageResponse,
    AnthropicUsage,
    ChatMessage,
    ChatRequest,
    UsageStats,
    get_server_max_tokens,
)
from ..engine import _split_thinking, _infer_reasoning_parser
from ..reasoning import get_parser as _get_reasoning_parser_cls

logger = logging.getLogger("xmlx_vlm.routes.anthropic")

import asyncio
import json

from ..prompt_utils import apply_chat_template
from ..generate import stream_generate
from ..engine import _build_gen_args, _read_tenant_id
from ..config import _maybe_clear_cache
from .completions import chat_completions_endpoint

router = APIRouter()

@router.post("/v1/messages", response_model=None)
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
    model, processor, config = get_store().get_or_load(request.model)
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

            if get_store().response_generator is not None:
                ctx, token_iter = await asyncio.to_thread(
                    get_store().response_generator.generate,
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
                    vision_cache=get_store().cache.get("vision_cache"),
                    apc_manager=get_store().apc_manager,
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


