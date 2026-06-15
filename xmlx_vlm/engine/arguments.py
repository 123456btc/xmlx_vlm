"""
Generation argument dataclass and request-to-args builder.

Extracted from server.py Phase 2 refactor.  This module is intentionally
free of FastAPI / uvicorn imports so it can be unit-tested in isolation.
"""
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import mlx.core as mx

from ..config import (
    DEFAULT_ENABLE_THINKING,
    DEFAULT_ENABLE_TOOL_LOGITS_BIAS,
    get_server_default_thinking_budget,
    get_server_enable_thinking,
    get_server_enable_tool_logits_bias,
)
from ..constrained import ThinkingAwareLogitsProcessor
from ..generate import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_THINKING_END_TOKEN,
    DEFAULT_THINKING_START_TOKEN,
    DEFAULT_TOP_P,
)
from ..server_schemas import get_server_max_tokens
from ..structured import build_json_schema_logits_processor
from ..tool_logits_bias import ToolLogitsBiasProcessor

logger = logging.getLogger("xmlx_vlm.engine.arguments")


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
    session_id: Optional[str] = None

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


# ---------------------------------------------------------------------------
# Request → GenerationArguments helpers
# ---------------------------------------------------------------------------

def _request_field_or_default(request, field_name: str, default):
    fields_set = getattr(request, "model_fields_set", None)
    if fields_set is not None and field_name not in fields_set:
        return default
    value = getattr(request, field_name, default)
    return default if value is None else value


def _model_config_field_or_default(processor, field_name: str, default):
    config = getattr(processor, "config", None)
    return getattr(config, field_name, default)


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
    # Apply server-wide default cap when client sends no budget at all.
    # This prevents infinite thinking loops in long multi-tool conversations
    # (e.g. Pi.dev after 10+ file reads) where the model never closes </think>.
    if thinking_budget is None:
        thinking_budget = get_server_default_thinking_budget()

    default_temperature = _model_config_field_or_default(
        processor, "temperature", DEFAULT_TEMPERATURE
    )
    default_top_p = _model_config_field_or_default(processor, "top_p", DEFAULT_TOP_P)
    default_top_k = _model_config_field_or_default(processor, "top_k", 0)
    if _model_config_field_or_default(processor, "do_sample", None) is False:
        default_temperature = 0.0

    args = GenerationArguments(
        max_tokens=max_tokens,
        temperature=_request_field_or_default(
            request, "temperature", default_temperature
        ),
        top_p=_request_field_or_default(request, "top_p", default_top_p),
        top_k=_request_field_or_default(request, "top_k", default_top_k),
        min_p=getattr(request, "min_p", 0.0),
        repetition_penalty=getattr(request, "repetition_penalty", None),
        logit_bias=logit_bias,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        thinking_start_token=getattr(request, "thinking_start_token", None),
        thinking_end_token=getattr(request, "thinking_end_token", None),
        tenant_id=tenant_id,
        enable_tool_logits_bias=enable_tool_logits_bias,
        session_id=getattr(request, "session_id", None),
    )
    if processor is not None:
        args.logits_processors = _build_structured_logits_processors(request, processor, args)
        # Jump-forward decoding: bias tool-call tokens when tools are present
        tools_present = bool(getattr(request, "tools", None))
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        args.maybe_add_tool_logits_bias(tokenizer, tools_present)
    return args
