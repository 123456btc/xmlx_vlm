"""
xmlx_vlm.engine — generation pipeline sub-package.

Public API re-exported here so callers can do:

    from .engine import (
        GenerationArguments, GenerationContext, StreamingToken,
        ResponseGenerator,
        suppress_tool_call_content, process_tool_calls,
        _build_gen_args, _read_tenant_id,
        _infer_reasoning_parser, _split_thinking,
        _count_thinking_tag_tokens, _decode_token, _make_logprob_content,
    )
"""

from .arguments import (
    GenerationArguments,
    _as_plain_dict,
    _build_gen_args,
    _build_structured_logits_processors,
    _extract_response_format_schema,
    _read_tenant_id,
    _request_field_or_default,
)
from .generation import (
    GenerationContext,
    ResponseGenerator,
    StreamingToken,
    _get_draft_block_size_from_env,
    _get_speculative_rounds_batch,
    _speculative_hidden_state,
    _speculative_prefill_kwargs,
)
from .streaming import (
    _count_thinking_tag_tokens,
    _decode_token,
    _infer_reasoning_parser,
    _make_logprob_content,
    _split_thinking,
    process_tool_calls,
    suppress_tool_call_content,
)

__all__ = [
    # arguments
    "GenerationArguments",
    "_build_gen_args",
    "_read_tenant_id",
    "_request_field_or_default",
    "_as_plain_dict",
    "_extract_response_format_schema",
    "_build_structured_logits_processors",
    # generation
    "GenerationContext",
    "StreamingToken",
    "ResponseGenerator",
    "_get_speculative_rounds_batch",
    "_speculative_prefill_kwargs",
    "_speculative_hidden_state",
    "_get_draft_block_size_from_env",
    # streaming
    "suppress_tool_call_content",
    "process_tool_calls",
    "_infer_reasoning_parser",
    "_count_thinking_tag_tokens",
    "_split_thinking",
    "_decode_token",
    "_make_logprob_content",
]
