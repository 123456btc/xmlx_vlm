"""
Streaming helpers: tool-call suppression, reasoning parsing, logprob formatting.

Extracted from server.py Phase 2 refactor.  Pure-Python — no MLX, no FastAPI.
"""
import json
import logging
import re
import uuid
from typing import List, Optional, Tuple

logger = logging.getLogger("xmlx_vlm.engine.streaming")


def suppress_tool_call_content(
    full_output: str,
    in_tool_call: bool,
    tc_start: Optional[str],
    delta_content: Optional[str],
    tc_end: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """Suppress tool-call markup from streamed delta.content.

    Returns updated ``(in_tool_call, delta_content)``.

    Tracks *both* ``tc_start`` and ``tc_end`` so that ``in_tool_call`` reverts
    to ``False`` once the closing delimiter has been consumed.

    Two distinct bugs are addressed here:

    1. **State reset**: without tracking ``tc_end``, ``in_tool_call`` stays
       ``True`` for the rest of the request — the Pi.dev "⠋ Working…" spinner
       that never resolves in long agent loops.

    2. **Closing-tag leak**: when the state machine transitions out of the
       tool-call window, the chunk that *contains* the closing delimiter (e.g.
       ``</tool_call>``) must also be suppressed; the old code returned it as
       ``delta_content``, which caused ``</parameter>``, ``</function>``, and
       ``</tool_call>`` to appear verbatim in the streamed response.

    Multiple tool calls in one reply are handled correctly: after each
    ``tc_end`` the state resets, and a subsequent ``tc_start`` re-enters the
    suppression window.  Any text that appears *after* ``tc_end`` within the
    same streaming chunk is recovered and passed through.
    """
    if not tc_start:
        return in_tool_call, delta_content

    def _after_last_end() -> bool:
        """True when the last tc_end in full_output comes after the last tc_start."""
        if not tc_end:
            return False
        last_end = full_output.rfind(tc_end)
        if last_end == -1:
            return False
        last_start = full_output.rfind(tc_start)
        # Must be past tc_start + its own length to count as "closed"
        return last_end >= last_start + len(tc_start)

    if not in_tool_call:
        if tc_start in full_output:
            if _after_last_end():
                # Tool call fully closed — pass content through
                return False, delta_content
            return True, None
        # tc_start not yet complete — suppress while we're building the prefix
        if any(full_output.endswith(tc_start[:j]) for j in range(1, len(tc_start))):
            return False, None
    else:
        # Currently inside a tool call; exit when the closing delimiter arrives
        if _after_last_end():
            # Suppress the closing delimiter itself; recover any text that
            # trails it within the same streaming chunk (rare but possible).
            after_end = ""
            if tc_end and delta_content and tc_end in delta_content:
                after_end = delta_content.split(tc_end, 1)[1]
            return False, after_end if after_end else None
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


def _infer_reasoning_parser(model, config) -> "str | None":
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
):
    """Build an OpenAI-style logprob entry for a single token."""
    # Lazy import to avoid pulling server_schemas at module load time
    from ..server_schemas import ChatLogprobContent, TopLogprob

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
