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

    Three bugs addressed:

    1. **State reset**: without ``tc_end`` tracking, ``in_tool_call`` stays
       ``True`` forever — the Pi.dev "⠋ Working…" spinner that never resolves.

    2. **Closing-tag leak**: on the True→False transition the chunk *containing*
       the closing delimiter was passed through as ``delta_content``, leaking
       ``</tool_call>`` verbatim.

    3. **Bare ``<function=`` leak** (the main cause of the visible XML noise):
       Some chat templates inject ``<tool_call>`` as a *generation prefix*
       (part of the prompt, not generated tokens). The model then outputs only
       ``<function=name><parameter=p>v</parameter></function>`` — the outer
       ``<tool_call>`` wrapper never appears in ``full_output``, so
       ``tc_start in full_output`` is always False and nothing gets suppressed.
       Fix: when ``tc_start == "<tool_call>"`` is absent from ``full_output``
       but ``<function=`` is present, switch to the bare-function delimiter pair
       ``("<function=", "</function>")``.  A trailing ``</tool_call>`` emitted
       by the model after ``</function>`` is also stripped.

    Multiple tool calls in one reply are handled correctly: after each end
    delimiter the state resets, and a subsequent start re-enters the suppression
    window.  Text that follows the closing delimiter within the same streaming
    chunk is recovered and passed through.
    """
    if not tc_start:
        return in_tool_call, delta_content

    # ── Bare-function mode detection ─────────────────────────────────────────
    # Triggered when the chat template injects tc_start as a prompt prefix so
    # it never appears in the generated token stream.
    _BARE_START = "<function="
    _BARE_END   = "</function>"

    # Decide which delimiter pair governs this call.
    # We lock to bare mode once <function= has been seen WITHOUT tc_start,
    # which happens on the token that completes "<function=".  Before that,
    # the prefix-suppression logic below eats the leading "<" silently.
    _use_bare = (
        tc_start == "<tool_call>"
        and tc_start not in full_output
        and _BARE_START in full_output
    )
    eff_start = _BARE_START if _use_bare else tc_start
    eff_end   = _BARE_END   if _use_bare else tc_end

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _after_last_end() -> bool:
        """True when the last eff_end in full_output comes after the last eff_start."""
        if not eff_end:
            return False
        last_end = full_output.rfind(eff_end)
        if last_end == -1:
            return False
        last_start = full_output.rfind(eff_start)
        return last_end >= last_start + len(eff_start)

    def _recover_after(delimiter: str) -> Optional[str]:
        """Return text that trails ``delimiter`` in delta_content, or None."""
        if delimiter and delta_content and delimiter in delta_content:
            tail = delta_content.split(delimiter, 1)[1]
            # Strip a stray </tool_call> that sometimes follows </function>
            if _use_bare and "</tool_call>" in tail:
                tail = tail.replace("</tool_call>", "")
            return tail if tail else None
        return None

    # ── State machine ─────────────────────────────────────────────────────────

    if not in_tool_call:
        if eff_start in full_output:
            if _after_last_end():
                # Standard mode: tool call fully closed — pass content through.
                return False, delta_content
            return True, None

        # eff_start not yet complete — suppress prefix build-up.
        # Check BOTH tc_start and _BARE_START so the leading "<" is caught
        # regardless of which format the model is using.
        prefixes_to_check = [tc_start]
        if tc_start == "<tool_call>":
            prefixes_to_check.append(_BARE_START)
        for pfx in prefixes_to_check:
            if any(full_output.endswith(pfx[:j]) for j in range(1, len(pfx))):
                return False, None

    else:
        # ── Standard mode exit ───────────────────────────────────────────────
        if not _use_bare:
            if _after_last_end():
                return False, _recover_after(eff_end)
            return True, None

        # ── Bare mode: stay suppressed until real content arrives ────────────
        # After </function> the model often emits \n</tool_call>\n before the
        # actual response.  We must suppress that entire cleanup window, not
        # just the </function> chunk.  Strategy:
        #   1. If </function> has NOT yet appeared → still inside the call.
        #   2. If </function> HAS appeared:
        #      a. Examine full_output tail after last </function>.
        #      b. If tail is only whitespace / </tool_call> → still cleaning up.
        #      c. Once real content appears in the tail → exit and recover it.
        func_end_pos = full_output.rfind(_BARE_END)
        if func_end_pos == -1:
            # </function> not yet seen
            return True, None

        func_start_pos = full_output.rfind(_BARE_START)
        if func_end_pos < func_start_pos + len(_BARE_START):
            # A NEW <function= opened AFTER the last </function> — still inside
            return True, None

        # </function> has closed the last <function= block.
        # Inspect everything that appeared after it.
        tail_after_func = full_output[func_end_pos + len(_BARE_END):]

        # Strip whitespace then any COMPLETE </tool_call> instances.
        tail_no_ws = re.sub(r'\s', '', tail_after_func)
        tail_no_tc = tail_no_ws.replace("</tool_call>", "")

        if not tail_no_tc:
            # Only whitespace / complete </tool_call> so far — still cleanup.
            return True, None

        if "</tool_call>".startswith(tail_no_tc):
            # Partial </tool_call> being assembled char-by-char — keep suppressing.
            return True, None

        # Real content has arrived.  Recover it from the current delta.
        if _BARE_END in delta_content:
            # Closing tag and real content arrived in the SAME chunk.
            after = delta_content.split(_BARE_END, 1)[1]
            after = after.replace("</tool_call>", "")
            return False, after if after else None
        else:
            # Closing tag was a previous chunk; this delta IS the real content
            # (possibly prefixed by whitespace / </tool_call> cleanup tokens).
            cleaned = delta_content.replace("</tool_call>", "") if delta_content else ""
            return False, cleaned if cleaned else None

    return in_tool_call, delta_content

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
