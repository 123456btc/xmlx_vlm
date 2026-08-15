# SPDX-License-Identifier: Apache-2.0
"""Broken-output auto-recovery for tool calls.

When a quantized model emits malformed tool calls (missing closing tags,
truncated JSON, mixed XML/text), this module attempts heuristic repair
before giving up.

Inspired by Rapid-MLX's auto-recovery layer:
https://github.com/raullenchai/Rapid-MLX
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_TOOL_CALL_START_MARKERS = [
    "<tool_call>",
    "<function=",
    "<|tool_call>",
    "[TOOL_CALLS]",
    "[Calling tool:",
    " Calling tool:",
    "<invoke>",
    "<minimax:tool_call>",
    "<atem:call",
    "<atem_tool_call>",
    "<atem>",
    "<|channel|>call:",
    "<|channel|>action:",
]

# Patterns that suggest a tool call was intended but garbled
_TOOL_CALL_SIGNALS = re.compile(
    r"(<tool_call>|<function=|\[TOOL_CALLS\]|\[Calling tool:|Calling tool:|"
    r"<invoke>|<minimax:tool_call>|<atem:call|<atem_tool_call>|<atem>|"
    r"<\|channel\|>call:|<\|channel\|>action:)",
    re.IGNORECASE,
)


def _repair_xml_tags(text: str) -> str:
    """Close unclosed XML tags that are common in tool-call formats."""
    if "<tool_call>" in text and "</tool_call>" not in text:
        text = text + "</tool_call>"
    if "<function=" in text and "</function>" not in text:
        idx = text.rfind("<function=")
        if idx >= 0 and "</function>" not in text[idx:]:
            text = text + "</function>"
    if "<invoke>" in text and "</invoke>" not in text:
        text = text + "</invoke>"
    if "<minimax:tool_call>" in text and "</minimax:tool_call>" not in text:
        text = text + "</minimax:tool_call>"
    if "<atem:call" in text and "</atem:call>" not in text:
        text = text + "</atem:call>"
    if "<atem_tool_call>" in text and "</atem_tool_call>" not in text:
        text = text + "</atem_tool_call>"
    if "<atem>" in text and "</atem>" not in text:
        text = text + "</atem>"
    return text


def _clean_json_str(s: str) -> str:
    """Strip markdown code fence if present and normalize trailing commas."""
    s = s.strip()
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()
    # Remove trailing commas before closing braces/brackets
    s = re.sub(r",\s*([\]}])", r"\1", s)
    return s


def _extract_json_objects(text: str) -> list[dict[str, Any]]:
    """Extract balanced JSON objects from messy text, tolerant of truncation."""
    results: list[dict[str, Any]] = []
    text = _clean_json_str(text)
    depth = 0
    start = None
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = _clean_json_str(text[start : i + 1])
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and any(
                        k in obj
                        for k in (
                            "name",
                            "type",
                            "function",
                            "tool",
                            "action",
                            "arguments",
                            "parameters",
                        )
                    ):
                        results.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
        elif depth < 0:
            depth = 0
            start = None
    return results


def _repair_truncated_json(text: str) -> str:
    """Heuristically close a truncated JSON object, handling unclosed strings & brackets."""
    text = _clean_json_str(text)
    brace_stack = []
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if ch == "{":
            brace_stack.append("}")
        elif ch == "[":
            brace_stack.append("]")
        elif ch in ("}", "]"):
            if brace_stack and brace_stack[-1] == ch:
                brace_stack.pop()

    repaired = text
    if in_string:
        repaired += '"'

    repaired = _clean_json_str(repaired)

    while brace_stack:
        closer = brace_stack.pop()
        repaired += closer

    return repaired


def _extract_from_repaired_xml(text: str) -> list[dict[str, Any]] | None:
    """Repair XML tags then look for JSON objects inside them."""
    repaired = _repair_xml_tags(text)
    results: list[dict[str, Any]] = []

    # Look inside <tool_call>...</tool_call>
    patterns = [
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        r"<atem(?:_tool)?_call>\s*(.*?)\s*</atem(?:_tool)?_call>",
        r"<minimax:tool_call>\s*(.*?)\s*</minimax:tool_call>",
        r"<invoke>\s*(.*?)\s*</invoke>",
    ]
    for pat in patterns:
        matches = re.findall(pat, repaired, re.DOTALL | re.IGNORECASE)
        for m in matches:
            m = m.strip()
            if not m:
                continue
            try:
                obj = json.loads(_clean_json_str(m))
                if isinstance(obj, dict):
                    results.append(obj)
            except json.JSONDecodeError:
                fixed = _repair_truncated_json(m)
                try:
                    obj = json.loads(_clean_json_str(fixed))
                    if isinstance(obj, dict):
                        results.append(obj)
                except json.JSONDecodeError:
                    pass

    return results if results else None


def _extract_bare_json(text: str) -> list[dict[str, Any]] | None:
    """Try to find raw JSON objects or arrays in the text."""
    text = _clean_json_str(text)
    # Try array first
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError:
            fixed_arr = _repair_truncated_json(text)
            try:
                parsed = json.loads(fixed_arr)
                if isinstance(parsed, list):
                    return [item for item in parsed if isinstance(item, dict)]
            except json.JSONDecodeError:
                pass

    # Try individual objects
    objs = _extract_json_objects(text)
    if objs:
        return objs

    # Try truncated-json repair on the whole text
    fixed = _repair_truncated_json(text)
    objs = _extract_json_objects(fixed)
    if objs:
        return objs
    return None


def attempt_recovery(text: str) -> list[dict[str, Any]] | None:
    """Attempt to recover tool calls from broken model output.

    Returns a list of recovered dicts (each must have at least a "name" or
    "type" key), or None if nothing recoverable was found.
    """
    if not text:
        return None

    # Quick reject: no tool-call signals at all
    if not _TOOL_CALL_SIGNALS.search(text):
        return None

    # Strategy 1: repair XML tags and extract JSON from inside
    result = _extract_from_repaired_xml(text)
    if result:
        logger.debug("Auto-recovery: extracted %d tool call(s) from repaired XML", len(result))
        return result

    # Strategy 2: bare JSON extraction (with truncation repair)
    result = _extract_bare_json(text)
    if result:
        logger.debug("Auto-recovery: extracted %d tool call(s) from bare JSON", len(result))
        return result

    return None


def auto_recover_tool_calls(text: str) -> list[dict[str, Any]] | None:
    """Alias for attempt_recovery."""
    return attempt_recovery(text)
