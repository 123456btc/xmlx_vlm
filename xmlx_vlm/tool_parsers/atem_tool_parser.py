# SPDX-License-Identifier: Apache-2.0
"""
ATEM / Muse Glimmer tool call parser.

Handles Muse Glimmer and ATEM-style agent tool execution models with
deliberation stripping and structured tool extraction.

Supported syntax patterns:
1. XML tag format:
   <atem:call name="tool_name">{"location": "Tokyo"}</atem:call>
   <atem_tool_call>{"name": "tool_name", "arguments": {...}}</atem_tool_call>
   <atem>{"name": "tool_name", "parameters": {...}}</atem>
2. Channel format:
   <|channel|>call:tool_name\\n{"location": "Tokyo"}\\n<|endofcall|>
   or <|channel|>action:tool_name\\n{...}
3. Deliberation tags:
   <atem:deliberation>...</atem:deliberation> or <|channel|>thought...
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)
from .recovery import auto_recover_tool_calls


def _generate_tool_id() -> str:
    return f"call_{uuid.uuid4().hex[:8]}"


# Regex patterns for ATEM
_ATEM_CALL_TAG_PATTERN = re.compile(
    r"<atem:call\s+name=[\"']?([a-zA-Z0-9_\-\.:]+)[\"']?\s*>(.*?)(?:</atem:call>|$)",
    re.DOTALL | re.IGNORECASE,
)

_ATEM_TOOL_CALL_PATTERN = re.compile(
    r"<atem(?:_tool)?_call>(.*?)(?:</atem(?:_tool)?_call>|$)",
    re.DOTALL | re.IGNORECASE,
)

_ATEM_CHANNEL_CALL_PATTERN = re.compile(
    r"<\|channel\|>(?:call|action):([a-zA-Z0-9_\-\.:]+)\s*(.*?)(?:<\|endofcall\|>|<\|channel\|>|$)",
    re.DOTALL | re.IGNORECASE,
)

_ATEM_DELIBERATION_PATTERN = re.compile(
    r"<atem:deliberation>(.*?)(?:</atem:deliberation>|$)",
    re.DOTALL | re.IGNORECASE,
)

_THOUGHT_CHANNEL_PATTERN = re.compile(
    r"<\|channel\|>thought\s*(.*?)(?:<\|channel\|>|$)",
    re.DOTALL | re.IGNORECASE,
)


@ToolParserManager.register_module(["atem", "muse", "muse-glimmer", "muse_glimmer"])
class AtemToolParser(ToolParser):
    """
    Tool parser for Muse Glimmer / ATEM agent tool execution models.
    """

    extra_stop_tokens = [
        "</atem:call>",
        "</atem_tool_call>",
        "</atem>",
        "<|endofcall|>",
    ]

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        if not model_output:
            return ExtractedToolCallInformation(tools_called=False, tool_calls=[], content="")

        tool_calls = []
        cleaned_content = model_output

        # 1. Check <atem:call name="...">...</atem:call>
        for match in _ATEM_CALL_TAG_PATTERN.finditer(model_output):
            tool_name = match.group(1).strip()
            args_str = match.group(2).strip()
            try:
                args = json.loads(args_str)
                arg_json = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args_str)
            except Exception:
                arg_json = args_str

            tool_calls.append({
                "id": _generate_tool_id(),
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": arg_json,
                },
            })

        # 2. Check <atem_tool_call> or <atem>
        if not tool_calls:
            for match in _ATEM_TOOL_CALL_PATTERN.finditer(model_output):
                body = match.group(1).strip()
                try:
                    data = json.loads(body)
                    if isinstance(data, dict):
                        name = data.get("name") or data.get("tool") or (
                            data.get("function", {}).get("name") if isinstance(data.get("function"), dict) else None
                        )
                        raw_args = data.get("arguments") or data.get("parameters") or (
                            data.get("function", {}).get("arguments") if isinstance(data.get("function"), dict) else None
                        ) or {}
                        arg_json = json.dumps(raw_args, ensure_ascii=False) if isinstance(raw_args, dict) else str(raw_args)
                        if name:
                            tool_calls.append({
                                "id": _generate_tool_id(),
                                "type": "function",
                                "function": {
                                    "name": str(name),
                                    "arguments": arg_json,
                                },
                            })
                except Exception:
                    pass

        # 3. Check Channel format <|channel|>call:name ...
        if not tool_calls:
            for match in _ATEM_CHANNEL_CALL_PATTERN.finditer(model_output):
                tool_name = match.group(1).strip()
                args_str = match.group(2).strip()
                try:
                    args = json.loads(args_str)
                    arg_json = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args_str)
                except Exception:
                    arg_json = args_str

                tool_calls.append({
                    "id": _generate_tool_id(),
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arg_json,
                    },
                })

        # 4. If no tools found yet, try auto-recovery
        if not tool_calls:
            recovered = auto_recover_tool_calls(model_output)
            if recovered:
                for item in recovered:
                    name = item.get("name") or (
                        item.get("function", {}).get("name") if isinstance(item.get("function"), dict) else None
                    )
                    args = item.get("arguments") or (
                        item.get("function", {}).get("arguments") if isinstance(item.get("function"), dict) else item.get("parameters", {})
                    )
                    if name:
                        tool_calls.append({
                            "id": _generate_tool_id(),
                            "type": "function",
                            "function": {
                                "name": str(name),
                                "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args),
                            },
                        })

        # Strip tool blocks and deliberation from user-facing content
        cleaned_content = _ATEM_CALL_TAG_PATTERN.sub("", cleaned_content)
        cleaned_content = _ATEM_TOOL_CALL_PATTERN.sub("", cleaned_content)
        cleaned_content = _ATEM_CHANNEL_CALL_PATTERN.sub("", cleaned_content)
        cleaned_content = _ATEM_DELIBERATION_PATTERN.sub("", cleaned_content)
        cleaned_content = _THOUGHT_CHANNEL_PATTERN.sub("", cleaned_content)
        cleaned_content = cleaned_content.strip()

        return ExtractedToolCallInformation(
            tools_called=len(tool_calls) > 0,
            tool_calls=tool_calls,
            content=cleaned_content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> ExtractedToolCallInformation:
        res = self.extract_tool_calls(current_text, request=request)
        if res.tools_called:
            return res
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=delta_text,
        )
