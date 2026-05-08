# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for GPT-OSS models using channel-based format.

GPT-OSS uses channel-based tokens:
  <|channel|>analysis<|message|>[reasoning]
  <|start|>assistant<|channel|>final<|message|>[content]<|return|>
"""

import re

from .base import DeltaMessage, ReasoningParser

_STRUCTURAL_TOKENS = re.compile(
    r"<\|start\|>|<\|end\|>|<\|channel\|>|<\|return\|>|<\|call\|>|<\|constrain\|>"
)

_CHANNEL_RE = re.compile(
    r"<\|channel\|>(analysis|final)(?:[^<]*(?:<\|constrain\|>[^<]*)?)?<\|message\|>"
)


def _extract_channel(text: str, channel_name: str) -> str | None:
    for m in _CHANNEL_RE.finditer(text):
        if m.group(1) == channel_name:
            start = m.end()
            end_match = _STRUCTURAL_TOKENS.search(text, start)
            content = text[start : end_match.start()] if end_match else text[start:]
            content = content.strip()
            return content if content else None
    return None


class GptOssReasoningParser(ReasoningParser):
    """Reasoning parser for GPT-OSS models."""

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        if not model_output or "<|channel|>" not in model_output:
            return None, model_output if model_output else None

        reasoning = _extract_channel(model_output, "analysis")
        content = _extract_channel(model_output, "final")

        if content:
            content = content.replace("<|return|>", "").strip()
            content = _STRUCTURAL_TOKENS.sub("", content).strip()
            content = content if content else None

        if reasoning:
            reasoning = _STRUCTURAL_TOKENS.sub("", reasoning).strip()
            reasoning = reasoning if reasoning else None

        if reasoning is None and content is None:
            return None, model_output

        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        prev_phase = self._detect_phase(previous_text)
        curr_phase = self._detect_phase(current_text)

        if curr_phase != prev_phase and curr_phase in ("analysis", "final"):
            after_marker = self._extract_content_after_marker_in_delta(
                current_text, curr_phase
            )
            if after_marker:
                after_marker = self._strip_return(after_marker)
                if curr_phase == "analysis":
                    return DeltaMessage(reasoning=after_marker)
                else:
                    return DeltaMessage(content=after_marker)
            return None

        if curr_phase == "analysis":
            cleaned = self._strip_return(delta_text)
            if _STRUCTURAL_TOKENS.search(cleaned):
                cleaned = _STRUCTURAL_TOKENS.sub("", cleaned)
            if cleaned:
                return DeltaMessage(reasoning=cleaned)
            return None
        elif curr_phase == "final":
            cleaned = self._strip_return(delta_text)
            if _STRUCTURAL_TOKENS.search(cleaned):
                cleaned = _STRUCTURAL_TOKENS.sub("", cleaned)
            if cleaned:
                return DeltaMessage(content=cleaned)
            return None

        return None

    @staticmethod
    def _detect_phase(text: str) -> str:
        matches = list(_CHANNEL_RE.finditer(text))
        if not matches:
            return "init"

        last = matches[-1]
        if last.group(1) == "final":
            return "final"

        after = text[last.end():]
        if _STRUCTURAL_TOKENS.search(after):
            return "transition"
        return "analysis"

    @staticmethod
    def _extract_content_after_marker_in_delta(
        current_text: str, phase: str
    ) -> str | None:
        channel_name = "analysis" if phase == "analysis" else "final"
        matches = list(_CHANNEL_RE.finditer(current_text))
        for m in reversed(matches):
            if m.group(1) == channel_name:
                return current_text[m.end():]
        return None

    @staticmethod
    def _strip_return(text: str) -> str:
        return text.replace("<|return|>", "")
