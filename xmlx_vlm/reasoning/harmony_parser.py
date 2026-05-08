# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for Harmony format.

Harmony uses channels for reasoning vs final content:

  <|channel|>analysis
  <|message|>Let me think about this...
  <|end|>
  <|channel|>final
  <|message|>The answer is 42.
  <|return|>
"""

import re

from .base import DeltaMessage, ReasoningParser

_ANALYSIS_PATTERN = re.compile(
    r"<\|channel\|>analysis\s*<\|message\|>(.*?)<\|end\|>",
    re.DOTALL,
)

_FINAL_PATTERN = re.compile(
    r"<\|channel\|>final\s*<\|message\|>(.*?)<\|return\|>",
    re.DOTALL,
)


class HarmonyReasoningParser(ReasoningParser):
    """Reasoning parser for GPT-OSS models using Harmony format."""

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._current_channel: str | None = None
        self._in_message: bool = False

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        analysis_blocks = _ANALYSIS_PATTERN.findall(model_output)
        reasoning = "\n".join(block.strip() for block in analysis_blocks) or None

        final_match = _FINAL_PATTERN.search(model_output)
        content = final_match.group(1).strip() if final_match else None

        return reasoning, content

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        if "<|channel|>" in delta_text:
            if "analysis" in delta_text:
                self._current_channel = "analysis"
                self._in_message = False
                return None
            elif "final" in delta_text:
                self._current_channel = "final"
                self._in_message = False
                return None
            elif "commentary" in delta_text:
                self._current_channel = "commentary"
                self._in_message = False
                return None

        if self._current_channel is None and "<|channel|>" in current_text:
            last_channel = current_text.rfind("<|channel|>")
            after = current_text[last_channel + len("<|channel|>") :]
            if after.startswith("analysis"):
                self._current_channel = "analysis"
            elif after.startswith("final"):
                self._current_channel = "final"
            elif after.startswith("commentary"):
                self._current_channel = "commentary"

        if "<|message|>" in delta_text:
            self._in_message = True
            return None

        if any(
            token in delta_text
            for token in ("<|end|>", "<|return|>", "<|call|>", "<|start|>")
        ):
            self._in_message = False
            return None

        if delta_text.strip().startswith("<|") and delta_text.strip().endswith("|>"):
            return None

        if self._in_message and self._current_channel == "analysis":
            return DeltaMessage(reasoning=delta_text)

        if self._in_message and self._current_channel == "final":
            return DeltaMessage(content=delta_text)

        return None

    def reset_state(self):
        """Reset streaming state for a new request."""
        self._current_channel = None
        self._in_message = False
