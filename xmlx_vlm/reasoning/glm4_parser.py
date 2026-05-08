# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for GLM-4 models.

GLM-4 uses <think>...</think> tags, same as Qwen3.
However, GLM-4 does NOT inject <think> in the prompt — the model decides
autonomously whether to reason. Output without tags = normal response.
"""

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser

_BOX_START = "<|begin_of_box|>"
_BOX_END = "<|end_of_box|>"


class Glm4ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for GLM-4 models."""

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        cleaned = model_output.replace(_BOX_START, "").replace(_BOX_END, "")
        return super().extract_reasoning(cleaned)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        delta_text = delta_text.replace(_BOX_START, "").replace(_BOX_END, "")
        if not delta_text:
            return None

        start_tok = self.start_token
        end_tok = self.end_token

        if self._phase == "pre_think":
            if start_tok in current_text:
                return super().extract_reasoning_streaming(
                    previous_text, current_text, delta_text
                )
            if end_tok in current_text:
                return super().extract_reasoning_streaming(
                    previous_text, current_text, delta_text
                )
            return DeltaMessage(content=delta_text)

        return super().extract_reasoning_streaming(
            previous_text, current_text, delta_text
        )
