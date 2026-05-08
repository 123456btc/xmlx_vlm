# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for DeepSeek-R1 models.

DeepSeek-R1 uses <think>...</think> tags. The model may sometimes start
outputting reasoning without the explicit <think> tag.
"""

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser


class DeepSeekR1ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for DeepSeek-R1 model."""

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
        # If we have end token but no start token, treat beginning as reasoning
        if self.end_token in model_output and self.start_token not in model_output:
            return self._extract_complete_reasoning(model_output)

        if self.end_token not in model_output and self.start_token not in model_output:
            return None, model_output

        return super().extract_reasoning(model_output)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        result = super().extract_reasoning_streaming(
            previous_text, current_text, delta_text
        )

        if result is not None:
            start_in_prev = self.start_token in previous_text
            start_in_delta = self.start_token in delta_text
            end_in_delta = self.end_token in delta_text

            if not start_in_prev and not start_in_delta and end_in_delta:
                idx = delta_text.find(self.end_token)
                reasoning_part = delta_text[:idx]
                content_part = delta_text[idx + len(self.end_token) :]
                return DeltaMessage(
                    reasoning=reasoning_part if reasoning_part else None,
                    content=content_part if content_part else None,
                )

        return result
