# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for Qwen3 models.

Qwen3 uses <think>...</think> tags for reasoning content.
Supports implicit reasoning mode where <think> is injected in the prompt
by AI agents and only </think> appears in the output.
"""

from .think_parser import BaseThinkingReasoningParser


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Qwen3 models."""

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
        # If no end token at all, treat as pure content
        if self.end_token not in model_output:
            return None, model_output

        return super().extract_reasoning(model_output)
