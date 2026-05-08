# SPDX-License-Identifier: Apache-2.0
"""Base classes for reasoning content extraction."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class DeltaMessage:
    """Delta message for streaming reasoning output."""

    role: str | None = None
    content: str | None = None
    reasoning: str | None = None

    @property
    def reasoning_content(self) -> str | None:
        """Deprecated: use reasoning instead."""
        return self.reasoning


class ReasoningParser(ABC):
    """Abstract base class for reasoning content extraction.

    Reasoning parsers extract thinking/reasoning content from model outputs,
    separating it from the final response content.
    """

    def __init__(self, tokenizer: Any | None = None):
        self.tokenizer = tokenizer

    @abstractmethod
    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning content from complete model output.

        Returns:
            Tuple of (reasoning_content, final_content).
            Either may be None if not present.
        """
        pass

    @abstractmethod
    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Extract reasoning from streaming delta.

        Uses the "previous + delta = current" model where:
        - previous_text: All text accumulated before this delta
        - current_text: All text including this delta
        - delta_text: Just the new text in this chunk
        """
        pass

    def reset_state(self):
        """Reset any internal state for a new request."""
        pass

    def finalize_stream(self) -> DeltaMessage | None:
        """Finalize streaming state at end of stream."""
        return None
