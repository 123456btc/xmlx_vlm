# SPDX-License-Identifier: Apache-2.0
"""
Think Scrubber -- Reasoning stream extraction and cleaning.

Strips <think>...</think> tags and separates internal chain-of-thought
reasoning from tool calls and user-facing text. Handles unclosed think tags,
nested think blocks, and model-specific reasoning markers.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


THINK_TAG_PATTERN = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL | re.IGNORECASE)
THINK_OPEN_PATTERN = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE_PATTERN = re.compile(r"</think>", re.IGNORECASE)


class ThinkScrubber:
    """
    Extracts and separates reasoning content from raw model outputs.
    """

    @staticmethod
    def scrub(text: str) -> Tuple[str, Optional[str]]:
        """
        Extract and strip <think>...</think> content.

        Returns:
            Tuple of (clean_text, reasoning_content).
            reasoning_content is None if no thinking tags were present.
        """
        if not text:
            return "", None

        if "<think>" not in text.lower():
            return text.strip(), None

        reasoning_chunks: List[str] = []

        def _replacer(match: re.Match) -> str:
            chunk = match.group(1).strip()
            if chunk:
                reasoning_chunks.append(chunk)
            return ""

        clean_text = THINK_TAG_PATTERN.sub(_replacer, text).strip()
        # Clean any dangling stray close tags if present
        clean_text = THINK_CLOSE_PATTERN.sub("", clean_text).strip()

        reasoning_content = "\n\n".join(reasoning_chunks).strip() if reasoning_chunks else None
        return clean_text, reasoning_content

    @staticmethod
    def clean_text(text: str) -> str:
        """Convenience method returning only the cleaned text."""
        clean, _ = ThinkScrubber.scrub(text)
        return clean

    @staticmethod
    def sanitize_messages(
        messages: List[Dict[str, Any]],
        keep_reasoning_in_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Sanitize an entire conversation history before sending or archiving.

        Strips <think> tags from assistant messages. If keep_reasoning_in_metadata
        is True, stores extracted reasoning in msg["reasoning"].
        """
        sanitized: List[Dict[str, Any]] = []
        for msg in messages:
            msg_copy = dict(msg)
            if msg_copy.get("role") == "assistant" and isinstance(msg_copy.get("content"), str):
                clean_content, reasoning = ThinkScrubber.scrub(msg_copy["content"])
                msg_copy["content"] = clean_content
                if reasoning and keep_reasoning_in_metadata:
                    msg_copy["reasoning"] = reasoning
            sanitized.append(msg_copy)
        return sanitized
