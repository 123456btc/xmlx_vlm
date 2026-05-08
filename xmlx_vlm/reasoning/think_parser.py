# SPDX-License-Identifier: Apache-2.0
"""
Base parser for models using <think>...</think> tags for reasoning.

Supports three scenarios:
1. Both tags in output: <think>reasoning</think>content
2. Only closing tag (think injected in prompt): reasoning</think>content
3. No tags: pure content

Performance: The streaming parser uses a simple state machine to track the
current phase (pre-think / thinking / content).
"""

import logging
import re
from abc import abstractmethod

from .base import DeltaMessage, ReasoningParser

logger = logging.getLogger(__name__)


class BaseThinkingReasoningParser(ReasoningParser):
    """Base parser for models using <think>...</think> style tags.

    Subclasses define the specific start and end tokens.
    """

    @property
    @abstractmethod
    def start_token(self) -> str:
        """The token/tag that starts reasoning content (e.g., '<think>')."""

    @property
    @abstractmethod
    def end_token(self) -> str:
        """The token/tag that ends reasoning content (e.g., '</think>')."""

    _TOOL_CALL_START = "<tool_call>"
    _TOOL_CALL_END = "</tool_call>"
    _TOOL_CALL_CLOSED_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    _TOOL_CALL_UNCLOSED_RE = re.compile(r"<tool_call>\s*[\{<].*$", re.DOTALL)

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._phase: str = "pre_think"
        self._content_started = False
        self._content_buffer = ""
        self._in_tool_call = False
        self._tool_call_buffer = ""

    def reset_state(self):
        """Reset state machine for a new streaming request."""
        self._phase = "pre_think"
        self._content_started = False
        self._content_buffer = ""
        self._in_tool_call = False
        self._tool_call_buffer = ""

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        """Extract reasoning from complete output.

        Handles three cases:
        1. Both tags present
        2. Only closing tag (think in prompt)
        3. No tags: pure content
        """
        text = model_output

        if self.end_token in text:
            reasoning, content = self._extract_complete_reasoning(text)
            return self._promote_tool_calls(reasoning, content)

        if self.start_token in text:
            _, _, reasoning = text.partition(self.start_token)
            reasoning = reasoning.strip() or None
            return self._promote_tool_calls(reasoning, None)

        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Extract reasoning from a streaming delta using state-machine tracking."""
        if not delta_text:
            return None

        start_tok = self.start_token
        end_tok = self.end_token

        if self._phase == "pre_think":
            if start_tok in current_text:
                self._phase = "thinking"
                idx = delta_text.find(start_tok)
                after = delta_text[idx + len(start_tok) :] if idx >= 0 else delta_text

                if end_tok in after:
                    self._phase = "content"
                    eidx = after.find(end_tok)
                    reasoning = after[:eidx]
                    content = after[eidx + len(end_tok) :]
                    return self._transition_to_content(reasoning, content)

                tc_start = self._TOOL_CALL_START
                if tc_start in after:
                    tc_idx = after.find(tc_start)
                    self._in_tool_call = True
                    self._tool_call_buffer = after[tc_idx:]
                    before = after[:tc_idx]
                    return DeltaMessage(reasoning=before) if before else None

                return DeltaMessage(reasoning=after) if after else None

            if end_tok in current_text:
                self._phase = "content"
                idx = delta_text.find(end_tok)
                if idx >= 0:
                    reasoning = delta_text[:idx]
                    content = delta_text[idx + len(end_tok) :]
                else:
                    reasoning = None
                    content = delta_text
                return self._transition_to_content(reasoning, content)

            return DeltaMessage(reasoning=delta_text)

        if self._phase == "thinking":
            if self._in_tool_call:
                return self._thinking_tool_call(previous_text, current_text, delta_text)

            tc_start = self._TOOL_CALL_START
            if tc_start in current_text and tc_start not in previous_text:
                self._in_tool_call = True
                idx = delta_text.find(tc_start)
                if idx >= 0:
                    reasoning = delta_text[:idx]
                    self._tool_call_buffer = delta_text[idx:]
                else:
                    self._tool_call_buffer = tc_start
                    reasoning = delta_text
                return DeltaMessage(reasoning=reasoning) if reasoning else None

            if end_tok in current_text and end_tok not in previous_text:
                self._phase = "content"
                idx = delta_text.find(end_tok)
                if idx >= 0:
                    reasoning = delta_text[:idx]
                    content = delta_text[idx + len(end_tok) :]
                else:
                    reasoning = delta_text
                    content = None
                return self._transition_to_content(reasoning, content)
            return DeltaMessage(reasoning=delta_text)

        if self._phase == "content":
            return self._content_delta(delta_text)

        return None

    def _extract_complete_reasoning(self, text: str) -> tuple[str | None, str | None]:
        """Split complete output into leading reasoning spans and final content."""
        reasoning_parts: list[str] = []
        remainder = text

        while remainder:
            stripped = remainder.lstrip()

            if stripped.startswith(self.start_token):
                after_start = stripped[len(self.start_token) :]
                reasoning, found, after_end = after_start.partition(self.end_token)
                if not found:
                    reasoning_parts.append(reasoning)
                    remainder = ""
                    break
                if reasoning.strip():
                    reasoning_parts.append(reasoning.strip())
                remainder = after_end
                continue

            start_idx = stripped.find(self.start_token)
            end_idx = stripped.find(self.end_token)
            if end_idx != -1 and (start_idx == -1 or end_idx < start_idx):
                reasoning = stripped[:end_idx]
                if reasoning.strip():
                    reasoning_parts.append(reasoning.strip())
                remainder = stripped[end_idx + len(self.end_token) :]
                continue

            remainder = stripped
            break

        reasoning = "\n".join(reasoning_parts).strip() or None
        content = remainder.strip() or None
        return reasoning, content

    def _transition_to_content(
        self, reasoning: str | None, content: str | None
    ) -> DeltaMessage | None:
        """Return a delta while suppressing leading post-transition think blocks."""
        reasoning, content = self._promote_tool_calls(reasoning, content)
        content_msg = self._content_delta(content or "")
        extra_reasoning = content_msg.reasoning if content_msg else None
        final_content = content_msg.content if content_msg else None
        reasoning_text = (reasoning or "") + (extra_reasoning or "")
        if not reasoning_text and not final_content:
            return None
        return DeltaMessage(
            reasoning=reasoning_text or None,
            content=final_content or None,
        )

    def _content_delta(self, delta_text: str) -> DeltaMessage | None:
        """Emit content after consuming repeated leading think blocks."""
        if not delta_text and not self._content_buffer:
            return None

        if self._content_started:
            return DeltaMessage(content=delta_text) if delta_text else None

        self._content_buffer += delta_text
        buffer = self._content_buffer.lstrip()
        reasoning_parts: list[str] = []

        while buffer:
            if buffer.startswith(self.end_token):
                buffer = buffer[len(self.end_token) :].lstrip()
                continue

            if buffer.startswith(self.start_token):
                after_start = buffer[len(self.start_token) :]
                end_idx = after_start.find(self.end_token)
                if end_idx == -1:
                    self._content_buffer = buffer
                    return None
                reasoning = after_start[:end_idx]
                if reasoning:
                    reasoning_parts.append(reasoning)
                buffer = after_start[end_idx + len(self.end_token) :].lstrip()
                continue

            if self.start_token.startswith(buffer):
                self._content_buffer = buffer
                return None

            if self.end_token.startswith(buffer):
                self._content_buffer = buffer
                return None

            self._content_started = True
            self._content_buffer = ""
            return DeltaMessage(
                reasoning="".join(reasoning_parts) or None,
                content=buffer,
            )

        self._content_buffer = ""
        if reasoning_parts:
            return DeltaMessage(reasoning="".join(reasoning_parts))
        return None

    def _thinking_tool_call(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        """Handle streaming while inside a <tool_call> during thinking phase."""
        tc_end = self._TOOL_CALL_END
        end_tok = self.end_token

        if tc_end in current_text and tc_end not in previous_text:
            self._tool_call_buffer += delta_text
            idx = self._tool_call_buffer.find(tc_end)
            promoted = self._tool_call_buffer[: idx + len(tc_end)]
            remainder = self._tool_call_buffer[idx + len(tc_end) :]
            self._tool_call_buffer = ""
            self._in_tool_call = False
            logger.warning("Promoted streaming tool_call block from reasoning")

            if end_tok in remainder:
                self._phase = "content"
                eidx = remainder.find(end_tok)
                reasoning = remainder[:eidx].strip() or None
                after_think = remainder[eidx + len(end_tok) :]
                content_msg = self._content_delta(after_think) if after_think else None
                final_content = promoted + (
                    (content_msg.content or "") if content_msg else ""
                )
                extra_r = content_msg.reasoning if content_msg else None
                r_text = (reasoning or "") + (extra_r or "")
                return DeltaMessage(
                    content=final_content or None,
                    reasoning=r_text or None,
                )

            tc_start = self._TOOL_CALL_START
            if tc_start in remainder:
                tc_idx = remainder.find(tc_start)
                self._in_tool_call = True
                self._tool_call_buffer = remainder[tc_idx:]
                reasoning = remainder[:tc_idx].strip() or None
                return DeltaMessage(content=promoted, reasoning=reasoning)

            reasoning = remainder.strip() or None
            return DeltaMessage(content=promoted, reasoning=reasoning)

        if end_tok in current_text and end_tok not in previous_text:
            self._tool_call_buffer += delta_text
            self._in_tool_call = False
            self._phase = "content"
            logger.warning(
                "Promoted unclosed streaming tool_call (think ended before tool_call closed)"
            )
            idx = self._tool_call_buffer.find(end_tok)
            if idx >= 0:
                promoted = self._tool_call_buffer[:idx]
                after = self._tool_call_buffer[idx + len(end_tok) :]
            else:
                promoted = self._tool_call_buffer
                after = ""
            self._tool_call_buffer = ""
            content_msg = self._content_delta(after) if after else None
            final_content = (
                promoted + (content_msg.content or "") if content_msg else promoted
            )
            return DeltaMessage(content=final_content or None)

        self._tool_call_buffer += delta_text
        return None

    def finalize_stream(self) -> DeltaMessage | None:
        """Flush any buffered tool call text at end of stream."""
        if self._in_tool_call and self._tool_call_buffer:
            promoted = self._tool_call_buffer
            self._tool_call_buffer = ""
            self._in_tool_call = False
            logger.warning("Promoted unclosed streaming tool_call at stream end")
            return DeltaMessage(content=promoted)
        return None

    @classmethod
    def _promote_tool_calls(
        cls, reasoning: str | None, content: str | None
    ) -> tuple[str | None, str | None]:
        if not reasoning or "<tool_call>" not in reasoning:
            return reasoning, content

        closed: list[str] = []

        def _collect_closed(match):
            closed.append(match.group(0))
            return ""

        cleaned = cls._TOOL_CALL_CLOSED_RE.sub(_collect_closed, reasoning)

        unclosed_match = cls._TOOL_CALL_UNCLOSED_RE.search(cleaned)
        unclosed_block = None
        if unclosed_match:
            unclosed_block = unclosed_match.group(0)
            cleaned = cleaned[: unclosed_match.start()]

        cleaned = cleaned.strip() or None
        promoted_count = len(closed) + (1 if unclosed_block else 0)

        if promoted_count == 0:
            return reasoning, content

        result_content = content or ""

        if unclosed_block:
            result_content = (
                unclosed_block + "\n" + result_content
                if result_content
                else unclosed_block
            )

        if closed:
            closed_text = "\n".join(closed)
            result_content = (
                result_content + "\n" + closed_text if result_content else closed_text
            )

        result_content = result_content.strip() or None

        logger.warning(
            "Promoted %d tool_call block(s) from reasoning to content (%d closed, %d unclosed)",
            promoted_count,
            len(closed),
            1 if unclosed_block else 0,
        )

        return cleaned, result_content
