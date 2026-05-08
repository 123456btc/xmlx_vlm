# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for Gemma 4 models.

Gemma 4 uses a channel-based protocol:
  <|channel>thought\n...thinking...<channel|>...response...
"""

from .base import DeltaMessage
from .think_parser import BaseThinkingReasoningParser

_THOUGHT_PREFIX = "thought"
_RESPONSE_MARKER = "<|channel>response"
_THOUGHT_MARKER = "<|channel>thought"


def _strip_channel_name(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        text = text[len(prefix) :]
    return text.lstrip("\n")


def _strip_channel_tokens(text: str) -> str:
    text = text.replace("<channel|>", "")
    text = text.replace("<|channel>", "")
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        s = line.strip()
        if s in ("thought", "response"):
            continue
        cleaned.append(line)
    text = "\n".join(cleaned)
    text = text.strip()
    for name in ("thought", "response"):
        if text.startswith(name + "\n"):
            text = text[len(name) + 1 :]
            break
        if text.startswith(name) and (
            len(text) == len(name) or not text[len(name)].isalpha()
        ):
            text = text[len(name) :]
            break
    return text.strip()


class Gemma4ReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for Gemma 4 models."""

    @property
    def start_token(self) -> str:
        return "<|channel>"

    @property
    def end_token(self) -> str:
        return "<channel|>"

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._pending: str = ""
        self._content_seen: bool = False

    def reset_state(self):
        super().reset_state()
        self._pending = ""
        self._content_seen = False

    def _trailing_partial_marker_len(self, text: str) -> int:
        markers = (_RESPONSE_MARKER, _THOUGHT_MARKER, self.end_token, self.start_token)
        max_len = 0
        for marker in markers:
            for i in range(min(len(marker) - 1, len(text)), 0, -1):
                if text.endswith(marker[:i]):
                    if not text.endswith(marker):
                        if i > max_len:
                            max_len = i
                    break
        return max_len

    def finalize_stream(self) -> DeltaMessage | None:
        if not self._pending:
            return None
        pending = self._pending
        self._pending = ""
        if self._phase == "content":
            return DeltaMessage(content=pending)
        return DeltaMessage(reasoning=pending)

    def extract_reasoning(
        self,
        model_output: str,
    ) -> tuple[str | None, str | None]:
        text = model_output

        if self.start_token in text and self.end_token in text:
            _, _, after_start = text.partition(self.start_token)
            reasoning, _, content = after_start.rpartition(self.end_token)
            reasoning = _strip_channel_tokens(reasoning)
            content = _strip_channel_tokens(content)
            return reasoning or None, content or None

        if text.count(self.start_token) >= 2 and _RESPONSE_MARKER in text:
            _, _, after_start = text.partition(self.start_token)
            last_resp = after_start.rfind(_RESPONSE_MARKER)
            reasoning = after_start[:last_resp]
            content = after_start[last_resp + len(_RESPONSE_MARKER) :]
            reasoning = _strip_channel_tokens(reasoning)
            content = _strip_channel_tokens(content)
            return reasoning or None, content or None

        if self.end_token in text:
            reasoning, _, content = text.rpartition(self.end_token)
            reasoning = _strip_channel_tokens(reasoning)
            content = _strip_channel_tokens(content)
            return reasoning or None, content or None

        if self.start_token in text:
            _, _, reasoning = text.partition(self.start_token)
            reasoning = _strip_channel_tokens(reasoning)
            return reasoning or None, None

        return None, model_output

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        trailing = self._trailing_partial_marker_len(current_text)
        safe_current = current_text[:-trailing] if trailing else current_text
        prev_trailing = self._trailing_partial_marker_len(previous_text)
        safe_previous = (
            previous_text[:-prev_trailing] if prev_trailing else previous_text
        )

        self._pending = current_text[len(safe_current) :]

        if len(safe_current) <= len(safe_previous):
            return None

        safe_delta = safe_current[len(safe_previous) :]
        return self._extract_from_safe_text(safe_previous, safe_current, safe_delta)

    @staticmethod
    def _strip_channel_tokens_from_delta(
        msg: DeltaMessage | None,
    ) -> DeltaMessage | None:
        if msg is None:
            return None
        c = msg.content
        r = msg.reasoning
        if c is not None:
            c = c.replace("<channel|>", "").replace("<|channel>", "")
        if r is not None:
            r = r.replace("<channel|>", "").replace("<|channel>", "")
        if not c and not r:
            return None
        if c == msg.content and r == msg.reasoning:
            return msg
        return DeltaMessage(reasoning=r or None, content=c or None)

    def _extract_from_safe_text(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> DeltaMessage | None:
        if self.start_token not in current_text and self.end_token not in current_text:
            return DeltaMessage(content=delta_text)

        if _RESPONSE_MARKER in current_text and _RESPONSE_MARKER not in previous_text:
            self._phase = "content"
            self._content_seen = False
            marker_pos = current_text.find(_RESPONSE_MARKER)
            after_marker = current_text[marker_pos + len(_RESPONSE_MARKER) :]
            after_marker = after_marker.lstrip("\n")
            if after_marker:
                self._content_seen = True
                return self._strip_channel_tokens_from_delta(
                    DeltaMessage(content=after_marker)
                )
            return None

        cur_starts = current_text.count(self.start_token)
        prev_starts = previous_text.count(self.start_token)
        cur_ends = current_text.count(self.end_token)
        prev_ends = previous_text.count(self.end_token)

        if cur_starts > prev_starts:
            if self._phase != "thinking":
                self._phase = "thinking"
                self._content_seen = False
            return None

        if cur_ends > prev_ends:
            self._phase = "content"
            self._content_seen = False
            last_end = delta_text.rfind(self.end_token)
            if last_end >= 0:
                after = delta_text[last_end + len(self.end_token) :]
                after = _strip_channel_name(after.lstrip("\n"), _THOUGHT_PREFIX)
                after = _strip_channel_name(after, "response")
                if after:
                    self._content_seen = True
                    return DeltaMessage(content=after)
            return None

        if self._phase == "content":
            if not self._content_seen:
                stripped = delta_text.lstrip("\n")
                stripped = _strip_channel_name(stripped, _THOUGHT_PREFIX)
                stripped = _strip_channel_name(stripped, "response")
                self._content_seen = bool(stripped)
                if not stripped:
                    return None
                return self._strip_channel_tokens_from_delta(
                    DeltaMessage(content=stripped)
                )
            return self._strip_channel_tokens_from_delta(
                DeltaMessage(content=delta_text)
            )

        if self._phase == "thinking":
            if cur_starts > 0:
                after_ch = current_text.split(self.start_token, 1)[1]
                if after_ch.startswith(_THOUGHT_PREFIX):
                    clean = after_ch[len(_THOUGHT_PREFIX) :].lstrip("\n")
                    prev_after = ""
                    if self.start_token in previous_text:
                        prev_after = previous_text.split(self.start_token, 1)[1]
                    if prev_after.startswith(_THOUGHT_PREFIX):
                        prev_after = prev_after[len(_THOUGHT_PREFIX) :].lstrip("\n")
                    r = clean[len(prev_after) :]
                    return DeltaMessage(reasoning=r) if r else None
            return DeltaMessage(reasoning=delta_text) if delta_text else None

        return DeltaMessage(reasoning=delta_text) if delta_text else None
