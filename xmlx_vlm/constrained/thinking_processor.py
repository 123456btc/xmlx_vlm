"""Thinking-aware logits processor for reasoning models.

Manages the full thinking lifecycle: budget enforcement, phase transitions,
and content-phase constrained decoding delegation.
"""

from __future__ import annotations

import enum
import os
from collections import deque
from typing import Callable, Optional, Union

import mlx.core as mx


class BoundedSuffixMatcher:
    """Detect a target token sequence in a stream using a rolling suffix buffer.

    Unlike a naive sequential matcher that resets to position 0 on mismatch,
    this uses a bounded buffer that catches overlapping prefixes.
    """

    __slots__ = ("target", "_buf", "_max_len")

    def __init__(self, target_ids: list[int]) -> None:
        if not target_ids:
            raise ValueError("target_ids must be non-empty")
        self.target = tuple(target_ids)
        self._max_len = len(target_ids)
        self._buf: deque[int] = deque(maxlen=self._max_len)

    def feed(self, token_id: int) -> bool:
        """Feed one token. Returns True when the buffer suffix equals the target."""
        self._buf.append(token_id)
        return len(self._buf) == self._max_len and tuple(self._buf) == self.target

    def reset(self) -> None:
        """Clear the buffer."""
        self._buf.clear()

    def snapshot(self) -> tuple[int, ...]:
        """Return a serializable copy of the current suffix buffer."""
        return tuple(self._buf)

    def restore(self, state: tuple[int, ...]) -> None:
        """Restore the suffix buffer from a previous snapshot."""
        self._buf.clear()
        self._buf.extend(state)


class Phase(enum.Enum):
    """Thinking lifecycle phases."""

    IDLE = "idle"
    THINKING = "thinking"
    TRANSITIONING = "transitioning"
    CONTENT = "content"


class AdaptiveThinkingBudget:
    """Compute thinking token budgets from string effort levels or prompt heuristics.

    Maps OpenAI-compatible ``reasoning_effort`` strings to concrete token budgets:
      - ``off`` / ``disabled`` -> 0
      - ``low`` / ``minimal``  -> 256
      - ``medium``             -> 512 (default)
      - ``high``               -> 1024
      - ``xhigh`` / ``max``    -> 2048

    When ``adaptive=True``, the budget is further scaled by prompt complexity
    heuristics (token count, code/math density).
    """

    _LEVELS = {
        "off": 0,
        "disabled": 0,
        "none": 0,
        "low": 256,
        "minimal": 256,
        "medium": 512,
        "high": 1024,
        "xhigh": 2048,
        "max": 2048,
    }

    def __init__(
        self,
        default_budget: int = 512,
        adaptive: bool = False,
        min_budget: int = 128,
        max_budget: int = 4096,
    ):
        self.default_budget = default_budget
        self.adaptive = adaptive
        self.min_budget = min_budget
        self.max_budget = max_budget

    def resolve(
        self,
        budget: Union[int, str, None],
        *,
        prompt_token_count: int = 0,
    ) -> int:
        """Resolve a budget specification to a concrete token count.

        Args:
            budget: Raw budget -- int, string level, or None.
            prompt_token_count: Number of tokens in the prompt for adaptive scaling.

        Returns:
            Concrete thinking token budget (>= 0).
        """
        if budget is None:
            base = self.default_budget
        elif isinstance(budget, int):
            base = max(0, budget)
        elif isinstance(budget, str):
            key = budget.strip().lower()
            base = self._LEVELS.get(key, self.default_budget)
        else:
            base = self.default_budget

        if not self.adaptive or base == 0:
            return base

        # Adaptive scaling: longer / more complex prompts get proportionally
        # more thinking budget, capped at max_budget.
        scale = 1.0
        if prompt_token_count > 4096:
            scale = 1.5
        elif prompt_token_count > 2048:
            scale = 1.25
        elif prompt_token_count > 512:
            scale = 1.1

        adjusted = int(base * scale)
        return max(self.min_budget, min(adjusted, self.max_budget))


def resolve_thinking_budget(
    budget: Union[int, str, None],
    *,
    prompt_token_count: int = 0,
    default_budget: int = 512,
) -> int:
    """Global convenience helper to resolve a thinking budget."""
    adaptive = os.environ.get("XMLX_VLM_ADAPTIVE_THINKING", "").lower() in (
        "1",
        "true",
        "yes",
    )
    resolver = AdaptiveThinkingBudget(
        default_budget=default_budget, adaptive=adaptive
    )
    return resolver.resolve(budget, prompt_token_count=prompt_token_count)


class ThinkingAwareLogitsProcessor:
    """Unified logits processor for thinking-model lifecycle management.

    Manages a four-phase state machine:
      IDLE -> THINKING -> TRANSITIONING -> CONTENT

    - IDLE: before reasoning start tokens. Pass through.
    - THINKING: inside reasoning span. Count tokens, pass through.
    - TRANSITIONING: forcing reasoning end sequence via logits masking.
    - CONTENT: after reasoning closed. Delegate to inner processor.

    No re-entry into THINKING after CONTENT is reached.
    """

    __slots__ = (
        "_start_matcher",
        "_end_matcher",
        "_end_token_ids",
        "_content_phase_mask_ids",
        "_thinking_token_budget",
        "_inner",
        "_vocab_size",
        "_state",
        "_thinking_tokens",
        "_transition_index",
        "_processed_len",
        "_processed_token_ids",
        "_snapshots",
    )

    def __init__(
        self,
        start_token_ids: list[int],
        end_token_ids: list[int],
        thinking_token_budget: Union[int, str, None] = 512,
        inner: Callable[[mx.array, mx.array], mx.array] | None = None,
        vocab_size: int = 152064,
        prompt_has_think_tag: bool = False,
        prompt_token_count: int = 0,
    ) -> None:
        self._start_matcher = BoundedSuffixMatcher(start_token_ids)
        self._end_matcher = BoundedSuffixMatcher(end_token_ids)
        self._end_token_ids = list(end_token_ids)
        # Mask only the first token of each sequence: sufficient because most
        # tokenizers encode <think>/<|think|> as a single special token.
        self._content_phase_mask_ids = tuple(
            dict.fromkeys([start_token_ids[0], end_token_ids[0]])
        )
        self._thinking_token_budget = resolve_thinking_budget(
            thinking_token_budget,
            prompt_token_count=prompt_token_count,
        )
        self._inner = inner
        self._vocab_size = vocab_size
        self._thinking_tokens = 0
        self._transition_index = 0
        # When the chat template already injected <think> into the prompt,
        # the first generated token is already inside the thinking span.
        # Start in THINKING (or TRANSITIONING if budget=0) instead of IDLE.
        if prompt_has_think_tag:
            if self._thinking_token_budget == 0:
                self._state = Phase.TRANSITIONING
            else:
                self._state = Phase.THINKING
        else:
            self._state = Phase.IDLE
        self._processed_len = 0
        self._processed_token_ids: list[int] = []
        self._snapshots = [self._snapshot_state()]

    @property
    def state(self) -> Phase:
        return self._state

    @property
    def thinking_tokens(self) -> int:
        return self._thinking_tokens

    @property
    def is_retired(self) -> bool:
        """True when the processor is in CONTENT with no inner constraint.

        The engine can use this signal to drop the processor and re-enable
        MTP for the remaining content generation (Phase 2 optimization).
        """
        return self._state == Phase.CONTENT and self._inner is None

    def clone(self) -> "ThinkingAwareLogitsProcessor":
        """Deep-clone for safe use across batch entries."""
        cloned = object.__new__(ThinkingAwareLogitsProcessor)
        cloned._start_matcher = BoundedSuffixMatcher(list(self._start_matcher.target))
        cloned._start_matcher.restore(self._start_matcher.snapshot())
        cloned._end_matcher = BoundedSuffixMatcher(list(self._end_matcher.target))
        cloned._end_matcher.restore(self._end_matcher.snapshot())
        cloned._end_token_ids = list(self._end_token_ids)
        cloned._content_phase_mask_ids = self._content_phase_mask_ids
        cloned._thinking_token_budget = self._thinking_token_budget
        cloned._inner = self._inner.clone() if hasattr(self._inner, "clone") else self._inner
        cloned._vocab_size = self._vocab_size
        cloned._state = self._state
        cloned._thinking_tokens = self._thinking_tokens
        cloned._transition_index = self._transition_index
        cloned._processed_len = self._processed_len
        cloned._processed_token_ids = list(self._processed_token_ids)
        cloned._snapshots = list(self._snapshots)
        return cloned

    def __call__(self, tokens: mx.array, logits: mx.array) -> mx.array:
        # The MLLM scheduler applies processors before the first completion
        # token is emitted, so ``tokens`` can be empty on step 0.
        if tokens.size == 0:
            if self._state == Phase.TRANSITIONING:
                return self._force_transition(logits)
            if self._state == Phase.CONTENT:
                return self._call_inner(tokens, logits)
            return logits

        self._sync_to_tokens(tokens)

        if self._state == Phase.TRANSITIONING:
            return self._force_transition(logits)

        # Phase.CONTENT
        if self._state == Phase.CONTENT:
            return self._call_inner(tokens, logits)
        return logits

    def _force_transition(self, logits: mx.array) -> mx.array:
        """Force the next token in the reasoning end sequence."""
        target_id = self._end_token_ids[self._transition_index]
        # Mask all logits to -inf, then set the target token to 0.
        # Handle both 1-D (vocab,) and 2-D (1, vocab) logits shapes.
        masked = mx.full(logits.shape, float("-inf"))
        if masked.ndim == 1:
            masked[target_id] = 0.0
        else:
            masked[..., target_id] = 0.0
        return masked

    def _call_inner(self, tokens: mx.array, logits: mx.array) -> mx.array:
        """Delegate to inner processor if present."""
        if self._inner is not None:
            logits = self._inner(tokens, logits)
        return self._mask_content_phase_control_tokens(logits)

    def _mask_content_phase_control_tokens(self, logits: mx.array) -> mx.array:
        """Prevent reserved think-tag starts from leaking into final content."""
        for token_id in self._content_phase_mask_ids:
            if logits.ndim == 1:
                logits[token_id] = float("-inf")
            else:
                logits[..., token_id] = float("-inf")
        return logits

    def _snapshot_state(
        self,
    ) -> tuple[Phase, int, int, tuple[int, ...], tuple[int, ...]]:
        return (
            self._state,
            self._thinking_tokens,
            self._transition_index,
            self._start_matcher.snapshot(),
            self._end_matcher.snapshot(),
        )

    def _restore_snapshot(self, processed_len: int) -> None:
        # In CONTENT phase, snapshots stop growing (see _sync_to_tokens).
        # If rollback targets a CONTENT position beyond the snapshot list,
        # use the last available snapshot -- the state is identical since
        # _advance_with_token is a no-op in CONTENT.
        snap_idx = min(processed_len, len(self._snapshots) - 1)
        (
            self._state,
            self._thinking_tokens,
            self._transition_index,
            start_state,
            end_state,
        ) = self._snapshots[snap_idx]
        self._start_matcher.restore(start_state)
        self._end_matcher.restore(end_state)
        self._processed_len = processed_len
        self._processed_token_ids = self._processed_token_ids[:processed_len]
        self._snapshots = self._snapshots[: snap_idx + 1]

    def _sync_to_tokens(self, tokens: mx.array) -> None:
        target_len = int(tokens.size)
        token_ids = tokens.tolist()
        common_len = 0
        max_common = min(target_len, self._processed_len)
        while (
            common_len < max_common
            and self._processed_token_ids[common_len] == token_ids[common_len]
        ):
            common_len += 1
        if common_len < self._processed_len:
            self._restore_snapshot(common_len)
        if target_len == self._processed_len:
            return
        for token_id in token_ids[self._processed_len :]:
            self._advance_with_token(token_id)
            self._processed_token_ids.append(token_id)
            self._processed_len += 1
            # Skip snapshots in CONTENT -- _advance_with_token is a no-op
            # there, so snapshots would just waste memory on long generations.
            if self._state != Phase.CONTENT:
                self._snapshots.append(self._snapshot_state())

    def _advance_with_token(self, token_id: int) -> None:
        if self._state == Phase.IDLE:
            if self._start_matcher.feed(token_id):
                self._state = Phase.THINKING
                if self._thinking_token_budget == 0:
                    self._state = Phase.TRANSITIONING
                    self._transition_index = 0
            return

        if self._state == Phase.THINKING:
            if self._end_matcher.feed(token_id):
                self._state = Phase.CONTENT
                return
            self._thinking_tokens += 1
            if self._thinking_tokens >= self._thinking_token_budget:
                self._state = Phase.TRANSITIONING
                self._transition_index = 0
            return

        if self._state == Phase.TRANSITIONING:
            expected = self._end_token_ids[self._transition_index]
            if token_id == expected:
                self._transition_index += 1
                if self._transition_index >= len(self._end_token_ids):
                    self._state = Phase.CONTENT
                    self._end_matcher.reset()
            return
