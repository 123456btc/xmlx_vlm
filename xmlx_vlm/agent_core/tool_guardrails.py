# SPDX-License-Identifier: Apache-2.0
"""
Tool Guardrails -- Pure-functional loop detection and execution guardrails.

Tracks per-turn tool call observations, detects infinite loops, repeated failures,
and stagnant execution states. Emits synthetic warning prompts or hard-stop halts
to prevent runaway agent loops.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple


IDEMPOTENT_TOOL_NAMES: FrozenSet[str] = frozenset(
    {
        "snapshot",
        "read_file",
        "search_files",
        "web_search",
        "web_extract",
        "session_search",
        "browser_snapshot",
        "get_state",
        "list_dir",
        "view_file",
        "grep_search",
    }
)

MUTATING_TOOL_NAMES: FrozenSet[str] = frozenset(
    {
        "click",
        "type_text",
        "type_into",
        "scroll",
        "navigate",
        "js",
        "comment",
        "write_file",
        "terminal",
        "run_command",
        "patch",
        "execute_code",
        "delegate_task",
        "post_comment",
    }
)


@dataclass(frozen=True)
class ToolCallGuardrailConfig:
    """Thresholds for per-turn tool-call loop and failure detection."""

    warnings_enabled: bool = True
    hard_stop_enabled: bool = True
    exact_failure_warn_after: int = 2
    exact_failure_block_after: int = 4
    same_tool_failure_warn_after: int = 3
    same_tool_failure_halt_after: int = 6
    no_progress_warn_after: int = 2
    no_progress_block_after: int = 4
    pattern_loop_warn_after: int = 2
    
    # Anti-overtrading & execution throttle
    re_entry_cooldown_seconds: float = 0.0      # E.g. 1800s (30m) cooldown after closing a symbol
    hourly_entry_limit: int = 0                 # E.g. max 2 new entries per hour (0 = unlimited)
    min_hold_seconds: float = 0.0               # E.g. 300s minimum holding time before closing
    
    idempotent_tools: FrozenSet[str] = field(default_factory=lambda: IDEMPOTENT_TOOL_NAMES)
    mutating_tools: FrozenSet[str] = field(default_factory=lambda: MUTATING_TOOL_NAMES)


@dataclass(frozen=True)
class GuardrailDecision:
    """Action verdict returned by the guardrail controller."""

    action: str  # "proceed", "warn", "block", "halt"
    synthetic_message: Optional[str] = None
    reason: Optional[str] = None

    @property
    def should_halt(self) -> bool:
        return self.action == "halt"

    @property
    def should_block(self) -> bool:
        return self.action in ("block", "halt")


@dataclass
class ToolCallObservation:
    """Observation record for a single tool call."""

    tool: str
    args_hash: str
    args_repr: str
    is_error: bool
    result_snippet: str
    state_signature: Optional[str] = None  # e.g., URL or page DOM hash


class ToolCallGuardrails:
    """
    Maintains conversation-scoped tool execution telemetry and evaluates loop safety.
    """

    def __init__(self, config: Optional[ToolCallGuardrailConfig] = None):
        self.config = config or ToolCallGuardrailConfig()
        self.history: List[ToolCallObservation] = []
        self.action_history: List[str] = []
        self._exact_failure_counts: Dict[str, int] = {}
        self._consecutive_same_tool_failures: Dict[str, int] = {}
        self._consecutive_no_progress_count: int = 0
        self._last_state_signature: Optional[str] = None
        
        # Position and throttle tracking
        self._position_open_times: Dict[str, float] = {}
        self._position_close_times: Dict[str, float] = {}
        self._recent_entries: List[float] = []

    @staticmethod
    def _compute_args_hash(tool: str, args: Dict[str, Any]) -> str:
        try:
            serialized = json.dumps({"tool": tool, "args": args}, sort_keys=True)
        except Exception:
            serialized = f"{tool}:{str(args)}"
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def observe_and_check(
        self,
        tool: str,
        args: Dict[str, Any],
        result: Any,
        is_error: bool = False,
        state_signature: Optional[str] = None,
    ) -> GuardrailDecision:
        """
        Record a completed tool call and evaluate whether safety thresholds are breached.
        """
        import time as _time
        now_ts = _time.time()
        args_dict = args if isinstance(args, dict) else {}

        # 0. Anti-Overtrading & Throttle Checks
        is_trading = tool in ("trading", "oms", "place_order")
        action = str(args_dict.get("action", "")).lower()
        symbol = str(args_dict.get("symbol", "")).upper()
        
        is_open_action = action in ("place_order", "open_long", "open_short", "buy", "sell")
        is_close_action = action in ("close_position", "close_long", "close_short", "emergency_stop")

        if is_trading and is_open_action and symbol:
            # Check re-entry cooldown
            if self.config.re_entry_cooldown_seconds > 0:
                last_close_time = self._position_close_times.get(symbol)
                if last_close_time and (now_ts - last_close_time) < self.config.re_entry_cooldown_seconds:
                    remaining = int(self.config.re_entry_cooldown_seconds - (now_ts - last_close_time))
                    return GuardrailDecision(
                        action="block",
                        reason=f"Re-entry cooldown active for {symbol} ({remaining}s remaining).",
                        synthetic_message=(
                            f"🛑 RE-ENTRY BLOCKED: Symbol `{symbol}` was closed recently. "
                            f"A mandatory cooldown of {remaining}s remains to prevent revenge trading / overtrading. "
                            "Wait for new setup or focus on other symbols."
                        ),
                    )
            
            # Check hourly entry limit
            if self.config.hourly_entry_limit > 0:
                one_hour_ago = now_ts - 3600
                valid_recent = [t for t in self._recent_entries if t >= one_hour_ago]
                if len(valid_recent) >= self.config.hourly_entry_limit:
                    return GuardrailDecision(
                        action="block",
                        reason=f"Hourly entry limit of {self.config.hourly_entry_limit} reached.",
                        synthetic_message=(
                            f"🛑 THROTTLE BLOCKED: Maximum {self.config.hourly_entry_limit} new entries per hour reached. "
                            "Hold existing positions or wait for the next trading window to avoid overtrading."
                        ),
                    )

        # Update position tracking
        if is_trading and not is_error:
            if is_open_action and symbol:
                self._position_open_times[symbol] = now_ts
                self._recent_entries.append(now_ts)
            elif is_close_action and symbol:
                self._position_close_times[symbol] = now_ts
                self._position_open_times.pop(symbol, None)

        args_hash = self._compute_args_hash(tool, args)
        result_str = str(result)
        if not is_error:
            # Auto-detect error in result string if not explicitly flagged
            err_lower = result_str.lower()
            if err_lower.startswith("error:") or "failed to" in err_lower or "exception:" in err_lower:
                is_error = True

        obs = ToolCallObservation(
            tool=tool,
            args_hash=args_hash,
            args_repr=json.dumps(args, sort_keys=True) if isinstance(args, dict) else str(args),
            is_error=is_error,
            result_snippet=result_str[:200],
            state_signature=state_signature,
        )
        self.history.append(obs)

        action_key = f"{tool}:{args_hash}"
        self.action_history.append(action_key)
        if len(self.action_history) > 16:
            self.action_history.pop(0)

        # 1. Track exact failure counts
        if is_error:
            self._exact_failure_counts[args_hash] = self._exact_failure_counts.get(args_hash, 0) + 1
            self._consecutive_same_tool_failures[tool] = self._consecutive_same_tool_failures.get(tool, 0) + 1
        else:
            self._consecutive_same_tool_failures[tool] = 0

        exact_fail_count = self._exact_failure_counts.get(args_hash, 0)
        tool_fail_count = self._consecutive_same_tool_failures.get(tool, 0)

        # Check exact failure thresholds
        if self.config.hard_stop_enabled and exact_fail_count >= self.config.exact_failure_block_after:
            return GuardrailDecision(
                action="block",
                reason=f"Exact call to '{tool}' failed {exact_fail_count} times.",
                synthetic_message=(
                    f"🛑 ACTION BLOCKED: Calling `{tool}` with identical arguments has failed {exact_fail_count} times. "
                    "You must switch to a different strategy, alternative tool, or summarize current progress."
                ),
            )

        if self.config.warnings_enabled and exact_fail_count >= self.config.exact_failure_warn_after:
            return GuardrailDecision(
                action="warn",
                reason=f"Exact call to '{tool}' failed {exact_fail_count} times.",
                synthetic_message=(
                    f"⚠️ WARNING: Identical call `{tool}` failed {exact_fail_count} times. "
                    "Stop repeating the same parameters and try another approach."
                ),
            )

        # Check consecutive tool failure thresholds
        if self.config.hard_stop_enabled and tool_fail_count >= self.config.same_tool_failure_halt_after:
            return GuardrailDecision(
                action="halt",
                reason=f"Tool '{tool}' failed {tool_fail_count} consecutive times.",
                synthetic_message=(
                    f"🛑 AGENT HALTED: Tool `{tool}` failed {tool_fail_count} times consecutively. "
                    "Stopping loop to avoid resource exhaustion."
                ),
            )

        if self.config.warnings_enabled and tool_fail_count >= self.config.same_tool_failure_warn_after:
            return GuardrailDecision(
                action="warn",
                reason=f"Tool '{tool}' failed {tool_fail_count} consecutive times.",
                synthetic_message=(
                    f"⚠️ WARNING: Tool `{tool}` has failed {tool_fail_count} times in a row. "
                    "Consider checking syntax, parameters, or utilizing an alternate tool."
                ),
            )

        # 2. Check Stagnant / No-Progress State for Mutating Actions
        if tool in self.config.mutating_tools and state_signature is not None:
            if self._last_state_signature is not None and state_signature == self._last_state_signature:
                self._consecutive_no_progress_count += 1
            else:
                self._consecutive_no_progress_count = 0
            self._last_state_signature = state_signature

            if self.config.hard_stop_enabled and self._consecutive_no_progress_count >= self.config.no_progress_block_after:
                return GuardrailDecision(
                    action="block",
                    reason=f"No state change after {self._consecutive_no_progress_count} mutating actions.",
                    synthetic_message=(
                        f"🛑 NO-PROGRESS BLOCKED: {self._consecutive_no_progress_count} consecutive mutating actions "
                        "resulted in no observable change in page/system state. Please verify prerequisites or try a completely new path."
                    ),
                )

            if self.config.warnings_enabled and self._consecutive_no_progress_count >= self.config.no_progress_warn_after:
                return GuardrailDecision(
                    action="warn",
                    reason=f"No state change after {self._consecutive_no_progress_count} mutating actions.",
                    synthetic_message=(
                        f"⚠️ WARNING: The state signature remained unchanged after `{tool}`. "
                        "The action likely had no effect. Try inspecting elements or running scripts instead."
                    ),
                )
        elif state_signature is not None:
            self._last_state_signature = state_signature

        # 3. Check Repeating Action Patterns (A-B-A-B loop)
        if len(self.action_history) >= 4:
            if self.action_history[-1] == self.action_history[-3] and self.action_history[-2] == self.action_history[-4]:
                return GuardrailDecision(
                    action="warn",
                    reason="A-B-A-B oscillating loop pattern detected.",
                    synthetic_message=(
                        "⚠️ LOOP DETECTED: You are oscillating between two repetitive actions without advancing. "
                        "Break out of this loop immediately and pursue a new angle."
                    ),
                )

        return GuardrailDecision(action="proceed")
