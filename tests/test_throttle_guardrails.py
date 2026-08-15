"""
Unit tests for anti-overtrading throttling & prompt discipline.
"""

import time
import pytest

from xmlx_vlm.agent_core.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrails
from xmlx_vlm.ai_trader.decision.prompt_builder import PromptBuilder, DEFAULT_SYSTEM_PROMPT


# ─── 1. Re-Entry Cooldown Tests ───────────────────────────────────────────────

def test_re_entry_cooldown_blocking():
    guardrails = ToolCallGuardrails(
        ToolCallGuardrailConfig(
            re_entry_cooldown_seconds=10.0,  # 10s cooldown for testing
        )
    )

    # 1. Open BTC position
    d1 = guardrails.observe_and_check(
        tool="trading",
        args={"action": "place_order", "symbol": "BTC", "side": "buy"},
        result="Order submitted",
        is_error=False,
    )
    assert d1.action == "proceed"

    # 2. Close BTC position
    d2 = guardrails.observe_and_check(
        tool="trading",
        args={"action": "close_position", "symbol": "BTC"},
        result="Position closed",
        is_error=False,
    )
    assert d2.action == "proceed"

    # 3. Immediately attempt to re-open BTC -> MUST be BLOCKED
    d3 = guardrails.observe_and_check(
        tool="trading",
        args={"action": "place_order", "symbol": "BTC", "side": "buy"},
        result="Attempting re-entry",
        is_error=False,
    )
    assert d3.action == "block"
    assert d3.should_block is True
    assert "RE-ENTRY BLOCKED" in d3.synthetic_message

    # 4. Opening a DIFFERENT symbol (ETH) is NOT blocked
    d4 = guardrails.observe_and_check(
        tool="trading",
        args={"action": "place_order", "symbol": "ETH", "side": "buy"},
        result="ETH submitted",
        is_error=False,
    )
    assert d4.action == "proceed"


# ─── 2. Hourly Entry Limits Tests ─────────────────────────────────────────────

def test_hourly_entry_limit_blocking():
    guardrails = ToolCallGuardrails(
        ToolCallGuardrailConfig(
            hourly_entry_limit=2,  # Maximum 2 entries per hour
        )
    )

    # Entry 1: BTC -> allowed
    d1 = guardrails.observe_and_check(
        tool="trading",
        args={"action": "place_order", "symbol": "BTC", "side": "buy"},
        result="BTC ok",
        is_error=False,
    )
    assert d1.action == "proceed"

    # Entry 2: ETH -> allowed
    d2 = guardrails.observe_and_check(
        tool="trading",
        args={"action": "place_order", "symbol": "ETH", "side": "buy"},
        result="ETH ok",
        is_error=False,
    )
    assert d2.action == "proceed"

    # Entry 3: SOL -> MUST BE BLOCKED (exceeds hourly limit of 2)
    d3 = guardrails.observe_and_check(
        tool="trading",
        args={"action": "place_order", "symbol": "SOL", "side": "buy"},
        result="SOL attempt",
        is_error=False,
    )
    assert d3.action == "block"
    assert d3.should_block is True
    assert "THROTTLE BLOCKED" in d3.synthetic_message


# ─── 3. Prompt Builder Anti-Overtrading Discipline Tests ───────────────────────

def test_prompt_builder_contains_anti_overtrading_discipline():
    builder = PromptBuilder(variant="default")
    assert "Anti-Overtrading" in DEFAULT_SYSTEM_PROMPT
    assert "2-4 笔高质量交易" in DEFAULT_SYSTEM_PROMPT
    assert "45-90 分钟" in DEFAULT_SYSTEM_PROMPT
