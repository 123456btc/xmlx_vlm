# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for AI Trader integration with agent core primitives.
"""

import os
import tempfile
import pytest
from decimal import Decimal

from xmlx_vlm.ai_trader.decision.engine import DecisionEngine, DecisionEngineConfig
from xmlx_vlm.ai_trader.decision.decision import Decision
from xmlx_vlm.ai_trader.runtime.kanban_bridge import TradingKanbanBridge
from xmlx_vlm.kanban import KanbanBoard
from xmlx_vlm.agent_core import (
    ContextCompressor,
    SUMMARY_PREFIX,
    ThinkScrubber,
    ToolCallGuardrailConfig,
    ToolCallGuardrails,
)


# ─── 1. DecisionEngine ThinkScrubber Integration Tests ────────────────────────

def test_decision_engine_parses_with_think_scrubber():
    # Simulate an RL / CoT model (e.g. DeepSeek-R1, Qwen-Thinking)
    raw_llm_response = """
    <think>
    Current BTC 1h RSI is 28 (oversold).
    CVD shows bullish divergence on 15m.
    Funding rate is negative (-0.01%), indicating shorts are paying longs.
    Risk/reward ratio is 3.5:1.
    I should open a long position with $500 size and 2% stop loss.
    </think>
    [
      {
        "symbol": "BTC",
        "action": "open_long",
        "position_size_usd": 500.0,
        "price": 65000.0,
        "stop_loss": 63700.0,
        "take_profit": 69500.0,
        "confidence": 85,
        "reasoning": "Oversold RSI with bullish CVD divergence"
      }
    ]
    """

    class DummyOMS:
        pass

    class DummyStore:
        pass

    class DummyLLM:
        pass

    config = DecisionEngineConfig(trader_id="test_trader")
    engine = DecisionEngine(
        oms=DummyOMS(),
        config=config,
        store=DummyStore(),
        llm_client=DummyLLM(),
    )

    # 1. Test parsing decision array after scrubbing reasoning
    decisions = engine._parse_decisions(raw_llm_response)
    assert len(decisions) == 1
    d = decisions[0]
    assert d.symbol == "BTC/USDC"
    assert d.action == "open_long"
    assert d.side == "buy"
    assert d.position_size_usd == Decimal("500.0")
    assert d.confidence == 85

    # 2. Test CoT extraction
    cot = engine._extract_cot(raw_llm_response)
    assert "Current BTC 1h RSI is 28" in cot
    assert "Funding rate is negative" in cot


# ─── 2. Trading Tool Guardrails Tests ─────────────────────────────────────────

def test_trading_guardrails_prevent_duplicate_order_failures():
    guardrails = ToolCallGuardrails(
        ToolCallGuardrailConfig(
            warnings_enabled=True,
            hard_stop_enabled=True,
            exact_failure_warn_after=2,
            exact_failure_block_after=3,
        )
    )

    # Attempt 1: Order fails (e.g. margin insufficient)
    d1 = guardrails.observe_and_check(
        tool="trading",
        args={"action": "place_order", "symbol": "ETH", "side": "buy", "qty": 10.0},
        result="Error: Insufficient margin in wallet",
        is_error=True,
    )
    assert d1.action == "proceed"

    # Attempt 2: Same exact order fails again -> triggers warn
    d2 = guardrails.observe_and_check(
        tool="trading",
        args={"action": "place_order", "symbol": "ETH", "side": "buy", "qty": 10.0},
        result="Error: Insufficient margin in wallet",
        is_error=True,
    )
    assert d2.action == "warn"
    assert "WARNING" in d2.synthetic_message

    # Attempt 3: Same exact order fails again -> triggers BLOCK
    d3 = guardrails.observe_and_check(
        tool="trading",
        args={"action": "place_order", "symbol": "ETH", "side": "buy", "qty": 10.0},
        result="Error: Insufficient margin in wallet",
        is_error=True,
    )
    assert d3.action == "block"
    assert d3.should_block is True
    assert "ACTION BLOCKED" in d3.synthetic_message


# ─── 3. Long-run Trading Context Compression with Anti-Hijack ─────────────────

def test_trading_context_compression_with_anti_hijack():
    compressor = ContextCompressor(
        max_context_tokens=300,
        compression_threshold=0.5,
        tail_token_budget=80,
    )

    messages = [
        {"role": "system", "content": "You are a quant trading assistant."},
        # Older turns with bulky L2 orderbook results
        {"role": "user", "content": "Fetch market data for BTC"},
        {"role": "assistant", "content": "Calling market_data"},
        {"role": "user", "content": f"Result: {{'bids': {[[65000, 1.2]] * 50}, 'asks': {[[65001, 2.0]] * 50}}}"},
        {"role": "assistant", "content": "BTC looks bullish based on 2h ago data"},
        # Recent active user instruction: emergency stop!
        {"role": "user", "content": "Emergency stop! Close all positions immediately."},
    ]

    compacted, was_compressed = compressor.compress(messages, force=True)
    assert was_compressed is True
    assert len(compacted) < len(messages)

    # 1. System prompt preserved
    assert compacted[0]["role"] == "system"

    # 2. Anti-hijack declaration present in summary
    summary_msg = compacted[1]
    assert summary_msg["role"] == "user"
    assert SUMMARY_PREFIX in summary_msg["content"]

    # 3. Latest emergency stop instruction is preserved in tail
    assert compacted[-1]["content"] == "Emergency stop! Close all positions immediately."


# ─── 4. Trading Kanban Multi-Agent Pipeline Tests ─────────────────────────────

def test_trading_kanban_pipeline_flow():
    fd, db_path = tempfile.mkstemp(prefix="trading_kanban_", suffix=".db")
    os.close(fd)

    try:
        board = KanbanBoard(db_path=db_path)
        bridge = TradingKanbanBridge(board=board)

        # 1. Submit market alert
        alert_task = bridge.submit_alert(
            alert_type="volatility_expansion",
            symbol="BTC",
            details={"5m_atr_expansion": 2.8, "cvd_delta": "+450k"},
            priority=5,
        )
        assert alert_task.assignee_profile == "scout"
        assert alert_task.status == "todo"

        # 2. Run pipeline cycle: Scout -> Analyst -> Risk -> Executor
        total_processed = bridge.run_pipeline_cycle(max_ticks=4)
        assert total_processed == 4  # All 4 stages executed
        
        # Verify all tasks are completed
        done_tasks = board.list_tasks(status="done")
        assert len(done_tasks) == 4  # All 4 pipeline stages completed
        
        # Verify stage progression in tasks
        profiles = {t.assignee_profile for t in done_tasks}
        assert profiles == {"scout", "analyst", "risk_officer", "executor"}
    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass
