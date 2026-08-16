# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for TradePostMortemGenerator.
"""

from decimal import Decimal
import pytest

from xmlx_vlm.ai_trader.decision.post_mortem import TradePostMortemGenerator


def test_post_mortem_winning_trade():
    # Long BTC from 60,000 to 63,000 (+5%), Peak reached 64,000 (+6.67%)
    report = TradePostMortemGenerator.generate(
        symbol="BTC/USDC",
        side="LONG",
        entry_price=60000,
        exit_price=63000,
        qty=0.5,
        entry_time_ms=1700000000000,
        exit_time_ms=1700003600000,  # 60 mins later
        entry_reason="1H EMA golden cross with strong CVD inflow",
        exit_reason="Take-profit target hit",
        highest_price=64000,
        lowest_price=59800,
    )

    assert report.pnl_usd == Decimal("1500.0")
    assert report.return_pct == Decimal("5.0")
    assert report.holding_duration_min == 60.0
    assert report.mfe_pct > 6.0
    assert report.mae_pct < 1.0
    assert report.category == "SOLID_TREND_CAPTURE"
    assert "### 📊 交易复盘: BTC/USDC" in report.to_markdown()


def test_post_mortem_winner_turned_loser_warning():
    # Long ETH entered @ 3000, surged to 3200 (+6.67%), but exited @ 2900 (-3.33%)
    report = TradePostMortemGenerator.generate(
        symbol="ETH/USDC",
        side="LONG",
        entry_price=3000,
        exit_price=2900,
        qty=2.0,
        entry_time_ms=1700000000000,
        exit_time_ms=1700007200000,  # 120 mins later
        entry_reason="Breakout trade",
        exit_reason="Stop loss hit",
        highest_price=3200,
        lowest_price=2880,
    )

    assert report.pnl_usd == Decimal("-200.0")
    assert report.return_pct < Decimal("0")
    assert report.mfe_pct > 6.0
    assert report.category == "WINNER_TURNED_LOSER"
    assert "未及时拉升保本损" in report.lessons
