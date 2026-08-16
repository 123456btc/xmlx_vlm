# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for Dynamic Position Sizing (ATR Volatility-Parity & Kelly Sizing).
"""

from decimal import Decimal
import pytest

from xmlx_vlm.ai_trader.decision.sizing import PositionSizer, SizingRecommendation


def test_atr_sizing_standard_long():
    sizer = PositionSizer(
        default_risk_pct=0.015,   # 1.5% of equity
        max_notional_ratio=0.25,  # 25% max notional cap
        default_atr_mult=2.0,     # Stop = 2 * ATR
        reward_risk_ratio=2.0,    # TP = 4 * ATR
    )

    # Equity = $10,000, BTC Mark = $60,000, ATR = $1,500
    # Max risk = 10000 * 0.015 = $150
    # Stop distance = 2.0 * 1500 = $3,000
    # Qty = 150 / 3000 = 0.05 BTC
    # Notional = 0.05 * 60000 = $3,000 (30% of equity -> clamped to 25% = $2,500)
    rec = sizer.calculate(
        account_equity=10000,
        mark_price=60000,
        atr=1500,
        is_long=True,
    )

    assert rec.stop_distance == Decimal("3000.0")
    assert rec.suggested_stop_loss == Decimal("57000.0")
    assert rec.suggested_take_profit == Decimal("66000.0")
    assert rec.is_clamped is True
    assert rec.recommended_notional_usd == Decimal("2500.0")
    assert rec.recommended_qty == Decimal("2500.0") / Decimal("60000.0")


def test_atr_sizing_unclamped_altcoin():
    sizer = PositionSizer(
        default_risk_pct=0.01,    # 1.0% of equity
        max_notional_ratio=0.50,  # 50% max notional cap
        default_atr_mult=2.0,
    )

    # Equity = $10,000, SUI Mark = $2.0, ATR = $0.20
    # Max risk = 10000 * 0.01 = $100
    # Stop distance = 2.0 * 0.20 = $0.40
    # Qty = 100 / 0.40 = 250 SUI
    # Notional = 250 * 2.0 = $500 (5% of equity, below 50% cap -> not clamped)
    rec = sizer.calculate(
        account_equity=10000,
        mark_price=2.0,
        atr=0.20,
        is_long=False,  # Short
    )

    assert rec.stop_distance == Decimal("0.40")
    assert rec.suggested_stop_loss == Decimal("2.40")
    assert rec.suggested_take_profit == Decimal("1.20")
    assert rec.is_clamped is False
    assert rec.recommended_qty == Decimal("250.0")
    assert rec.recommended_notional_usd == Decimal("500.0")


def test_kelly_scaling():
    sizer = PositionSizer(kelly_fraction=0.25)

    # High win rate 60%, 2:1 win-loss ratio
    # p=0.6, b=2.0 -> f = (0.6*3 - 1)/2 = 0.8/2 = 0.4
    # Kelly adjustment > 1.0
    mult_high = sizer.compute_kelly_fraction(win_rate=0.60, win_loss_ratio=2.0)
    assert mult_high > Decimal("1.0")

    # Poor win rate 30%, 1:1 win-loss ratio
    # f < 0 -> returns minimum defense 0.5
    mult_low = sizer.compute_kelly_fraction(win_rate=0.30, win_loss_ratio=1.0)
    assert mult_low == Decimal("0.5")
