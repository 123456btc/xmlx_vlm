# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for Chandelier Dynamic Trailing Stop and Breakeven Manager.
"""

from decimal import Decimal
import pytest

from xmlx_vlm.ai_trader.oms.constants import PositionSide
from xmlx_vlm.ai_trader.oms.risk.trailing_manager import TrailingStopManager


def test_long_trailing_and_breakeven():
    manager = TrailingStopManager(default_trailing_mult=3.0, breakeven_trigger_r=1.0)

    # Buy BTC @ $60,000, initial stop @ $58,000, ATR = $1,000, TP = $66,000
    state = manager.register_position(
        symbol="BTC/USDC",
        side=PositionSide.LONG,
        entry_price=60000,
        initial_stop_loss=58000,
        take_profit=66000,
        atr=1000,
    )

    assert state.highest_price == Decimal("60000")
    assert state.current_stop_loss == Decimal("58000")
    assert state.breakeven_triggered is False

    # 1. Price rises to $60,500 (less than 1R=1000) -> hold, no BE yet
    sig1 = manager.update_price("BTC/USDC", 60500)
    assert sig1.should_close is False
    assert state.highest_price == Decimal("60500")
    assert state.breakeven_triggered is False

    # 2. Price rises to $61,200 (>= 1R=1000) -> Auto Break-Even activated! Stop raised to $60,000
    sig2 = manager.update_price("BTC/USDC", 61200)
    assert sig2.should_close is False
    assert state.breakeven_triggered is True
    assert state.current_stop_loss >= Decimal("60000")

    # 3. Price surges to $65,000 -> Chandelier trailing: $65,000 - 3*1000 = $62,000!
    sig3 = manager.update_price("BTC/USDC", 65000)
    assert sig3.should_close is False
    assert state.highest_price == Decimal("65000")
    assert state.current_stop_loss == Decimal("62000")
    assert state.chandelier_active is True

    # 4. Price pulls back to $61,800 (< $62,000 trailing stop) -> triggers close!
    sig4 = manager.update_price("BTC/USDC", 61800)
    assert sig4.should_close is True
    assert sig4.trigger_type == "chandelier_stop"
    assert "Stop Loss hit" in sig4.reason


def test_short_trailing_and_tp():
    manager = TrailingStopManager(default_trailing_mult=3.0, breakeven_trigger_r=1.0)

    # Short ETH @ $3,000, stop @ $3,200, TP @ $2,700, ATR = $50
    state = manager.register_position(
        symbol="ETH/USDC",
        side=PositionSide.SHORT,
        entry_price=3000,
        initial_stop_loss=3200,
        take_profit=2700,
        atr=50,
    )

    # 1. Price drops to $2,940 (profit > 1R=50) -> BE activated
    sig1 = manager.update_price("ETH/USDC", 2940)
    assert sig1.should_close is False
    assert state.breakeven_triggered is True
    assert state.current_stop_loss == Decimal("3000")

    # 2. Price drops to $2,690 -> hits TP
    sig2 = manager.update_price("ETH/USDC", 2690)
    assert sig2.should_close is True
    assert sig2.trigger_type == "take_profit"
