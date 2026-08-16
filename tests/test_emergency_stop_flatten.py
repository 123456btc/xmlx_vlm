# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for emergency stop, position flattening, and reduce-only kill-switch exemption.
"""

from decimal import Decimal
from unittest.mock import patch, AsyncMock
import pytest

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderType, PositionSide
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.core.position import Position
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.oms.config.settings import get_settings
from xmlx_vlm.ai_trader.oms.exceptions import CircuitTrippedError


@pytest.mark.asyncio
async def test_emergency_stop_flattens_all_positions():
    settings = get_settings()
    engine = OMSEngine(settings=settings)

    # Setup 2 active positions: BTC long, ETH short
    pos_btc = Position(
        symbol="BTC/USDC",
        side=PositionSide.LONG,
        qty=Decimal("0.5"),
        avg_entry_price=Decimal("60000"),
    )
    pos_eth = Position(
        symbol="ETH/USDC",
        side=PositionSide.SHORT,
        qty=Decimal("4.0"),
        avg_entry_price=Decimal("3000"),
    )
    engine.portfolio.sync_positions({"BTC/USDC": pos_btc, "ETH/USDC": pos_eth})

    submitted_orders = []

    async def fake_submit(order, mark_price=None, oracle_price=None):
        submitted_orders.append(order)
        return {"status": "ok", "order": order.to_dict()}

    with patch.object(engine, "sync", new_callable=AsyncMock), \
         patch.object(engine, "submit_order", side_effect=fake_submit):
        
        result = await engine.emergency_stop(flatten=True)
        assert result["status"] == "killed"
        assert len(result["flatten_results"]) == 2
        assert len(submitted_orders) == 2

        # Verify BTC close order (was Long -> now Sell, reduce_only=True)
        btc_order = next(o for o in submitted_orders if o.symbol == "BTC/USDC")
        assert btc_order.side == OrderSide.SELL
        assert btc_order.qty == Decimal("0.5")
        assert btc_order.reduce_only is True

        # Verify ETH close order (was Short -> now Buy, reduce_only=True)
        eth_order = next(o for o in submitted_orders if o.symbol == "ETH/USDC")
        assert eth_order.side == OrderSide.BUY
        assert eth_order.qty == Decimal("4.0")
        assert eth_order.reduce_only is True

        # Verify kill switch is locked after stop
        assert engine.kill_switch.is_locked is True


@pytest.mark.asyncio
async def test_reduce_only_orders_exempt_from_kill_switch_lock():
    settings = get_settings()
    engine = OMSEngine(settings=settings)

    # Manually lock kill switch
    engine.kill_switch.trigger("test", "test lock")
    assert engine.kill_switch.is_locked is True

    # 1. New opening order (reduce_only=False) must be REJECTED
    new_order = engine.create_order(
        symbol="SOL/USDC",
        side="buy",
        qty=1.0,
        order_type="market",
        price=150.0,
        reduce_only=False,
    )
    with pytest.raises(CircuitTrippedError):
        await engine.submit_order(new_order, mark_price=150.0)

    # 2. Risk reducing / close order (reduce_only=True) must be ALLOWED through
    close_order = engine.create_order(
        symbol="SOL/USDC",
        side="sell",
        qty=1.0,
        order_type="market",
        price=150.0,
        reduce_only=True,
    )
    with patch.object(engine._adapter, "submit", new_callable=AsyncMock) as mock_adapter_submit:
        mock_adapter_submit.return_value = type("Ack", (), {"success": True, "order_id": "123", "message": "ok", "raw": {}})()
        res = await engine.submit_order(close_order, mark_price=150.0)
        assert res is not None


@pytest.mark.asyncio
async def test_reduce_only_exempt_from_all_risk_rules():
    """Verify Margin, PositionLimit, DailyLoss, RateLimit rules allow reduce_only closing orders."""
    from xmlx_vlm.ai_trader.oms.risk.rules.margin_rule import MarginRule
    from xmlx_vlm.ai_trader.oms.risk.rules.position_limit_rule import PositionLimitRule
    from xmlx_vlm.ai_trader.oms.risk.rules.daily_loss_rule import DailyLossRule
    from xmlx_vlm.ai_trader.oms.risk.rules.rate_limit_rule import RateLimitRule
    from xmlx_vlm.ai_trader.oms.interfaces.risk_engine import RiskContext
    from xmlx_vlm.ai_trader.oms.core.portfolio import Portfolio
    from xmlx_vlm.ai_trader.oms.constants import RiskDecisionType

    portfolio = Portfolio()
    context = RiskContext(
        portfolio=portfolio,
        mark_price=Decimal("100"),
        account_equity=Decimal("1000"),
    )

    close_order = Order(
        symbol="BTC/USDC",
        side=OrderSide.SELL,
        order_type="market",
        qty=Decimal("1.0"),
        price=Decimal("100"),
        reduce_only=True,
    )

    # 1. MarginRule
    margin_rule = MarginRule(min_available_margin_pct=Decimal("50.0"))
    decision = margin_rule.pre_trade(close_order, context)
    assert decision.decision == RiskDecisionType.PASS

    # 2. PositionLimitRule
    pos_rule = PositionLimitRule(max_single_position_pct=Decimal("10.0"))
    decision = pos_rule.pre_trade(close_order, context)
    assert decision.decision == RiskDecisionType.PASS

    # 3. DailyLossRule
    loss_rule = DailyLossRule(max_daily_loss_pct=Decimal("1.0"))
    loss_rule._daily_realized_pnl = Decimal("-500")  # Huge loss
    loss_rule._starting_equity = Decimal("1000")
    decision = loss_rule.pre_trade(close_order, context)
    assert decision.decision == RiskDecisionType.PASS

    # 4. RateLimitRule
    rate_rule = RateLimitRule(max_orders_per_second=1)
    rate_rule._second_window.append(1000)
    rate_rule._second_window.append(1001)  # Window full
    decision = rate_rule.pre_trade(close_order, context)
    assert decision.decision == RiskDecisionType.PASS

