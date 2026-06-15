"""测试风控规则."""

from decimal import Decimal

from xmlx_vlm.ai_trader.oms.constants import OrderSide, RiskDecisionType
from xmlx_vlm.ai_trader.oms.core.account import AccountSnapshot
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.core.portfolio import Portfolio
from xmlx_vlm.ai_trader.oms.interfaces.risk_engine import RiskContext
from xmlx_vlm.ai_trader.oms.risk.rules import (
    DailyLossRule,
    MarginRule,
    OrderSizeRule,
    PositionLimitRule,
    PriceDeviationRule,
    RateLimitRule,
)


def make_context(equity: Decimal = Decimal("100000")) -> RiskContext:
    portfolio = Portfolio()
    portfolio.sync_account(AccountSnapshot(equity=equity, available_margin=equity))
    return RiskContext(portfolio=portfolio)


def test_order_size_rule_pass():
    rule = OrderSizeRule(max_single_order_notional=Decimal("10000"))
    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("0.1"), price=Decimal("50000"))
    decision = rule.pre_trade(order, make_context())
    assert decision.passed


def test_order_size_rule_reject():
    rule = OrderSizeRule(max_single_order_notional=Decimal("1000"))
    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("0.1"), price=Decimal("50000"))
    decision = rule.pre_trade(order, make_context())
    assert decision.rejected


def test_position_limit_rule_reject():
    rule = PositionLimitRule(
        max_single_position_pct=Decimal("10"),
        max_total_position_pct=Decimal("50"),
    )
    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("1"), price=Decimal("50000"))
    decision = rule.pre_trade(order, make_context(equity=Decimal("100000")))
    assert decision.rejected


def test_price_deviation_rule_reject():
    rule = PriceDeviationRule(max_price_deviation_pct=Decimal("1"))
    order = Order(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        order_type="limit",
        qty=Decimal("0.1"),
        price=Decimal("60000"),
    )
    ctx = make_context()
    ctx.mark_price = Decimal("50000")
    decision = rule.pre_trade(order, ctx)
    assert decision.rejected


def test_rate_limit_rule():
    rule = RateLimitRule(max_orders_per_second=1)
    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("0.1"))
    ctx = make_context()
    assert rule.pre_trade(order, ctx).passed
    assert rule.pre_trade(order, ctx).rejected


def test_margin_rule_reject():
    rule = MarginRule(min_available_margin_pct=Decimal("50"))
    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("1"), price=Decimal("60000"))
    ctx = make_context(equity=Decimal("100000"))
    decision = rule.pre_trade(order, ctx)
    assert decision.rejected
