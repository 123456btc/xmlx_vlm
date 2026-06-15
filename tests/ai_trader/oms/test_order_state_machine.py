"""测试订单状态机."""

from decimal import Decimal

import pytest

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState
from xmlx_vlm.ai_trader.oms.core.order import Fill, Order
from xmlx_vlm.ai_trader.oms.exceptions import OrderStateError


def test_order_creation():
    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("0.1"))
    assert order.state == OrderState.DRAFT
    assert order.remaining_qty == Decimal("0.1")


def test_state_transition():
    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("0.1"))
    order.transition_to(OrderState.PRE_TRADE_OK)
    order.transition_to(OrderState.SUBMITTED)
    order.transition_to(OrderState.ACKNOWLEDGED)
    assert order.state == OrderState.ACKNOWLEDGED


def test_invalid_transition_raises():
    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("0.1"))
    order.transition_to(OrderState.REJECTED)
    with pytest.raises(OrderStateError):
        order.transition_to(OrderState.SUBMITTED)


def test_apply_fill_full():
    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("0.1"))
    order.transition_to(OrderState.PRE_TRADE_OK)
    order.transition_to(OrderState.SUBMITTED)
    order.transition_to(OrderState.ACKNOWLEDGED)
    fill = Fill(
        fill_id="f1",
        order_id="o1",
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        qty=Decimal("0.1"),
        price=Decimal("50000"),
    )
    order.apply_fill(fill)
    assert order.state == OrderState.FILLED
    assert order.filled_qty == Decimal("0.1")
    assert order.avg_fill_price == Decimal("50000")


def test_apply_fill_partial():
    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("0.1"))
    order.transition_to(OrderState.PRE_TRADE_OK)
    order.transition_to(OrderState.SUBMITTED)
    order.transition_to(OrderState.ACKNOWLEDGED)
    fill = Fill(
        fill_id="f1",
        order_id="o1",
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        qty=Decimal("0.04"),
        price=Decimal("50000"),
    )
    order.apply_fill(fill)
    assert order.state == OrderState.PARTIAL_FILLED
    assert order.remaining_qty == Decimal("0.06")
