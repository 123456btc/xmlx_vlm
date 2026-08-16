# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for SmartOrderRouter (SOR).
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock
import pytest

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState, OrderType
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.interfaces.execution_adapter import OrderAck
from xmlx_vlm.ai_trader.oms.execution.sor import SmartOrderRouter


@pytest.mark.asyncio
async def test_sor_direct_market_when_disabled():
    sor = SmartOrderRouter()
    adapter = MagicMock()
    adapter.submit = AsyncMock(return_value=OrderAck(success=True, order_id="123", message="submitted"))

    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("0.1"), order_type=OrderType.MARKET)
    ack = await sor.route_order(adapter, order, mark_price=Decimal("60000"), maker_first=False)

    assert ack.success is True
    assert ack.order_id == "123"
    adapter.submit.assert_called_once_with(order)


@pytest.mark.asyncio
async def test_sor_maker_first_immediate_fill():
    sor = SmartOrderRouter()
    adapter = MagicMock()

    async def fake_submit(o: Order):
        o.order_id = "maker-oid-1"
        o.transition_to(OrderState.FILLED)
        return OrderAck(success=True, order_id="maker-oid-1", message="filled")

    adapter.submit = AsyncMock(side_effect=fake_submit)

    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("0.1"), order_type=OrderType.MARKET)
    ack = await sor.route_order(adapter, order, mark_price=Decimal("60000"), maker_first=True)

    assert ack.success is True
    assert order.state == OrderState.FILLED
    assert order.order_id == "maker-oid-1"
