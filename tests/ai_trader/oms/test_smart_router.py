"""测试 SmartOrderRouter."""

from decimal import Decimal

import pytest

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState, OrderType, TimeInForce
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.execution.paper.adapter import PaperExecutionAdapter
from xmlx_vlm.ai_trader.oms.impact.market_impact import AlmgrenChrissImpactModel
from xmlx_vlm.ai_trader.oms.market_data.models import OrderBook, OrderBookLevel, Quote
from xmlx_vlm.ai_trader.oms.market_data.provider import StaticMarketDataProvider
from xmlx_vlm.ai_trader.oms.routing.context import RoutingContext
from xmlx_vlm.ai_trader.oms.routing.router import SmartOrderRouter


@pytest.fixture
def router():
    adapter = PaperExecutionAdapter()
    model = AlmgrenChrissImpactModel()
    return SmartOrderRouter(adapter, model, default_max_slippage_pct=Decimal("5.0"))


@pytest.mark.anyio
async def test_router_converts_market_to_limit(router):
    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("0.01"), order_type="market")
    ack = await router.submit(order, RoutingContext(mark_price=Decimal("50000"), urgency="normal"))
    assert ack.success
    assert order.order_type == OrderType.LIMIT
    assert order.time_in_force == TimeInForce.IOC
    assert order.price is not None


@pytest.mark.anyio
async def test_router_rejects_high_slippage():
    adapter = PaperExecutionAdapter()
    model = AlmgrenChrissImpactModel()
    router = SmartOrderRouter(adapter, model, default_max_slippage_pct=Decimal("0.001"))
    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("10"), order_type="market")
    ack = await router.submit(
        order,
        RoutingContext(
            mark_price=Decimal("50000"),
            spread_pct=Decimal("0.01"),
            recent_volume=Decimal("100"),
            volatility=Decimal("0.03"),
            urgency="aggressive",
        ),
    )
    assert not ack.success
    assert order.state == OrderState.REJECTED


@pytest.mark.anyio
async def test_router_uses_provider_quote():
    provider = StaticMarketDataProvider(
        quotes={
            "BTC/USDC": Quote(
                symbol="BTC/USDC", bid=Decimal("49900"), ask=Decimal("50100"), mark=Decimal("50000")
            )
        },
        books={
            "BTC/USDC": OrderBook(
                symbol="BTC/USDC",
                bids=[OrderBookLevel(Decimal("49900"), Decimal("1"))],
                asks=[OrderBookLevel(Decimal("50100"), Decimal("1"))],
            )
        },
    )
    adapter = PaperExecutionAdapter(market_data_provider=provider)
    model = AlmgrenChrissImpactModel()
    router = SmartOrderRouter(adapter, model, default_max_slippage_pct=Decimal("5.0"))

    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("0.5"), order_type="market")
    ack = await router.submit(order)
    assert ack.success
    assert order.state == OrderState.FILLED
