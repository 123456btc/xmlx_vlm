"""测试纸盘撮合引擎."""

from decimal import Decimal

import pytest

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderType, TimeInForce
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.execution.paper.matcher import PaperMatcher
from xmlx_vlm.ai_trader.oms.market_data.models import OrderBook, OrderBookLevel, Quote


@pytest.fixture
def matcher():
    return PaperMatcher(default_price=Decimal("50000"))


@pytest.mark.anyio
async def test_market_buy_matches_ask(matcher):
    order = Order(symbol="BTC/USDC", side=OrderSide.BUY, qty=Decimal("1"), order_type="market")
    price, qty = await matcher.match(order)
    assert price is not None
    assert qty == Decimal("1")


@pytest.mark.anyio
async def test_limit_buy_not_crossed(matcher):
    order = Order(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        qty=Decimal("1"),
        order_type="limit",
        price=Decimal("49000"),
        time_in_force=TimeInForce.GTC,
    )
    price, qty = await matcher.match(order)
    assert qty == Decimal("0")


@pytest.mark.anyio
async def test_limit_buy_crossed():
    book = OrderBook(
        symbol="BTC/USDC",
        bids=[OrderBookLevel(Decimal("49900"), Decimal("1"))],
        asks=[OrderBookLevel(Decimal("50100"), Decimal("1"))],
    )
    matcher = PaperMatcher(market_data_provider=None)
    matcher._provider = None
    matcher._synthetic_book = lambda s: book
    matcher._synthetic_quote = lambda s: Quote(symbol=s, bid=Decimal("49900"), ask=Decimal("50100"))

    order = Order(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        qty=Decimal("0.5"),
        order_type="limit",
        price=Decimal("50200"),
        time_in_force=TimeInForce.IOC,
    )
    price, qty = await matcher.match(order)
    assert qty == Decimal("0.5")
    assert price is not None


@pytest.mark.anyio
async def test_fok_partial_cancelled():
    matcher = PaperMatcher(default_price=Decimal("50000"), synthetic_depth_qty=Decimal("0.1"))
    order = Order(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        qty=Decimal("1"),
        order_type="limit",
        price=Decimal("51000"),
        time_in_force=TimeInForce.FOK,
    )
    price, qty = await matcher.match(order)
    assert qty == Decimal("0")
