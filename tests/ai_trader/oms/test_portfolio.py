"""测试仓位簿."""

from decimal import Decimal

from xmlx_vlm.ai_trader.oms.constants import OrderSide, PositionSide
from xmlx_vlm.ai_trader.oms.core.portfolio import Portfolio
from xmlx_vlm.ai_trader.oms.core.trade import Trade


def test_update_with_trade_open_long():
    portfolio = Portfolio()
    trade = Trade(
        trade_id="t1",
        order_id="o1",
        client_order_id="c1",
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        qty=Decimal("0.1"),
        price=Decimal("50000"),
    )
    portfolio.update_with_trade(trade)
    pos = portfolio.get_position("BTC/USDC")
    assert pos is not None
    assert pos.side == PositionSide.LONG
    assert pos.qty == Decimal("0.1")
    assert pos.avg_entry_price == Decimal("50000")


def test_update_with_trade_close_partial():
    portfolio = Portfolio()
    portfolio.update_with_trade(
        Trade(
            trade_id="t1",
            order_id="o1",
            client_order_id="c1",
            symbol="BTC/USDC",
            side=OrderSide.BUY,
            qty=Decimal("0.1"),
            price=Decimal("50000"),
        )
    )
    portfolio.update_with_trade(
        Trade(
            trade_id="t2",
            order_id="o2",
            client_order_id="c2",
            symbol="BTC/USDC",
            side=OrderSide.SELL,
            qty=Decimal("0.04"),
            price=Decimal("55000"),
        )
    )
    pos = portfolio.get_position("BTC/USDC")
    assert pos.qty == Decimal("0.06")
    assert pos.realized_pnl == Decimal("200")


def test_gross_exposure():
    portfolio = Portfolio()
    portfolio.update_with_trade(
        Trade(
            trade_id="t1",
            order_id="o1",
            client_order_id="c1",
            symbol="BTC/USDC",
            side=OrderSide.BUY,
            qty=Decimal("0.1"),
            price=Decimal("50000"),
        )
    )
    portfolio.update_mark_prices({"BTC/USDC": Decimal("51000")})
    assert portfolio.gross_exposure() == Decimal("5100")
