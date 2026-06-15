"""测试 TradingContext 构建."""

from decimal import Decimal

from xmlx_vlm.ai_trader.decision.context import TradingContext, TradingStats, RecentOrder
from xmlx_vlm.ai_trader.oms.core.account import AccountSnapshot
from xmlx_vlm.ai_trader.oms.core.position import Position
from xmlx_vlm.ai_trader.oms.constants import PositionSide


def test_trading_context_to_dict():
    account = AccountSnapshot(equity=Decimal("10000"), available_margin=Decimal("8000"))
    pos = Position(symbol="BTC/USDC", side=PositionSide.LONG, qty=Decimal("0.1"), mark_price=Decimal("70000"))
    ctx = TradingContext(
        current_time="2026-01-01T00:00:00Z",
        runtime_minutes=10,
        cycle_number=2,
        trader_id="t1",
        account=account,
        positions=[pos],
        candidate_symbols=["BTC/USDC"],
    )
    data = ctx.to_dict()
    assert data["trader_id"] == "t1"
    assert data["account"]["equity"] == "10000"
    assert len(data["positions"]) == 1


def test_trading_stats_to_dict():
    stats = TradingStats(total_trades=10, win_rate=Decimal("55.5"), total_pnl=Decimal("123.45"))
    data = stats.to_dict()
    assert data["total_trades"] == 10
    assert data["win_rate"] == "55.5"


def test_recent_order_to_dict():
    order = RecentOrder(
        symbol="BTC/USDC",
        side="buy",
        entry_price=Decimal("60000"),
        exit_price=Decimal("65000"),
        realized_pnl=Decimal("500"),
        pnl_pct=Decimal("8.33"),
        entry_time="t1",
        exit_time="t2",
        hold_duration="1h",
    )
    assert order.to_dict()["symbol"] == "BTC/USDC"
