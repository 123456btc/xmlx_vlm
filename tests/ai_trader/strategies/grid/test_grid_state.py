"""测试 GridState."""

from decimal import Decimal

from xmlx_vlm.ai_trader.strategies.grid.grid_state import GridState


def test_grid_state_builds_levels():
    state = GridState(
        symbol="BTC/USDC",
        upper_price=Decimal("70000"),
        lower_price=Decimal("60000"),
        grid_count=5,
        total_investment=Decimal("1000"),
        max_drawdown_pct=Decimal("5"),
        daily_loss_limit_pct=Decimal("2"),
    )
    assert len(state.levels) == 6
    assert state.levels[0].price == Decimal("60000")
    assert state.levels[-1].price == Decimal("70000")


def test_grid_state_check_breakout():
    state = GridState(
        symbol="BTC/USDC",
        upper_price=Decimal("70000"),
        lower_price=Decimal("60000"),
        grid_count=5,
        total_investment=Decimal("1000"),
        max_drawdown_pct=Decimal("5"),
        daily_loss_limit_pct=Decimal("2"),
    )
    assert state.check_breakout(Decimal("75000")) == "upper"
    assert state.check_breakout(Decimal("55000")) == "lower"
    assert state.check_breakout(Decimal("65000")) is None


def test_grid_state_max_drawdown():
    state = GridState(
        symbol="BTC/USDC",
        upper_price=Decimal("70000"),
        lower_price=Decimal("60000"),
        grid_count=5,
        total_investment=Decimal("1000"),
        max_drawdown_pct=Decimal("5"),
        daily_loss_limit_pct=Decimal("2"),
    )
    assert not state.check_max_drawdown(Decimal("10000"))
    assert state.check_max_drawdown(Decimal("9400"))


def test_grid_state_daily_loss_limit():
    state = GridState(
        symbol="BTC/USDC",
        upper_price=Decimal("70000"),
        lower_price=Decimal("60000"),
        grid_count=5,
        total_investment=Decimal("1000"),
        max_drawdown_pct=Decimal("5"),
        daily_loss_limit_pct=Decimal("2"),
    )
    state.update_daily_pnl(Decimal("-25"))
    assert state.check_daily_loss_limit()
