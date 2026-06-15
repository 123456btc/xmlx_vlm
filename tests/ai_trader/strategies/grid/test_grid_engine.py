"""测试 GridEngine."""

from decimal import Decimal

import pytest

from xmlx_vlm.ai_trader.oms.config.settings import OMSSettings
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.store.base import DecisionStore, EquitySnapshot
from xmlx_vlm.ai_trader.strategies.grid.grid_engine import GridEngine, GridEngineConfig


class FakeMarketData:
    def __init__(self, mark_price: float):
        from xmlx_vlm.ai_trader.market_service.models import MarketSummary

        self.summary = MarketSummary(
            symbol="BTC/USDC",
            mark_price=mark_price,
            oracle_price=None,
            basis_pct=0.0,
            bid=mark_price - 1,
            ask=mark_price + 1,
            spread=2.0,
            high_24h=None,
            low_24h=None,
            change_24h_pct=0.0,
            volume_24h=0.0,
            atr14=None,
            atr_pct=None,
            adx14=None,
            rsi14=None,
            ema20=None,
            ema50=None,
        )

    def get_summary_object(self, symbol: str):
        return self.summary


class FakeStore(DecisionStore):
    def __init__(self):
        self.snapshots: list = []

    def save_decision(self, record):
        pass

    def save_equity_snapshot(self, snapshot: EquitySnapshot) -> None:
        self.snapshots.append(snapshot)

    def list_decisions(self, *args, **kwargs):
        return []

    def list_equity_snapshots(self, *args, **kwargs):
        return []

    def get_latest_equity_snapshot(self, *args, **kwargs):
        return None

    def close(self):
        pass


@pytest.fixture
def grid_engine(tmp_path):
    settings = OMSSettings(
        exchange="paper",
        live_enabled=False,
        risk_profile="custom",
        audit_db_path=tmp_path / "audit.db",
        max_orders_per_second=100,
        max_orders_per_minute=1000,
    )
    oms = OMSEngine(settings=settings)
    config = GridEngineConfig(
        trader_id="grid1",
        symbol="BTC/USDC",
        upper_price=Decimal("70000"),
        lower_price=Decimal("60000"),
        grid_count=5,
        total_investment=Decimal("1000"),
        scan_interval_seconds=10,
    )
    engine = GridEngine(
        oms=oms,
        config=config,
        store=FakeStore(),
        market_data=FakeMarketData(65000.0),
    )
    yield engine
    oms.close()


@pytest.mark.anyio
async def test_grid_engine_places_orders(grid_engine):
    result = await grid_engine.run_cycle()
    assert result["status"] == "ok"
    assert len(grid_engine._active_orders) > 0


@pytest.mark.anyio
async def test_grid_engine_detects_breakout(grid_engine):
    grid_engine.market_data = FakeMarketData(75000.0)
    result = await grid_engine.run_cycle()
    assert result["status"] == "breakout"
    assert grid_engine.state.is_paused


@pytest.mark.anyio
async def test_grid_engine_start_stop(grid_engine):
    await grid_engine.start()
    assert grid_engine.is_running
    await grid_engine.stop()
    assert not grid_engine.is_running
