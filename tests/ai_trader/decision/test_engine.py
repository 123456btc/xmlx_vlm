"""测试 DecisionEngine."""

import asyncio
from decimal import Decimal
from typing import List

import pytest

from xmlx_vlm.ai_trader.decision.decision import Decision, FullDecision
from xmlx_vlm.ai_trader.decision.engine import DecisionEngine, DecisionEngineConfig, LLMClient
from xmlx_vlm.ai_trader.oms.config.settings import OMSSettings
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.store.base import DecisionStore, EquitySnapshot


class FakeMarketData:
    def get_summary_object(self, symbol: str):
        return None


class FakeLLMClient:
    def __init__(self, response: str):
        self.response = response

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        return self.response


class FakeStore(DecisionStore):
    def __init__(self):
        self.decisions: List[FullDecision] = []
        self.snapshots: List[EquitySnapshot] = []

    def save_decision(self, record: FullDecision) -> None:
        self.decisions.append(record)

    def save_equity_snapshot(self, snapshot: EquitySnapshot) -> None:
        self.snapshots.append(snapshot)

    def list_decisions(self, trader_id: str, limit: int = 100, offset: int = 0) -> List[FullDecision]:
        return [d for d in self.decisions if d.trader_id == trader_id]

    def list_equity_snapshots(
        self, trader_id: str, limit: int = 1000, offset: int = 0
    ) -> List[EquitySnapshot]:
        return [s for s in self.snapshots if s.trader_id == trader_id]

    def get_latest_equity_snapshot(self, trader_id: str):
        rows = self.list_equity_snapshots(trader_id, limit=1)
        return rows[0] if rows else None

    def close(self) -> None:
        pass


@pytest.fixture
def paper_engine(tmp_path):
    settings = OMSSettings(exchange="paper", live_enabled=False, audit_db_path=tmp_path / "audit.db")
    return OMSEngine(settings=settings)


@pytest.mark.anyio
async def test_decision_engine_parses_llm_response(paper_engine):
    response = '[{"action": "wait", "symbol": "BTC/USDC", "confidence": 0, "reasoning": "calm"}]'
    store = FakeStore()
    config = DecisionEngineConfig(trader_id="t1", scan_interval_seconds=10, min_confidence=0)
    engine = DecisionEngine(
        oms=paper_engine,
        config=config,
        store=store,
        llm_client=FakeLLMClient(response),
        market_data=FakeMarketData(),
    )
    full = await engine.run_cycle()
    assert full.trader_id == "t1"
    assert full.cycle_number == 1
    assert len(full.decisions) == 1
    assert full.decisions[0].action == "wait"
    assert len(store.decisions) == 1
    assert len(store.snapshots) == 1
    paper_engine.close()


@pytest.mark.anyio
async def test_decision_engine_filters_low_confidence(paper_engine):
    response = '[{"action": "open_long", "symbol": "BTC/USDC", "confidence": 30, "reasoning": "weak"}]'
    store = FakeStore()
    config = DecisionEngineConfig(trader_id="t1", scan_interval_seconds=10, min_confidence=60)
    engine = DecisionEngine(
        oms=paper_engine,
        config=config,
        store=store,
        llm_client=FakeLLMClient(response),
        market_data=FakeMarketData(),
    )
    full = await engine.run_cycle()
    assert len(full.decisions) == 0
    paper_engine.close()


@pytest.mark.anyio
async def test_decision_engine_start_stop(paper_engine):
    response = '[{"action": "wait", "symbol": "BTC/USDC", "confidence": 0}]'
    store = FakeStore()
    config = DecisionEngineConfig(trader_id="t1", scan_interval_seconds=1)
    engine = DecisionEngine(
        oms=paper_engine,
        config=config,
        store=store,
        llm_client=FakeLLMClient(response),
        market_data=FakeMarketData(),
    )
    await engine.start()
    await asyncio.sleep(0.2)
    assert engine.is_running
    await engine.stop()
    assert not engine.is_running
    paper_engine.close()


@pytest.mark.anyio
async def test_decision_engine_partial_close(paper_engine):
    from xmlx_vlm.ai_trader.oms.core.position import Position
    from xmlx_vlm.ai_trader.oms.constants import PositionSide

    pos = Position(
        symbol="BTC/USDC",
        side=PositionSide.LONG,
        qty=Decimal("0.1"),
        avg_entry_price=Decimal("50000"),
    )
    paper_engine._adapter._positions["BTC/USDC"] = pos

    response = '[{"action": "close_long", "symbol": "BTC/USDC", "position_size_usd": 2000, "confidence": 100, "reasoning": "partial close"}]'

    class CustomFakeMarketData:
        class FakeSummary:
            mark_price = Decimal("50000")
            oracle_price = Decimal("50000")
            basis_pct = Decimal("0.0")
            change_24h_pct = Decimal("0.0")
            volume_24h = Decimal("100000")
            spread = Decimal("0.1")
            atr14 = Decimal("100")
            rsi14 = Decimal("50")
            oi_change_1h_pct = Decimal("0.0")
            oi_change_24h_pct = Decimal("0.0")
            cvd_1h = Decimal("0.0")
            cvd_4h = Decimal("0.0")
        def get_summary_object(self, symbol: str):
            return self.FakeSummary()

    store = FakeStore()
    config = DecisionEngineConfig(trader_id="t1", scan_interval_seconds=10, min_confidence=0)
    engine = DecisionEngine(
        oms=paper_engine,
        config=config,
        store=store,
        llm_client=FakeLLMClient(response),
        market_data=CustomFakeMarketData(),
    )

    await engine.run_cycle()
    orders = paper_engine.list_orders()
    assert len(orders) >= 1
    sell_order = [o for o in orders if o.side == "sell"][0]
    assert sell_order.qty == Decimal("0.04")
    paper_engine.close()
