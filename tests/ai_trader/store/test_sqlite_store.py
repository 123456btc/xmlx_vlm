"""测试 SQLiteDecisionStore."""

from decimal import Decimal

from xmlx_vlm.ai_trader.decision.decision import Decision, FullDecision
from xmlx_vlm.ai_trader.store.base import EquitySnapshot
from xmlx_vlm.ai_trader.store.sqlite_store import SQLiteDecisionStore


def test_save_and_list_decisions(tmp_path):
    db = SQLiteDecisionStore(tmp_path / "test.db")
    record = FullDecision(
        trader_id="t1",
        cycle_number=1,
        decisions=[Decision(action="wait", symbol="BTC/USDC", confidence=0)],
        raw_response="[]",
    )
    db.save_decision(record)
    rows = db.list_decisions("t1")
    assert len(rows) == 1
    assert rows[0].trader_id == "t1"
    assert rows[0].decisions[0].action == "wait"
    db.close()


def test_save_and_list_equity_snapshots(tmp_path):
    db = SQLiteDecisionStore(tmp_path / "test.db")
    snapshot = EquitySnapshot(
        trader_id="t1",
        timestamp_ms=1_000,
        total_equity=Decimal("10000"),
        available_margin=Decimal("8000"),
        position_count=2,
    )
    db.save_equity_snapshot(snapshot)
    latest = db.get_latest_equity_snapshot("t1")
    assert latest is not None
    assert latest.total_equity == Decimal("10000")
    assert latest.position_count == 2
    db.close()


def test_latest_equity_snapshot_returns_none_for_missing_trader(tmp_path):
    db = SQLiteDecisionStore(tmp_path / "test.db")
    assert db.get_latest_equity_snapshot("missing") is None
    db.close()
