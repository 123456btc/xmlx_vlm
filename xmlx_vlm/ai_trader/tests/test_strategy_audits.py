import pytest
from fastapi.testclient import TestClient
from xmlx_vlm.ai_trader.web_server import app
from xmlx_vlm.ai_trader.store.sqlite_store import SQLiteDecisionStore
from xmlx_vlm.ai_trader.decision.decision import Decision, FullDecision
from datetime import datetime, timezone
from decimal import Decimal

def test_strategy_decisions_endpoint(tmp_path, monkeypatch):
    # Setup temporary database in temp path
    db_file = tmp_path / "ai_trader.db"
    monkeypatch.setattr("xmlx_vlm.ai_trader.web_server.LOGS_DIR", tmp_path)
    
    # Manually insert 8 test decisions to verify endpoint works
    store = SQLiteDecisionStore(db_file)
    base_time = datetime.now(timezone.utc).timestamp()
    for i in range(8):
        d = FullDecision(
            trader_id="trend_follow_btc",
            cycle_number=100 + i,
            # Increment timestamp by i seconds so they sort deterministically
            timestamp=datetime.fromtimestamp(base_time + i, tz=timezone.utc),
            latency_ms=100,
            system_prompt="system prompt",
            user_prompt="user prompt",
            cot_trace="cot trace",
            decisions=[
                Decision(
                    action="hold",
                    symbol="BTC/USDC",
                    position_size_usd=Decimal("0"),
                    leverage=3,
                    price=Decimal("67000"),
                    confidence=90,
                    reasoning="reasoning"
                )
            ],
            raw_response="{}"
        )
        store.save_decision(d)
    
    # Verify records were inserted
    records = store.list_decisions(trader_id="trend_follow_btc")
    assert len(records) == 8
    assert records[0].cycle_number == 107  # LIFO order
    store.close()
    
    # Test FastAPI Endpoint
    client = TestClient(app)
    response = client.get("/api/strategy/decisions?trader_id=trend_follow_btc")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 8
    assert data[0]["trader_id"] == "trend_follow_btc"
    assert "cot_trace" in data[0]

    # Test /api/strategy/list endpoint
    list_resp = client.get("/api/strategy/list")
    assert list_resp.status_code == 200
    strat_list = list_resp.json()
    assert len(strat_list) == 1
    assert strat_list[0]["id"] == "trend_follow_btc"
    assert strat_list[0]["count"] == 8

    # Test default trader_id resolution
    default_resp = client.get("/api/strategy/decisions")
    assert default_resp.status_code == 200
    default_data = default_resp.json()
    assert len(default_data) == 8
    assert default_data[0]["trader_id"] == "trend_follow_btc"
