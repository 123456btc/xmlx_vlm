import pytest
import time
from pathlib import Path
from xmlx_vlm.ai_trader.market_service.models import Bar, Tick
from xmlx_vlm.ai_trader.market_service.state import SymbolState
from xmlx_vlm.ai_trader.market_service.kline_db import KlineDB

def test_kline_db_save_load(tmp_path):
    db_file = tmp_path / "test_kline.db"
    db = KlineDB(db_file)
    
    # Create some mock bars
    bar1 = Bar("BTC", "1m", 60000.0, 60500.0, 59900.0, 60200.0, 10.5, 1000000, 4.0, 6.5)
    bar2 = Bar("BTC", "1m", 60200.0, 61000.0, 60100.0, 60800.0, 15.0, 1060000, 10.0, 5.0)
    
    # Save to db
    db.save_bars([bar1, bar2])
    
    # Load from db
    loaded = db.load_bars("BTC", "1m")
    assert len(loaded) == 2
    assert loaded[0].timestamp_ms == 1000000
    assert loaded[0].open == 60000.0
    assert loaded[0].close == 60200.0
    assert loaded[0].volume == 10.5
    
    assert loaded[1].timestamp_ms == 1060000
    assert loaded[1].open == 60200.0
    
    # Test get latest timestamp
    latest = db.get_latest_timestamp("BTC", "1m")
    assert latest == 1060000

def test_symbol_state_incremental_aggregation(tmp_path):
    db_file = tmp_path / "test_kline_state.db"
    state = SymbolState("ETH")
    # Point states's kline_db to our temp file
    state._kline_db = KlineDB(db_file)
    
    # 1. Feed 1m bars and check aggregation into 5m timeframe
    base_ts = 1700000100000  # aligned to 5m (1700000100000 is divisible by 300,000)
    
    # Feed first 1m bar (0-1 minute)
    bar1 = Bar("ETH", "1m", 3000.0, 3050.0, 2990.0, 3010.0, 100.0, base_ts)
    state.update_candle(bar1)
    
    assert state._current_bar["1m"] == bar1
    assert len(state._bars["1m"]) == 0
    
    # Feed second 1m bar (1-2 minute)
    bar2 = Bar("ETH", "1m", 3010.0, 3020.0, 3000.0, 3015.0, 120.0, base_ts + 60000)
    state.update_candle(bar2)
    
    # Previous bar1 should be closed and stored
    assert len(state._bars["1m"]) == 1
    assert state._bars["1m"][0].timestamp_ms == base_ts
    assert state._current_bar["1m"] == bar2
    
    # 5m bar should be in progress
    assert state._current_bar["5m"] is not None
    assert state._current_bar["5m"].timestamp_ms == base_ts
    assert state._current_bar["5m"].open == 3000.0
    assert state._current_bar["5m"].high == 3050.0
    assert state._current_bar["5m"].low == 2990.0
    assert state._current_bar["5m"].close == 3010.0
    assert state._current_bar["5m"].volume == 100.0  # only bar1 has closed and been merged
    
    # Feed bars to cross 5m boundary
    # We feed candles for min 2, 3, 4, 5
    for i in range(2, 7):
        ts = base_ts + i * 60000
        bar = Bar("ETH", "1m", 3015.0, 3030.0, 3010.0, 3020.0, 50.0, ts)
        state.update_candle(bar)
        
    # The 5m boundary is base_ts + 300,000.
    # The candle at base_ts + 300,000 (i=5) has started, which means base_ts + 0 to base_ts + 240,000 has closed.
    # So the first 5m bar (base_ts) should have closed and moved to state._bars["5m"]!
    assert len(state._bars["5m"]) == 1
    closed_5m = state._bars["5m"][0]
    assert closed_5m.timestamp_ms == base_ts
    assert closed_5m.open == 3000.0
    assert closed_5m.high == 3050.0
    assert closed_5m.low == 2990.0
    
    # Verify it was saved to DB
    db_bars = state._kline_db.load_bars("ETH", "5m")
    assert len(db_bars) == 1
    assert db_bars[0].timestamp_ms == base_ts

def test_aggregate_bar_bypassed_when_candle_stream_active(tmp_path):
    db_file = tmp_path / "test_aggregate_bar_bypassed.db"
    state = SymbolState("SOL")
    state._kline_db = KlineDB(db_file)
    state._bars["1m"] = []
    state._current_bar["1m"] = None
    
    # Initially no candle stream
    assert not state._has_candle_stream
    
    # Trigger an update via tick
    state.update_tick(Tick("SOL", 100.0, 1000000))
    # Should update K-line because _has_candle_stream is False
    assert len(state._bars["1m"]) == 0
    assert state._current_bar["1m"] is not None
    assert state._current_bar["1m"].close == 100.0
    
    # Activate candle stream
    bar = Bar("SOL", "1m", 100.0, 105.0, 95.0, 102.0, 10.0, 1060000)
    state.update_candle(bar)
    assert state._has_candle_stream
    
    # Now send another tick
    state.update_tick(Tick("SOL", 200.0, 1120000))
    # The tick should update latest_tick and latest_quote, but NOT run K-line aggregation!
    # So the current_bar["1m"] close price should still be 102.0 (from the candle update), not 200.0
    assert state.latest_tick.price == 200.0
    assert state._current_bar["1m"].close == 102.0
