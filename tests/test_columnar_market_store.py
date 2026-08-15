"""
Unit tests for Columnar Market Store & Point-in-Time (as_of) Time Travel.
"""

import json
import os
import tempfile
import time
import pytest

from xmlx_vlm.ai_trader.market_service.columnar_store import (
    ColumnarChunk,
    ColumnarMarketStore,
    SymbolPartition,
)
from xmlx_vlm.ai_trader.tools.market import MarketDataTool


# ─── 1. ColumnarChunk Unit Tests ──────────────────────────────────────────────

def test_columnar_chunk_append_and_slice():
    chunk = ColumnarChunk(symbol="BTC", timeframe="1m", chunk_id="test_chunk_1")
    
    # Append 10 sequential rows with 60s intervals
    base_ts = 1700000000000
    for i in range(10):
        ts = base_ts + i * 60000
        chunk.append_row(
            ts=ts,
            o=60000.0 + i * 10,
            h=60050.0 + i * 10,
            l=59950.0 + i * 10,
            c=60020.0 + i * 10,
            v=1.5 + i * 0.1,
            cvd=100.0 + i * 20,
            oi=5000.0 + i * 50,
            funding=0.0001,
            imbalance=0.15,
        )

    assert chunk.row_count == 10
    assert chunk.start_ts == base_ts
    assert chunk.end_ts == base_ts + 9 * 60000

    # Test full slice
    full_slice = chunk.slice()
    assert len(full_slice["timestamp"]) == 10
    assert full_slice["close"][0] == 60020.0
    assert full_slice["close"][-1] == 60020.0 + 90

    # Test range slice
    range_slice = chunk.slice(
        start_ts=base_ts + 2 * 60000,
        end_ts=base_ts + 5 * 60000,
    )
    assert len(range_slice["timestamp"]) == 4  # index 2, 3, 4, 5
    assert range_slice["timestamp"][0] == base_ts + 2 * 60000
    assert range_slice["timestamp"][-1] == base_ts + 5 * 60000


# ─── 2. Point-in-Time (as_of) Lookahead Bias Prevention ────────────────────────

def test_point_in_time_as_of_filtering():
    base_ts = 1700000000000
    chunk = ColumnarChunk(symbol="ETH", timeframe="5m", chunk_id="eth_chunk_1")
    
    # 5 rows representing 10:00, 10:05, 10:10, 10:15, 10:20
    for i in range(5):
        chunk.append_row(
            ts=base_ts + i * 300000,
            o=3000.0 + i,
            h=3010.0 + i,
            l=2990.0 + i,
            c=3005.0 + i,
            v=10.0,
        )

    # Query as of 10:10 (index 2)
    as_of_time = base_ts + 2 * 300000
    time_travel_slice = chunk.slice(as_of_ms=as_of_time)

    # MUST only see data up to 10:10 (3 rows: 10:00, 10:05, 10:10)
    assert len(time_travel_slice["timestamp"]) == 3
    assert time_travel_slice["timestamp"][-1] == as_of_time
    assert 3005.0 + 3 not in time_travel_slice["close"]  # 10:15 future data is strictly excluded
    assert 3005.0 + 4 not in time_travel_slice["close"]  # 10:20 future data is strictly excluded


# ─── 3. SymbolPartition Auto-Freezing & Disk Storage ──────────────────────────

def test_symbol_partition_chunking_and_storage():
    with tempfile.TemporaryDirectory() as tmp_dir:
        part = SymbolPartition(symbol="SOL", chunk_size=5, storage_dir=tmp_dir)

        base_ts = 1700000000000
        # Write 12 rows (should create 2 frozen chunks of 5 and 1 active buffer of 2)
        for i in range(12):
            part.append(
                timeframe="1m",
                ts=base_ts + i * 60000,
                o=150.0 + i,
                h=155.0 + i,
                l=149.0 + i,
                c=152.0 + i,
                v=100.0,
            )

        assert len(part._frozen_chunks.get("1m", [])) == 2
        assert part._hot_buffers.get("1m").row_count == 2

        # Verify full query merges frozen chunks and active buffer seamlessly
        res = part.query(timeframe="1m")
        assert len(res["timestamp"]) == 12
        assert res["close"][-1] == 152.0 + 11

        # Verify limit
        limited = part.query(timeframe="1m", limit=3)
        assert len(limited["timestamp"]) == 3
        assert limited["close"][-1] == 152.0 + 11


# ─── 4. Global ColumnarMarketStore & Snapshot Rollback ────────────────────────

def test_columnar_store_state_snapshots():
    store = ColumnarMarketStore(chunk_size=100)
    
    t0 = 1700000000000
    t1 = t0 + 60000
    t2 = t0 + 120000

    # Record state snapshots at t0, t1, t2
    store.append_state_snapshot("BTC", {"price": 60000, "regime": "ranging"}, ts_ms=t0)
    store.append_state_snapshot("BTC", {"price": 61000, "regime": "breakout_long"}, ts_ms=t1)
    store.append_state_snapshot("BTC", {"price": 62000, "regime": "overbought"}, ts_ms=t2)

    # Point-in-Time snapshot at t1
    snap_t1 = store.get_snapshot_as_of("BTC", as_of_ms=t1)
    assert snap_t1 is not None
    assert snap_t1["price"] == 61000
    assert snap_t1["regime"] == "breakout_long"

    # Point-in-Time snapshot at t0 + 30s (should resolve to t0 state)
    snap_mid = store.get_snapshot_as_of("BTC", as_of_ms=t0 + 30000)
    assert snap_mid is not None
    assert snap_mid["price"] == 60000

    # Latest snapshot
    snap_latest = store.get_snapshot_as_of("BTC")
    assert snap_latest["price"] == 62000


# ─── 5. MarketDataTool Integration with as_of ─────────────────────────────────

def test_market_tool_columnar_actions():
    store = ColumnarMarketStore.get_instance()
    
    base_ts = 1700000000000
    for i in range(5):
        store.append_bar(
            symbol="AVAX",
            timeframe="15m",
            ts=base_ts + i * 900000,
            o=25.0 + i,
            h=26.0 + i,
            l=24.5 + i,
            c=25.5 + i,
            v=500.0,
            cvd=50.0 * i,
        )

    tool = MarketDataTool()

    # 1. Test get_columnar_series
    raw_col = tool.run(
        action="get_columnar_series",
        symbol="AVAX",
        timeframe="15m",
    )
    data = json.loads(raw_col)
    assert data["symbol"] == "AVAX"
    assert data["row_count"] >= 5
    assert "columns" in data
    assert "timestamp" in data["columns"]

    # 2. Test get_columnar_series with as_of
    raw_as_of = tool.run(
        action="get_columnar_series",
        symbol="AVAX",
        timeframe="15m",
        as_of=base_ts + 2 * 900000,
    )
    data_as_of = json.loads(raw_as_of)
    assert data_as_of["row_count"] == 3  # index 0, 1, 2 only
