"""Unit tests for ColumnarMarketStore ACDB binary chunk compression adapter."""

import pytest
import time
from pathlib import Path
from xmlx_vlm.ai_trader.market_service.columnar_store import (
    MAGIC_HEADER,
    ChunkCompressionAdapter,
    ColumnarChunk,
    ColumnarMarketStore,
    SymbolPartition,
)


class TestColumnarCompression:
    """Test suite for binary ACDB compression, CRC32 integrity, and storage statistics."""

    def test_chunk_compression_adapter_roundtrip(self):
        raw_payload = b"Hello, Arctic Columnar Store! Repeated patterns " * 50
        compressed = ChunkCompressionAdapter.compress(raw_payload, level=6)

        # Assert binary header
        assert compressed[:4] == MAGIC_HEADER
        assert len(compressed) < len(raw_payload)  # Significant compression

        decompressed = ChunkCompressionAdapter.decompress(compressed)
        assert decompressed == raw_payload

    def test_crc32_and_corruption_detection(self):
        raw_payload = b"Important Financial K-Line Time Series Data" * 20
        compressed = bytearray(ChunkCompressionAdapter.compress(raw_payload))

        # Corrupt one byte in the payload
        compressed[-5] ^= 0xFF

        with pytest.raises(Exception):  # Should raise CRC32 or Decompression error
            ChunkCompressionAdapter.decompress(bytes(compressed))

    def test_columnar_chunk_compressed_bytes(self):
        chunk = ColumnarChunk(
            symbol="BTC",
            timeframe="1h",
            chunk_id="BTC_1h_001",
            version=1,
        )
        # Populate 200 rows of realistic market data
        base_ts = 1700000000000
        for i in range(200):
            chunk.append_row(
                ts=base_ts + i * 3600000,
                o=60000.0 + i * 10,
                h=60050.0 + i * 10,
                l=59950.0 + i * 10,
                c=60020.0 + i * 10,
                v=150.0 + i * 2,
                cvd=50.0 + i,
                oi=12000.0 + i * 5,
                funding=0.0001,
                imbalance=0.15,
            )

        compressed_bytes = chunk.to_compressed_bytes()
        assert compressed_bytes[:4] == MAGIC_HEADER

        # Restored chunk verification
        restored = ColumnarChunk.from_compressed_bytes(compressed_bytes)
        assert restored.symbol == "BTC"
        assert restored.timeframe == "1h"
        assert restored.chunk_id == "BTC_1h_001"
        assert restored.row_count == 200
        assert restored.opens == chunk.opens
        assert restored.closes == chunk.closes
        assert restored.cvds == chunk.cvds
        assert restored.ois == chunk.ois

    def test_compression_ratio_achievement(self):
        chunk = ColumnarChunk(symbol="ETH", timeframe="1m", chunk_id="ETH_1m_001")
        # 500 rows
        for i in range(500):
            chunk.append_row(
                ts=1700000000000 + i * 60000,
                o=3000.0 + (i % 10),
                h=3005.0 + (i % 10),
                l=2995.0 + (i % 10),
                c=3002.0 + (i % 10),
                v=80.0,
                cvd=10.0,
                oi=50000.0,
                funding=0.0001,
                imbalance=0.05,
            )

        import json
        raw_json_len = len(json.dumps(chunk.to_dict()).encode("utf-8"))
        compressed_len = len(chunk.to_compressed_bytes())

        # Should achieve at least 70% space savings
        saved_pct = (raw_json_len - compressed_len) / raw_json_len * 100.0
        assert saved_pct > 70.0, f"Expected >70% compression, got {saved_pct:.2f}%"

    def test_partition_disk_flush_and_hydration(self, tmp_path):
        storage_dir = tmp_path / "acdb_storage"
        part = SymbolPartition(symbol="SOL", chunk_size=50, storage_dir=storage_dir)

        # Append 120 bars -> should trigger 2 freezes into disk (.acdb) + 20 in hot buffer
        for i in range(120):
            part.append(
                timeframe="5m",
                ts=1700000000000 + i * 300000,
                o=150.0 + i,
                h=152.0 + i,
                l=149.0 + i,
                c=151.0 + i,
                v=1000.0,
            )

        # Check disk files
        acdb_files = list((storage_dir / "SOL" / "5m").glob("*.acdb"))
        assert len(acdb_files) >= 2

        # Create a new partition instance with same storage_dir (simulating restart cold load)
        restarted_part = SymbolPartition(symbol="SOL", chunk_size=50, storage_dir=storage_dir)
        stats = restarted_part.storage_stats()
        assert stats["total_chunks"] >= 2
        assert stats["total_rows"] >= 100
        assert stats["compression_ratio_pct"] > 70.0

        # Verify Point-in-Time querying on loaded chunks
        res = restarted_part.query(timeframe="5m", limit=30)
        assert len(res["close"]) == 30

    def test_global_store_storage_stats(self, tmp_path):
        store = ColumnarMarketStore(storage_dir=tmp_path / "global_acdb", chunk_size=50)

        # Write data for BTC and ETH
        for i in range(60):
            store.append_bar("BTC", "15m", 1700000000000 + i * 900000, 60000, 60100, 59900, 60050, 100)
            store.append_bar("ETH", "15m", 1700000000000 + i * 900000, 3000, 3010, 2990, 3005, 50)

        stats = store.storage_stats()
        assert stats["total_symbols"] == 2
        assert stats["total_rows"] == 120
        assert stats["raw_bytes"] > 0
        assert stats["compressed_bytes"] > 0
        assert stats["compression_ratio_pct"] > 60.0
        assert "BTC" in stats["by_symbol"]
        assert "ETH" in stats["by_symbol"]
