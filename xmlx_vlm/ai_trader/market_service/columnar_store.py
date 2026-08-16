"""
High-Performance Columnar Market Data Store.

Software Architecture & Philosophy:
1. Columnar Native & Zero-Copy: Data stored as contiguous columnar arrays (time, OHLCV, CVD, OI, Funding, Imbalance)
2. Symbol-Partitioned Sharding: Independent per-symbol storage units with lock-free concurrency
3. Chunk-Level Immutability: Historical chunks are frozen, versions are tracked
4. Point-in-Time (as_of) Time Travel: Strict prevention of lookahead bias for backtesting and AI agent auditing
"""

from __future__ import annotations

import json
import logging
import math
import os
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# ACDB Protocol Constants (Arctic Columnar Data Block)
MAGIC_HEADER = b"ACDB"
CURRENT_ACDB_VERSION = 1
CODEC_ZLIB = 1


class ChunkCompressionAdapter:
    """High-performance binary compression and decompression adapter for ColumnarChunks."""

    @staticmethod
    def compress(raw_bytes: bytes, level: int = 6) -> bytes:
        """
        Compress raw bytes into ACDB container format:
        [4B Magic][1B Version][1B Codec][4B UncompressedSize][4B CRC32][CompressedPayload]
        """
        uncompressed_size = len(raw_bytes)
        crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
        compressed_payload = zlib.compress(raw_bytes, level=level)
        header = struct.pack(
            ">4sBBII",
            MAGIC_HEADER,
            CURRENT_ACDB_VERSION,
            CODEC_ZLIB,
            uncompressed_size,
            crc,
        )
        return header + compressed_payload

    @staticmethod
    def decompress(container_bytes: bytes) -> bytes:
        """Decompress an ACDB container and verify data integrity via CRC32."""
        if len(container_bytes) < 14:
            raise ValueError(f"ACDB container too short: {len(container_bytes)} bytes")
        magic, version, codec, uncompressed_size, expected_crc = struct.unpack(
            ">4sBBII", container_bytes[:14]
        )
        if magic != MAGIC_HEADER:
            raise ValueError(f"Invalid magic header: {magic!r}, expected {MAGIC_HEADER!r}")
        if codec != CODEC_ZLIB:
            raise ValueError(f"Unsupported codec ID: {codec}")

        compressed_payload = container_bytes[14:]
        raw_bytes = zlib.decompress(compressed_payload)
        if len(raw_bytes) != uncompressed_size:
            raise ValueError(
                f"Decompressed size mismatch: got {len(raw_bytes)}, expected {uncompressed_size}"
            )
        actual_crc = zlib.crc32(raw_bytes) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError(f"CRC32 mismatch: got {actual_crc:#x}, expected {expected_crc:#x}")
        return raw_bytes


@dataclass
class ColumnarChunk:
    """
    An immutable columnar data block containing time-series market data.
    """
    symbol: str
    timeframe: str
    chunk_id: str
    version: int = 1
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    
    # Column arrays (contiguous lists / primitive memory buffers)
    timestamps: List[int] = field(default_factory=list)
    opens: List[float] = field(default_factory=list)
    highs: List[float] = field(default_factory=list)
    lows: List[float] = field(default_factory=list)
    closes: List[float] = field(default_factory=list)
    volumes: List[float] = field(default_factory=list)
    cvds: List[float] = field(default_factory=list)
    ois: List[float] = field(default_factory=list)
    fundings: List[float] = field(default_factory=list)
    imbalances: List[float] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.timestamps)

    @property
    def start_ts(self) -> Optional[int]:
        return self.timestamps[0] if self.timestamps else None

    @property
    def end_ts(self) -> Optional[int]:
        return self.timestamps[-1] if self.timestamps else None

    def append_row(
        self,
        ts: int,
        o: float,
        h: float,
        l: float,
        c: float,
        v: float,
        cvd: float = 0.0,
        oi: float = 0.0,
        funding: float = 0.0,
        imbalance: float = 0.0,
    ) -> None:
        """Append a row to columnar buffers (only valid when chunk is mutable)."""
        self.timestamps.append(int(ts))
        self.opens.append(float(o))
        self.highs.append(float(h))
        self.lows.append(float(l))
        self.closes.append(float(c))
        self.volumes.append(float(v))
        self.cvds.append(float(cvd))
        self.ois.append(float(oi))
        self.fundings.append(float(funding))
        self.imbalances.append(float(imbalance))

    def slice(
        self,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        as_of_ms: Optional[int] = None,
    ) -> Dict[str, List[Any]]:
        """
        Point-in-Time slice query.
        Returns columnar dictionary filtered strictly before or at `as_of_ms`.
        """
        if not self.timestamps:
            return self._empty_columns()

        # Effective upper bound
        effective_end = end_ts
        if as_of_ms is not None:
            effective_end = min(effective_end, as_of_ms) if effective_end is not None else as_of_ms

        indices = []
        for i, ts in enumerate(self.timestamps):
            if start_ts is not None and ts < start_ts:
                continue
            if effective_end is not None and ts > effective_end:
                continue
            indices.append(i)

        if not indices:
            return self._empty_columns()

        return {
            "timestamp": [self.timestamps[i] for i in indices],
            "open": [self.opens[i] for i in indices],
            "high": [self.highs[i] for i in indices],
            "low": [self.lows[i] for i in indices],
            "close": [self.closes[i] for i in indices],
            "volume": [self.volumes[i] for i in indices],
            "cvd": [self.cvds[i] for i in indices],
            "oi": [self.ois[i] for i in indices],
            "funding": [self.fundings[i] for i in indices],
            "imbalance": [self.imbalances[i] for i in indices],
        }

    def _empty_columns(self) -> Dict[str, List[Any]]:
        return {
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
            "cvd": [],
            "oi": [],
            "funding": [],
            "imbalance": [],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "chunk_id": self.chunk_id,
            "version": self.version,
            "created_at_ms": self.created_at_ms,
            "row_count": self.row_count,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "data": {
                "timestamp": self.timestamps,
                "open": self.opens,
                "high": self.highs,
                "low": self.lows,
                "close": self.closes,
                "volume": self.volumes,
                "cvd": self.cvds,
                "oi": self.ois,
                "funding": self.fundings,
                "imbalance": self.imbalances,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnarChunk":
        d = data.get("data", {})
        chunk = cls(
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            chunk_id=data["chunk_id"],
            version=data.get("version", 1),
            created_at_ms=data.get("created_at_ms", int(time.time() * 1000)),
            timestamps=list(d.get("timestamp", [])),
            opens=list(d.get("open", [])),
            highs=list(d.get("high", [])),
            lows=list(d.get("low", [])),
            closes=list(d.get("close", [])),
            volumes=list(d.get("volume", [])),
            cvds=list(d.get("cvd", [])),
            ois=list(d.get("oi", [])),
            fundings=list(d.get("funding", [])),
            imbalances=list(d.get("imbalance", [])),
        )
        return chunk

    def to_compressed_bytes(self, level: int = 6) -> bytes:
        """Serialize chunk to compressed ACDB binary format."""
        raw_payload = json.dumps(self.to_dict()).encode("utf-8")
        return ChunkCompressionAdapter.compress(raw_payload, level=level)

    @classmethod
    def from_compressed_bytes(cls, container_bytes: bytes) -> "ColumnarChunk":
        """Deserialize chunk from compressed ACDB binary format."""
        raw_payload = ChunkCompressionAdapter.decompress(container_bytes)
        data = json.loads(raw_payload.decode("utf-8"))
        return cls.from_dict(data)


class SymbolPartition:
    """
    Symbol-Partitioned storage unit managing mutable hot buffers and immutable cold chunks.
    """

    def __init__(self, symbol: str, chunk_size: int = 1000, storage_dir: Optional[Path] = None):
        self.symbol = symbol.upper()
        self.chunk_size = chunk_size
        self.storage_dir = storage_dir
        self._lock = threading.RLock()
        
        # timeframe -> List[ColumnarChunk]
        self._frozen_chunks: Dict[str, List[ColumnarChunk]] = {}
        # timeframe -> active mutable ColumnarChunk
        self._hot_buffers: Dict[str, ColumnarChunk] = {}
        self._version_counter: int = 1

        if self.storage_dir:
            self.load_from_disk()

    def append(
        self,
        timeframe: str,
        ts: int,
        o: float,
        h: float,
        l: float,
        c: float,
        v: float,
        cvd: float = 0.0,
        oi: float = 0.0,
        funding: float = 0.0,
        imbalance: float = 0.0,
    ) -> None:
        with self._lock:
            if timeframe not in self._hot_buffers:
                self._hot_buffers[timeframe] = ColumnarChunk(
                    symbol=self.symbol,
                    timeframe=timeframe,
                    chunk_id=f"{self.symbol}_{timeframe}_v{self._version_counter}_{int(time.time()*1000)}",
                    version=self._version_counter,
                )

            hot = self._hot_buffers[timeframe]
            hot.append_row(ts, o, h, l, c, v, cvd, oi, funding, imbalance)

            # If hot buffer exceeds chunk_size, freeze it into immutable segments
            if hot.row_count >= self.chunk_size:
                self._freeze_hot_buffer(timeframe)

    def _freeze_hot_buffer(self, timeframe: str) -> None:
        """Freeze current mutable chunk into immutable historical store."""
        hot = self._hot_buffers.pop(timeframe, None)
        if not hot or hot.row_count == 0:
            return

        if timeframe not in self._frozen_chunks:
            self._frozen_chunks[timeframe] = []

        self._frozen_chunks[timeframe].append(hot)
        self._version_counter += 1
        logger.debug(
            "Frozen chunk [%s] (%d rows) for %s:%s",
            hot.chunk_id,
            hot.row_count,
            self.symbol,
            timeframe,
        )

        # Disk flush (compressed ACDB format)
        if self.storage_dir:
            self._flush_chunk_to_disk(hot)

    def _flush_chunk_to_disk(self, chunk: ColumnarChunk) -> None:
        """Flush chunk to compressed ACDB file on disk."""
        try:
            target_dir = self.storage_dir / self.symbol / chunk.timeframe
            target_dir.mkdir(parents=True, exist_ok=True)
            chunk_file = target_dir / f"{chunk.chunk_id}.acdb"
            compressed_bytes = chunk.to_compressed_bytes()
            with open(chunk_file, "wb") as f:
                f.write(compressed_bytes)
        except Exception as exc:
            logger.warning("Failed to flush compressed chunk to disk: %s", exc)

    def load_from_disk(self) -> int:
        """Load all compressed ACDB chunks (and legacy JSON chunks) from storage_dir."""
        if not self.storage_dir or not self.storage_dir.exists():
            return 0
        
        sym_dir = self.storage_dir / self.symbol
        if not sym_dir.exists():
            return 0

        loaded_count = 0
        with self._lock:
            for tf_dir in sym_dir.iterdir():
                if not tf_dir.is_dir():
                    continue
                tf = tf_dir.name
                chunks: List[ColumnarChunk] = []

                # Scan ACDB compressed files
                for fpath in sorted(tf_dir.glob("*.acdb")):
                    try:
                        with open(fpath, "rb") as f:
                            raw = f.read()
                        chk = ColumnarChunk.from_compressed_bytes(raw)
                        chunks.append(chk)
                        loaded_count += 1
                    except Exception as e:
                        logger.warning("Failed to load compressed chunk %s: %s", fpath, e)

                # Scan legacy JSON files if any
                for fpath in sorted(tf_dir.glob("*.json")):
                    try:
                        with open(fpath, "r", encoding="utf-8") as f:
                            d = json.load(f)
                        chk = ColumnarChunk.from_dict(d)
                        chunks.append(chk)
                        loaded_count += 1
                    except Exception as e:
                        logger.warning("Failed to load legacy chunk %s: %s", fpath, e)

                # Sort chunks by start_ts
                chunks.sort(key=lambda c: c.start_ts or 0)
                if tf not in self._frozen_chunks:
                    self._frozen_chunks[tf] = []
                self._frozen_chunks[tf].extend(chunks)

        return loaded_count

    def query(
        self,
        timeframe: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        limit: Optional[int] = None,
        as_of_ms: Optional[int] = None,
    ) -> Dict[str, List[Any]]:
        """
        Point-in-Time query across both frozen chunks and active hot buffer.
        """
        with self._lock:
            all_slices = []

            # 1. Query frozen immutable chunks
            chunks = self._frozen_chunks.get(timeframe, [])
            for chunk in chunks:
                sl = chunk.slice(start_ts, end_ts, as_of_ms)
                if sl["timestamp"]:
                    all_slices.append(sl)

            # 2. Query active hot buffer
            hot = self._hot_buffers.get(timeframe)
            if hot:
                sl = hot.slice(start_ts, end_ts, as_of_ms)
                if sl["timestamp"]:
                    all_slices.append(sl)

            if not all_slices:
                return ColumnarChunk(self.symbol, timeframe, "empty")._empty_columns()

            # Merge columnar slices into a single unified contiguous columnar map
            merged = {k: [] for k in all_slices[0].keys()}
            for sl in all_slices:
                for k in merged.keys():
                    merged[k].extend(sl[k])

            if limit is not None and len(merged["timestamp"]) > limit:
                # Return most recent `limit` rows
                for k in merged.keys():
                    merged[k] = merged[k][-limit:]

            return merged

    def storage_stats(self) -> Dict[str, Any]:
        """Calculate uncompressed memory bytes vs compressed ACDB bytes across all chunks."""
        with self._lock:
            total_rows = 0
            total_chunks = 0
            raw_bytes = 0
            compressed_bytes = 0

            # Measure all frozen chunks
            for tf, chunks in self._frozen_chunks.items():
                for c in chunks:
                    total_chunks += 1
                    total_rows += c.row_count
                    json_bytes = json.dumps(c.to_dict()).encode("utf-8")
                    c_bytes = c.to_compressed_bytes()
                    raw_bytes += len(json_bytes)
                    compressed_bytes += len(c_bytes)

            # Measure hot buffers
            for tf, hot in self._hot_buffers.items():
                if hot.row_count > 0:
                    total_chunks += 1
                    total_rows += hot.row_count
                    json_bytes = json.dumps(hot.to_dict()).encode("utf-8")
                    c_bytes = hot.to_compressed_bytes()
                    raw_bytes += len(json_bytes)
                    compressed_bytes += len(c_bytes)

            saved_bytes = max(0, raw_bytes - compressed_bytes)
            ratio_pct = (saved_bytes / raw_bytes * 100.0) if raw_bytes > 0 else 0.0

            return {
                "symbol": self.symbol,
                "total_chunks": total_chunks,
                "total_rows": total_rows,
                "raw_bytes": raw_bytes,
                "compressed_bytes": compressed_bytes,
                "saved_bytes": saved_bytes,
                "compression_ratio_pct": round(ratio_pct, 2),
            }


class ColumnarMarketStore:
    """
    Global Columnar Market Data Infrastructure.
    Provides Point-in-Time slicing, symbol partitioning, and zero-copy data views.
    """

    _instance: Optional[ColumnarMarketStore] = None

    @classmethod
    def get_instance(cls, storage_dir: Optional[Union[str, Path]] = None) -> "ColumnarMarketStore":
        if cls._instance is None:
            cls._instance = cls(storage_dir=storage_dir)
        return cls._instance

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None, chunk_size: int = 1000):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.chunk_size = chunk_size
        self._partitions: Dict[str, SymbolPartition] = {}
        self._lock = threading.RLock()
        
        # Recent state snapshots: symbol -> List of Point-in-Time state dicts
        self._point_in_time_snapshots: Dict[str, List[Dict[str, Any]]] = {}

    def _get_partition(self, symbol: str) -> SymbolPartition:
        sym = symbol.upper().strip()
        with self._lock:
            if sym not in self._partitions:
                self._partitions[sym] = SymbolPartition(
                    symbol=sym,
                    chunk_size=self.chunk_size,
                    storage_dir=self.storage_dir,
                )
            return self._partitions[sym]

    def append_bar(
        self,
        symbol: str,
        timeframe: str,
        ts: int,
        o: float,
        h: float,
        l: float,
        c: float,
        v: float,
        cvd: float = 0.0,
        oi: float = 0.0,
        funding: float = 0.0,
        imbalance: float = 0.0,
    ) -> None:
        """Write a new Bar into the columnar store."""
        part = self._get_partition(symbol)
        part.append(timeframe, ts, o, h, l, c, v, cvd, oi, funding, imbalance)

    def append_state_snapshot(self, symbol: str, state_dict: Dict[str, Any], ts_ms: Optional[int] = None) -> None:
        """Record an immutable Point-in-Time market state snapshot for auditing/backtesting."""
        sym = symbol.upper().strip()
        now_ts = ts_ms if ts_ms is not None else int(time.time() * 1000)
        with self._lock:
            if sym not in self._point_in_time_snapshots:
                self._point_in_time_snapshots[sym] = []
            
            entry = {
                "timestamp_ms": now_ts,
                "symbol": sym,
                "state": state_dict,
            }
            self._point_in_time_snapshots[sym].append(entry)
            # Keep recent 500 snapshots in memory
            if len(self._point_in_time_snapshots[sym]) > 500:
                self._point_in_time_snapshots[sym] = self._point_in_time_snapshots[sym][-500:]

    def query_columnar(
        self,
        symbol: str,
        timeframe: str = "1m",
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        limit: Optional[int] = None,
        as_of_ms: Optional[int] = None,
    ) -> Dict[str, List[Any]]:
        """
        Query historical time-series data with Point-in-Time (`as_of_ms`) isolation.
        """
        part = self._get_partition(symbol)
        return part.query(
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=limit,
            as_of_ms=as_of_ms,
        )

    def get_snapshot_as_of(self, symbol: str, as_of_ms: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve the exact market state snapshot as it existed at `as_of_ms`.
        Guarantees zero lookahead bias.
        """
        sym = symbol.upper().strip()
        with self._lock:
            snapshots = self._point_in_time_snapshots.get(sym, [])
            if not snapshots:
                return None
            
            if as_of_ms is None:
                return snapshots[-1]["state"]

            # Binary search / linear scan for snapshot strictly <= as_of_ms
            best = None
            for s in snapshots:
                if s["timestamp_ms"] <= as_of_ms:
                    best = s["state"]
                else:
                    break
            return best

    def storage_stats(self) -> Dict[str, Any]:
        """Aggregate storage stats across all symbol partitions."""
        with self._lock:
            total_chunks = 0
            total_rows = 0
            raw_bytes = 0
            compressed_bytes = 0
            by_symbol = {}
            for sym, part in self._partitions.items():
                p_stat = part.storage_stats()
                by_symbol[sym] = p_stat
                total_chunks += p_stat["total_chunks"]
                total_rows += p_stat["total_rows"]
                raw_bytes += p_stat["raw_bytes"]
                compressed_bytes += p_stat["compressed_bytes"]

            saved_bytes = max(0, raw_bytes - compressed_bytes)
            ratio_pct = (saved_bytes / raw_bytes * 100.0) if raw_bytes > 0 else 0.0

            return {
                "total_symbols": len(self._partitions),
                "total_chunks": total_chunks,
                "total_rows": total_rows,
                "raw_bytes": raw_bytes,
                "compressed_bytes": compressed_bytes,
                "saved_bytes": saved_bytes,
                "compression_ratio_pct": round(ratio_pct, 2),
                "by_symbol": by_symbol,
            }

