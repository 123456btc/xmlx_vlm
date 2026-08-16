import sqlite3
import logging
import threading
from pathlib import Path
from typing import List, Optional
from xmlx_vlm.ai_trader.config import LOGS_DIR
from xmlx_vlm.ai_trader.market_service.models import Bar

logger = logging.getLogger(__name__)


class KlineDB:
    """SQLite-backed K-line store.

    Uses a **persistent connection** (one per KlineDB instance) to avoid the
    open→PRAGMA→commit→close overhead on every ``save_bar`` call.  WAL mode
    and a write-lock guard make concurrent reads from the query thread safe.
    """

    def __init__(self, db_path: Optional[Path] = None):
        import sys
        if "pytest" in sys.modules:
            self.db_path = db_path or (LOGS_DIR / "ai_trader_test.db")
            if db_path is None:
                try:
                    if self.db_path.exists():
                        self.db_path.unlink()
                except Exception:
                    pass
        else:
            self.db_path = db_path or (LOGS_DIR / "ai_trader.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Persistent connection — thread-safe via _lock.
        # check_same_thread=False is intentional: we serialise access with _lock.
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self.db_path), timeout=30.0, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")  # safe with WAL, faster commits
        self._init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS klines (
                    symbol TEXT,
                    timeframe TEXT,
                    timestamp_ms INTEGER,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    buy_volume REAL,
                    sell_volume REAL,
                    PRIMARY KEY (symbol, timeframe, timestamp_ms)
                )
            """)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Write helpers
    # ------------------------------------------------------------------

    _UPSERT_SQL = """
        INSERT INTO klines
            (symbol, timeframe, timestamp_ms, open, high, low, close, volume, buy_volume, sell_volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, timeframe, timestamp_ms) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            volume=excluded.volume,
            buy_volume=excluded.buy_volume,
            sell_volume=excluded.sell_volume
    """

    @staticmethod
    def _bar_tuple(bar: Bar) -> tuple:
        return (
            bar.symbol, bar.timeframe, bar.timestamp_ms,
            bar.open, bar.high, bar.low, bar.close, bar.volume,
            bar.buy_volume, bar.sell_volume,
        )

    def save_bar(self, bar: Bar) -> None:
        try:
            with self._lock:
                self._conn.execute(self._UPSERT_SQL, self._bar_tuple(bar))
                self._conn.commit()
        except Exception as e:
            logger.error("Error saving bar to SQLite: %s", e)

    def save_bars(self, bars: List[Bar]) -> None:
        if not bars:
            return
        try:
            with self._lock:
                self._conn.executemany(self._UPSERT_SQL, [self._bar_tuple(b) for b in bars])
                self._conn.commit()
        except Exception as e:
            logger.error("Error saving bars to SQLite: %s", e)

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def load_bars(self, symbol: str, timeframe: str, limit: int = 200) -> List[Bar]:
        try:
            with self._lock:
                cursor = self._conn.execute("""
                    SELECT open, high, low, close, volume, buy_volume, sell_volume, timestamp_ms
                    FROM klines
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY timestamp_ms DESC
                    LIMIT ?
                """, (symbol, timeframe, limit))
                rows = cursor.fetchall()
            bars = []
            for row in reversed(rows):
                bars.append(Bar(
                    symbol=symbol,
                    timeframe=timeframe,
                    open=row[0],
                    high=row[1],
                    low=row[2],
                    close=row[3],
                    volume=row[4],
                    timestamp_ms=row[7],
                    buy_volume=row[5],
                    sell_volume=row[6],
                ))
            return bars
        except Exception as e:
            logger.error("Error loading bars from SQLite: %s", e)
            return []

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        try:
            with self._lock:
                cursor = self._conn.execute("""
                    SELECT MAX(timestamp_ms)
                    FROM klines
                    WHERE symbol = ? AND timeframe = ?
                """, (symbol, timeframe))
                row = cursor.fetchone()
            return row[0] if row and row[0] is not None else None
        except Exception as e:
            logger.error("Error getting latest K-line timestamp: %s", e)
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Explicitly close the persistent connection (call on shutdown)."""
        try:
            with self._lock:
                self._conn.close()
        except Exception:
            pass
