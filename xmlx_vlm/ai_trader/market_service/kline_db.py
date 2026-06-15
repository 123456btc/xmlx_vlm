import sqlite3
import logging
from pathlib import Path
from typing import List, Optional
from xmlx_vlm.ai_trader.config import LOGS_DIR
from xmlx_vlm.ai_trader.market_service.models import Bar

logger = logging.getLogger(__name__)

class KlineDB:
    def __init__(self, db_path: Optional[Path] = None):
        import sys
        if "pytest" in sys.modules:
            self.db_path = db_path or Path("/tmp/ai_trader_test.db")
            if db_path is None:
                try:
                    if self.db_path.exists():
                        self.db_path.unlink()
                except Exception:
                    pass
        else:
            self.db_path = db_path or (LOGS_DIR / "ai_trader.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        return conn

    def _init_db(self):
        conn = self._connect()
        try:
            conn.execute("""
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
            conn.commit()
        finally:
            conn.close()

    def save_bar(self, bar: Bar) -> None:
        conn = self._connect()
        try:
            conn.execute("""
                INSERT INTO klines (symbol, timeframe, timestamp_ms, open, high, low, close, volume, buy_volume, sell_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe, timestamp_ms) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume,
                    buy_volume=excluded.buy_volume,
                    sell_volume=excluded.sell_volume
            """, (
                bar.symbol, bar.timeframe, bar.timestamp_ms,
                bar.open, bar.high, bar.low, bar.close, bar.volume,
                bar.buy_volume, bar.sell_volume
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving bar to SQLite: {e}")
        finally:
            conn.close()

    def save_bars(self, bars: List[Bar]) -> None:
        if not bars:
            return
        conn = self._connect()
        try:
            conn.executemany("""
                INSERT INTO klines (symbol, timeframe, timestamp_ms, open, high, low, close, volume, buy_volume, sell_volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe, timestamp_ms) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume,
                    buy_volume=excluded.buy_volume,
                    sell_volume=excluded.sell_volume
            """, [
                (
                    b.symbol, b.timeframe, b.timestamp_ms,
                    b.open, b.high, b.low, b.close, b.volume,
                    b.buy_volume, b.sell_volume
                ) for b in bars
            ])
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving bars to SQLite: {e}")
        finally:
            conn.close()

    def load_bars(self, symbol: str, timeframe: str, limit: int = 200) -> List[Bar]:
        conn = self._connect()
        try:
            cursor = conn.execute("""
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
            logger.error(f"Error loading bars from SQLite: {e}")
            return []
        finally:
            conn.close()

    def get_latest_timestamp(self, symbol: str, timeframe: str) -> Optional[int]:
        conn = self._connect()
        try:
            cursor = conn.execute("""
                SELECT MAX(timestamp_ms)
                FROM klines
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            row = cursor.fetchone()
            return row[0] if row and row[0] is not None else None
        except Exception as e:
            logger.error(f"Error getting latest K-line timestamp: {e}")
            return None
        finally:
            conn.close()
