"""SQLite 持久化实现."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.decision.decision import Decision, FullDecision
from xmlx_vlm.ai_trader.store.base import DecisionStore, EquitySnapshot


class SQLiteDecisionStore(DecisionStore):
    """基于 SQLite 的决策与权益快照存储."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        self._init_tables()

    def _init_tables(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decision_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trader_id TEXT NOT NULL,
                cycle_number INTEGER NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                system_prompt TEXT,
                user_prompt TEXT,
                cot_trace TEXT,
                decisions TEXT,
                raw_response TEXT,
                latency_ms INTEGER
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trader_id TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                total_equity TEXT,
                available_margin TEXT,
                unrealized_pnl TEXT,
                realized_pnl TEXT,
                margin_used_pct TEXT,
                position_count INTEGER
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_decision_time
            ON decision_records(timestamp_ms)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_decision_trader
            ON decision_records(trader_id)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_equity_time
            ON equity_snapshots(timestamp_ms)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_equity_trader
            ON equity_snapshots(trader_id)
            """
        )
        self._conn.commit()

    def save_decision(self, record: FullDecision) -> None:
        timestamp_ms = int(record.timestamp.timestamp() * 1000)
        self._conn.execute(
            """
            INSERT INTO decision_records
            (trader_id, cycle_number, timestamp_ms, system_prompt, user_prompt,
             cot_trace, decisions, raw_response, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.trader_id,
                record.cycle_number,
                timestamp_ms,
                record.system_prompt,
                record.user_prompt,
                record.cot_trace,
                json.dumps([d.to_dict() for d in record.decisions], ensure_ascii=False),
                record.raw_response,
                record.latency_ms,
            ),
        )
        self._conn.commit()

    def save_equity_snapshot(self, snapshot: EquitySnapshot) -> None:
        self._conn.execute(
            """
            INSERT INTO equity_snapshots
            (trader_id, timestamp_ms, total_equity, available_margin, unrealized_pnl,
             realized_pnl, margin_used_pct, position_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.trader_id,
                snapshot.timestamp_ms,
                str(snapshot.total_equity),
                str(snapshot.available_margin),
                str(snapshot.unrealized_pnl),
                str(snapshot.realized_pnl),
                str(snapshot.margin_used_pct),
                snapshot.position_count,
            ),
        )
        self._conn.commit()

    def list_decisions(
        self,
        trader_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[FullDecision]:
        cursor = self._conn.execute(
            """
            SELECT trader_id, cycle_number, timestamp_ms, system_prompt, user_prompt,
                   cot_trace, decisions, raw_response, latency_ms
            FROM decision_records
            WHERE trader_id = ?
            ORDER BY timestamp_ms DESC
            LIMIT ? OFFSET ?
            """,
            (trader_id, limit, offset),
        )
        results: List[FullDecision] = []
        for row in cursor.fetchall():
            results.append(self._row_to_full_decision(row))
        return results

    def list_equity_snapshots(
        self,
        trader_id: str,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[EquitySnapshot]:
        cursor = self._conn.execute(
            """
            SELECT trader_id, timestamp_ms, total_equity, available_margin, unrealized_pnl,
                   realized_pnl, margin_used_pct, position_count
            FROM equity_snapshots
            WHERE trader_id = ?
            ORDER BY timestamp_ms DESC
            LIMIT ? OFFSET ?
            """,
            (trader_id, limit, offset),
        )
        return [self._row_to_equity_snapshot(row) for row in cursor.fetchall()]

    def get_latest_equity_snapshot(self, trader_id: str) -> Optional[EquitySnapshot]:
        cursor = self._conn.execute(
            """
            SELECT trader_id, timestamp_ms, total_equity, available_margin, unrealized_pnl,
                   realized_pnl, margin_used_pct, position_count
            FROM equity_snapshots
            WHERE trader_id = ?
            ORDER BY timestamp_ms DESC
            LIMIT 1
            """,
            (trader_id,),
        )
        row = cursor.fetchone()
        return self._row_to_equity_snapshot(row) if row else None

    def close(self) -> None:
        try:
            self._conn.commit()
            self._conn.close()
        except sqlite3.Error:
            pass

    def _row_to_full_decision(self, row: sqlite3.Row) -> FullDecision:
        decisions_raw = json.loads(row[6] or "[]")
        return FullDecision(
            trader_id=row[0],
            cycle_number=row[1],
            timestamp=datetime.fromtimestamp(row[2] / 1000, tz=timezone.utc),
            system_prompt=row[3] or "",
            user_prompt=row[4] or "",
            cot_trace=row[5] or "",
            decisions=[Decision.from_dict(d) for d in decisions_raw],
            raw_response=row[7] or "",
            latency_ms=row[8] or 0,
        )

    def _row_to_equity_snapshot(self, row: sqlite3.Row) -> EquitySnapshot:
        return EquitySnapshot(
            trader_id=row[0],
            timestamp_ms=row[1],
            total_equity=row[2],
            available_margin=row[3],
            unrealized_pnl=row[4],
            realized_pnl=row[5],
            margin_used_pct=row[6],
            position_count=row[7] or 0,
        )
