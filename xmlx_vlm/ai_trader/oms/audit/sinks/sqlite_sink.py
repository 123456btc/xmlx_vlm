"""SQLite 审计 sink."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from xmlx_vlm.ai_trader.oms.audit.events import AuditEvent
from xmlx_vlm.ai_trader.oms.interfaces.audit_sink import AuditSink

logger = logging.getLogger(__name__)


class SQLiteAuditSink(AuditSink):
    """把审计事件写入 SQLite 数据库."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        self._init_tables()

    @property
    def name(self) -> str:
        return "sqlite"

    def write(self, event: AuditEvent) -> None:
        data = event.to_dict()
        try:
            self._conn.execute(
                """
                INSERT INTO audit_events
                (event_id, event_type, timestamp_ms, client_order_id, order_id, symbol, payload, raw)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["event_id"],
                    data["event_type"],
                    data["timestamp_ms"],
                    data.get("client_order_id"),
                    data.get("order_id"),
                    data.get("symbol"),
                    json.dumps(data.get("payload") or {}, ensure_ascii=False),
                    json.dumps(data.get("raw"), ensure_ascii=False, default=str)
                    if data.get("raw") is not None
                    else None,
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.error("sqlite audit write failed: %s", exc)

    def flush(self) -> None:
        try:
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.error("sqlite audit flush failed: %s", exc)

    def close(self) -> None:
        try:
            self._conn.commit()
            self._conn.close()
        except sqlite3.Error as exc:
            logger.error("sqlite audit close failed: %s", exc)

    def _init_tables(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                client_order_id TEXT,
                order_id TEXT,
                symbol TEXT,
                payload TEXT,
                raw TEXT
            )
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_audit_time
            ON audit_events(timestamp_ms)
            """
        )
        self._conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_audit_cloid
            ON audit_events(client_order_id)
            """
        )
        self._conn.commit()
