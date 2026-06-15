"""QuantSessionDB - SQLite session database for persistent chat logs and trades."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from xmlx_vlm.ai_trader.config import DATA_DIR

class QuantSessionDB:
    """SQLite-based storage manager for chat sessions, messages, and executed trades."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else (DATA_DIR / "trader_sessions.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            
            # Sessions Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    model TEXT NOT NULL,
                    mode TEXT NOT NULL, -- 'paper' or 'live'
                    created_at REAL NOT NULL,
                    last_active_at REAL NOT NULL
                );
            """)

            # Messages Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL, -- 'user', 'assistant', 'system', 'tool'
                    content TEXT NOT NULL, -- JSON string or raw text
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );
            """)

            # Trades Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL, -- 'buy' or 'sell'
                    qty REAL NOT NULL,
                    price REAL NOT NULL,
                    pnl REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    status TEXT NOT NULL, -- 'simulated' or 'filled'
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );
            """)

            # KMS Config Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kms_config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)

            # KMS Keys Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kms_keys (
                    key_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    wallet_address TEXT NOT NULL,
                    encrypted_private_key TEXT NOT NULL, -- JSON formatted encrypted key data
                    testnet INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    status TEXT DEFAULT 'inactive'
                );
            """)

            # KMS Audit Logs Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kms_audit_logs (
                    log_id TEXT PRIMARY KEY,
                    timestamp REAL NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT NOT NULL
                );
            """)

            # Trade Reflections Table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reflections (
                    reflection_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    pnl REAL NOT NULL,
                    trade_details TEXT NOT NULL,
                    lesson TEXT NOT NULL,
                    timestamp REAL NOT NULL
                );
            """)
            conn.commit()

    # --- Session Management ---

    def create_session(
        self, session_id: str, title: str, model: str, mode: str
    ) -> Dict[str, Any]:
        now = time.time()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, title, model, mode, created_at, last_active_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (session_id, title, model, mode, now, now),
            )
            conn.commit()
        return {
            "session_id": session_id,
            "title": title,
            "model": model,
            "mode": mode,
            "created_at": now,
            "last_active_at": now,
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY last_active_at DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def update_session_activity(self, session_id: str, title: Optional[str] = None):
        now = time.time()
        with self._get_conn() as conn:
            if title:
                conn.execute(
                    """
                    UPDATE sessions 
                    SET last_active_at = ?, title = ? 
                    WHERE session_id = ?
                """,
                    (now, title, session_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE sessions 
                    SET last_active_at = ? 
                    WHERE session_id = ?
                """,
                    (now, session_id),
                )
            conn.commit()

    def delete_session(self, session_id: str):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()

    # --- Message Management ---

    def add_message(
        self, message_id: str, session_id: str, role: str, content: str | List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        now = time.time()
        content_str = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO messages (message_id, session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """,
                (message_id, session_id, role, content_str, now),
            )
            conn.commit()
        self.update_session_activity(session_id)
        return {
            "message_id": message_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "timestamp": now,
        }

    def get_messages(self, session_id: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            ).fetchall()
            messages = []
            for row in rows:
                msg = dict(row)
                # Try to decode JSON content if it's saved as list/dict
                try:
                    msg["content"] = json.loads(msg["content"])
                except Exception:
                    pass
                messages.append(msg)
            return messages

    # --- Trade Logging ---

    def log_trade(
        self,
        trade_id: str,
        session_id: str,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        pnl: float,
        status: str,
    ) -> Dict[str, Any]:
        now = time.time()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO trades (trade_id, session_id, symbol, side, qty, price, pnl, timestamp, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (trade_id, session_id, symbol, side, qty, price, pnl, now, status),
            )
            conn.commit()
        return {
            "trade_id": trade_id,
            "session_id": session_id,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "pnl": pnl,
            "timestamp": now,
            "status": status,
        }

    def get_trades(self, session_id: str) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trades WHERE session_id = ? ORDER BY timestamp DESC",
                (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    # --- KMS Vault Database Helpers ---

    def init_kms_vault(self, salt_hex: str, verifier_hex: str) -> None:
        """Initialize the KMS vault settings."""
        with self._get_conn() as conn:
            conn.execute("INSERT OR REPLACE INTO kms_config (key, value) VALUES (?, ?)", ("vault_initialized", "true"))
            conn.execute("INSERT OR REPLACE INTO kms_config (key, value) VALUES (?, ?)", ("vault_salt", salt_hex))
            conn.execute("INSERT OR REPLACE INTO kms_config (key, value) VALUES (?, ?)", ("vault_verifier", verifier_hex))
            conn.commit()

    def get_kms_config(self, key: str) -> Optional[str]:
        """Fetch a configuration value for KMS."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT value FROM kms_config WHERE key = ?", (key,)).fetchone()
            return row["value"] if row else None

    def add_kms_key(self, key_id: str, label: str, wallet_address: str, encrypted_key: str, testnet: bool) -> None:
        """Add an encrypted exchange key to the vault."""
        now = time.time()
        with self._get_conn() as conn:
            # First set all existing keys to inactive
            conn.execute("UPDATE kms_keys SET status = 'inactive'")
            # Insert the new key as active
            conn.execute(
                """
                INSERT INTO kms_keys (key_id, label, wallet_address, encrypted_private_key, testnet, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, 'active')
            """,
                (key_id, label, wallet_address, encrypted_key, 1 if testnet else 0, now),
            )
            conn.commit()

    def list_kms_keys(self) -> List[Dict[str, Any]]:
        """List all stored keys with masked encrypted payload details."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT key_id, label, wallet_address, testnet, created_at, status FROM kms_keys ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def get_encrypted_kms_key(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a specific encrypted key payload."""
        with self._get_conn() as conn:
            row = conn.execute("SELECT * FROM kms_keys WHERE key_id = ?", (key_id,)).fetchone()
            return dict(row) if row else None

    def delete_kms_key(self, key_id: str) -> None:
        """Delete an exchange key from the vault."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM kms_keys WHERE key_id = ?", (key_id,))
            conn.commit()

    def activate_kms_key(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Set a single key to active and all other keys to inactive. Returns the activated key info."""
        with self._get_conn() as conn:
            # First set all keys to inactive
            conn.execute("UPDATE kms_keys SET status = 'inactive'")
            # Set the selected key to active
            conn.execute("UPDATE kms_keys SET status = 'active' WHERE key_id = ?", (key_id,))
            conn.commit()
            
            # Fetch the activated key details
            row = conn.execute("SELECT * FROM kms_keys WHERE key_id = ?", (key_id,)).fetchone()
            return dict(row) if row else None

    def deactivate_all_kms_keys(self) -> None:
        """Deactivate all keys."""
        with self._get_conn() as conn:
            conn.execute("UPDATE kms_keys SET status = 'inactive'")
            conn.commit()

    def log_kms_audit(self, action: str, details: str) -> None:
        """Log a security or key action in the audit logs."""
        import uuid
        now = time.time()
        log_id = str(uuid.uuid4())
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO kms_audit_logs (log_id, timestamp, action, details)
                VALUES (?, ?, ?, ?)
            """,
                (log_id, now, action, details),
            )
            conn.commit()

    def get_kms_audit_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve the latest vault audit events."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM kms_audit_logs ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
            return [dict(row) for row in rows]

    # --- Reflection Management ---

    def add_reflection(
        self, symbol: str, pnl: float, trade_details: str | Dict[str, Any], lesson: str
    ) -> Dict[str, Any]:
        """Save a post-trade reflection to the database."""
        import uuid
        now = time.time()
        reflection_id = str(uuid.uuid4())
        details_str = trade_details if isinstance(trade_details, str) else json.dumps(trade_details, ensure_ascii=False)
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO reflections (reflection_id, symbol, pnl, trade_details, lesson, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (reflection_id, symbol.upper(), pnl, details_str, lesson, now),
            )
            conn.commit()
        return {
            "reflection_id": reflection_id,
            "symbol": symbol.upper(),
            "pnl": pnl,
            "trade_details": trade_details,
            "lesson": lesson,
            "timestamp": now,
        }

    def get_recent_reflections(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch the most recent post-trade reflections."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM reflections ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            reflections = []
            for row in rows:
                ref = dict(row)
                try:
                    ref["trade_details"] = json.loads(ref["trade_details"])
                except Exception:
                    pass
                reflections.append(ref)
            return reflections
