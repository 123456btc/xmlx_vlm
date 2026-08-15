# SPDX-License-Identifier: Apache-2.0
"""
Kanban Board -- Persistent SQLite task store for multi-agent workflows.

Provides durable task tracking, atomic task claiming, heartbeat monitoring,
and automatic deadlock/failure mitigation across heterogeneous specialist agents.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_KANBAN_DB = os.path.expanduser("~/.cache/xmlx_vlm/kanban.db")


@dataclass
class KanbanTask:
    """Task entity in the Kanban board."""

    id: str
    title: str
    description: str
    assignee_profile: str = "default"
    status: str = "todo"  # "todo" | "in_progress" | "done" | "blocked" | "archived"
    priority: int = 3  # 1 (lowest) to 5 (highest)
    worker_id: Optional[str] = None
    failure_count: int = 0
    last_error: Optional[str] = None
    result_summary: Optional[str] = None
    heartbeat_at: Optional[float] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KanbanBoard:
    """
    Thread-safe SQLite-backed board for multi-agent task orchestration.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_KANBAN_DB
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kanban_tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    assignee_profile TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    worker_id TEXT,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    result_summary TEXT,
                    heartbeat_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    metadata TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kanban_status ON kanban_tasks(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_kanban_profile ON kanban_tasks(assignee_profile)")
            conn.commit()

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> KanbanTask:
        meta_raw = row["metadata"]
        meta_dict = json.loads(meta_raw) if meta_raw else None
        return KanbanTask(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            assignee_profile=row["assignee_profile"],
            status=row["status"],
            priority=row["priority"],
            worker_id=row["worker_id"],
            failure_count=row["failure_count"],
            last_error=row["last_error"],
            result_summary=row["result_summary"],
            heartbeat_at=row["heartbeat_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=meta_dict,
        )

    def create_task(
        self,
        title: str,
        description: str,
        assignee_profile: str = "default",
        priority: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KanbanTask:
        """Create and queue a new task."""
        now = time.time()
        task_id = str(uuid.uuid4())[:8]
        task = KanbanTask(
            id=task_id,
            title=title,
            description=description,
            assignee_profile=assignee_profile,
            status="todo",
            priority=priority,
            created_at=now,
            updated_at=now,
            metadata=metadata,
        )

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO kanban_tasks (
                        id, title, description, assignee_profile, status, priority,
                        worker_id, failure_count, last_error, result_summary,
                        heartbeat_at, created_at, updated_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task.id,
                        task.title,
                        task.description,
                        task.assignee_profile,
                        task.status,
                        task.priority,
                        task.worker_id,
                        task.failure_count,
                        task.last_error,
                        task.result_summary,
                        task.heartbeat_at,
                        task.created_at,
                        task.updated_at,
                        json.dumps(metadata) if metadata else None,
                    ),
                )
                conn.commit()

        return task

    def get_task(self, task_id: str) -> Optional[KanbanTask]:
        """Fetch a task by ID."""
        with self._connect() as conn:
            cur = conn.execute("SELECT * FROM kanban_tasks WHERE id = ?", (task_id,))
            row = cur.fetchone()
            if row:
                return self._row_to_task(row)
        return None

    def list_tasks(
        self,
        status: Optional[str] = None,
        assignee_profile: Optional[str] = None,
        limit: int = 50,
    ) -> List[KanbanTask]:
        """List tasks matching filter criteria, ordered by priority descending."""
        query = "SELECT * FROM kanban_tasks WHERE 1=1"
        params: List[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if assignee_profile:
            query += " AND assignee_profile = ?"
            params.append(assignee_profile)

        query += " ORDER BY priority DESC, created_at ASC LIMIT ?"
        params.append(limit)

        with self._connect() as conn:
            cur = conn.execute(query, params)
            return [self._row_to_task(row) for row in cur.fetchall()]

    def claim_task(
        self,
        assignee_profile: str,
        worker_id: str,
    ) -> Optional[KanbanTask]:
        """
        Atomically claim the highest priority 'todo' task for a given profile.
        """
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                # Find best candidate
                cur = conn.execute(
                    """
                    SELECT id FROM kanban_tasks
                    WHERE status = 'todo' AND (assignee_profile = ? OR assignee_profile = 'default')
                    ORDER BY priority DESC, created_at ASC LIMIT 1
                    """,
                    (assignee_profile,),
                )
                row = cur.fetchone()
                if not row:
                    return None

                task_id = row["id"]
                conn.execute(
                    """
                    UPDATE kanban_tasks
                    SET status = 'in_progress', worker_id = ?, heartbeat_at = ?, updated_at = ?
                    WHERE id = ? AND status = 'todo'
                    """,
                    (worker_id, now, now, task_id),
                )
                conn.commit()

        return self.get_task(task_id)

    def heartbeat(self, task_id: str, worker_id: str) -> bool:
        """Update task heartbeat timestamp to avoid stale reclaim."""
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    UPDATE kanban_tasks
                    SET heartbeat_at = ?, updated_at = ?
                    WHERE id = ? AND worker_id = ? AND status = 'in_progress'
                    """,
                    (now, now, task_id, worker_id),
                )
                conn.commit()
                return cur.rowcount > 0

    def complete_task(self, task_id: str, result_summary: str = "") -> bool:
        """Mark task as successfully completed."""
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    UPDATE kanban_tasks
                    SET status = 'done', result_summary = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (result_summary, now, task_id),
                )
                conn.commit()
                return cur.rowcount > 0

    def fail_task(self, task_id: str, error: str = "", failure_limit: int = 2) -> bool:
        """
        Record a failure attempt. If failure count >= failure_limit, auto-block task.
        """
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute("SELECT failure_count FROM kanban_tasks WHERE id = ?", (task_id,))
                row = cur.fetchone()
                if not row:
                    return False

                new_count = row["failure_count"] + 1
                new_status = "blocked" if new_count >= failure_limit else "todo"

                conn.execute(
                    """
                    UPDATE kanban_tasks
                    SET status = ?, failure_count = ?, last_error = ?, worker_id = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (new_status, new_count, error, now, task_id),
                )
                conn.commit()
                return True

    def block_task(self, task_id: str, reason: str = "") -> bool:
        """Explicitly block a task."""
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    UPDATE kanban_tasks
                    SET status = 'blocked', last_error = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (reason, now, task_id),
                )
                conn.commit()
                return cur.rowcount > 0

    def reclaim_stale_tasks(self, stale_timeout_seconds: float = 120.0) -> int:
        """Reclaim in_progress tasks whose heartbeat expired."""
        now = time.time()
        cutoff = now - stale_timeout_seconds
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    UPDATE kanban_tasks
                    SET status = 'todo', worker_id = NULL, updated_at = ?
                    WHERE status = 'in_progress' AND (heartbeat_at IS NULL OR heartbeat_at < ?)
                    """,
                    (now, cutoff),
                )
                conn.commit()
                return cur.rowcount
