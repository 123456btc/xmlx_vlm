"""JSON Lines 文件审计 sink."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from xmlx_vlm.ai_trader.oms.audit.events import AuditEvent
from xmlx_vlm.ai_trader.oms.interfaces.audit_sink import AuditSink
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms

logger = logging.getLogger(__name__)


class FileAuditSink(AuditSink):
    """把审计事件追加写入 JSON Lines 文件."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._file: Any = None
        self._current_date: str = ""
        self._open_file()

    @property
    def name(self) -> str:
        return "file"

    def write(self, event: AuditEvent) -> None:
        self._rotate_if_needed()
        line = event.to_json()
        self._file.write(line + "\n")

    def flush(self) -> None:
        if self._file:
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def _rotate_if_needed(self):
        from datetime import datetime, timezone

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self._current_date:
            self._current_date = today
            self._open_file()

    def _open_file(self):
        if self._file:
            self._file.close()
        from datetime import datetime, timezone

        self._current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.log_dir / f"oms_audit_{self._current_date}.jsonl"
        self._file = open(path, "a", encoding="utf-8")
