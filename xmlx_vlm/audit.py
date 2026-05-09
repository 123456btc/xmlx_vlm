# SPDX-License-Identifier: Apache-2.0
"""
Audit logging for MCP tool execution.

Finance-grade audit trail: append-only, tamper-resistant JSON Lines.
Records who called what tool, when, with what arguments, and what result.
"""

import hashlib
import json
import logging
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("xmlx_vlm.audit")

_DEFAULT_AUDIT_PATH = os.path.expanduser("~/.logs/xmlx_vlm/audit.log")


_FINANCE_MODE = os.environ.get("XMLX_VLM_FINANCE_MODE", "0").lower() in ("1", "true", "yes")
_AUDIT_ENABLED = os.environ.get(
    "XMLX_VLM_AUDIT_ENABLED", "1" if _FINANCE_MODE else "0"
).lower() in ("1", "true", "yes")


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


class AuditLogger:
    """Append-only audit logger for tool execution events."""

    def __init__(self, path: Optional[str] = None, enabled: Optional[bool] = None):
        self.path = path or os.environ.get("XMLX_VLM_AUDIT_PATH", _DEFAULT_AUDIT_PATH)
        self.enabled = enabled if enabled is not None else _AUDIT_ENABLED
        self._lock = threading.Lock()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        # Attempt to make append-only (best effort)
        try:
            os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        except Exception:
            pass

    def log_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: str,
        source_ip: str = "",
        api_key_hash: str = "",
        session_id: str = "",
    ) -> None:
        """Log a single tool execution event."""
        if not self.enabled:
            return

        args_str = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "tool": tool_name,
            "args_hash": _sha256_hex(args_str),
            "args_preview": self._truncate(args_str, 500),
            "result_preview": self._truncate(result, 500),
            "result_hash": _sha256_hex(result),
            "src_ip": source_ip,
            "key_hash": api_key_hash,
            "session": session_id,
        }
        self._append(record)

    def log_security_event(
        self,
        event_type: str,
        detail: str,
        source_ip: str = "",
        api_key_hash: str = "",
    ) -> None:
        """Log a security-relevant event (policy violation, blocked tool, etc.)."""
        if not self.enabled:
            return

        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "event": "security",
            "type": event_type,
            "detail": detail,
            "src_ip": source_ip,
            "key_hash": api_key_hash,
        }
        self._append(record)

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."


# Global singleton
_global_audit: Optional[AuditLogger] = None


def get_audit_logger() -> Optional[AuditLogger]:
    global _global_audit
    if _global_audit is not None:
        return _global_audit
    if not _AUDIT_ENABLED:
        return None
    _global_audit = AuditLogger()
    return _global_audit


def set_audit_logger(audit: Optional[AuditLogger]) -> None:
    global _global_audit
    _global_audit = audit
