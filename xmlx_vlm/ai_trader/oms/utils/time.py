"""时间工具."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional


def utc_now_ms() -> int:
    """返回 UTC 毫秒时间戳."""
    return int(time.time() * 1000)


def utc_now_iso() -> str:
    """返回 ISO 8601 UTC 时间字符串."""
    return datetime.now(timezone.utc).isoformat()


def ms_to_iso(ts_ms: Optional[int]) -> Optional[str]:
    """毫秒时间戳转 ISO 字符串."""
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()
