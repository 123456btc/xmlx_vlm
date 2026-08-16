"""时间工具."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from xmlx_vlm.ai_trader.oms.utils.clock import get_clock


def utc_now_ms() -> int:
    """返回 UTC 毫秒时间戳 (由 ClockProvider 提供)."""
    return get_clock().now_ms()


def utc_now_iso() -> str:
    """返回 ISO 8601 UTC 时间字符串 (由 ClockProvider 提供)."""
    return get_clock().now_iso()


def ms_to_iso(ts_ms: Optional[int]) -> Optional[str]:
    """毫秒时间戳转 ISO 字符串."""
    if ts_ms is None:
        return None
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()

