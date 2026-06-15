"""API 错误熔断器."""

from __future__ import annotations

from collections import deque
from typing import Optional

from xmlx_vlm.ai_trader.oms.circuit.breaker import CircuitBreaker
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms


class ApiErrorCircuitBreaker(CircuitBreaker):
    """连续 API 异常次数达到阈值时熔断."""

    def __init__(
        self,
        max_errors: int = 5,
        window_ms: int = 60_000,
        cooldown_ms: int = 300_000,
    ):
        self.max_errors = max_errors
        self.window_ms = window_ms
        self.cooldown_ms = cooldown_ms
        self._errors: deque = deque()
        self._tripped_at: Optional[int] = None

    @property
    def name(self) -> str:
        return "api_error"

    def record_error(self, timestamp_ms: Optional[int] = None) -> None:
        now = timestamp_ms or utc_now_ms()
        cutoff = now - self.window_ms
        while self._errors and self._errors[0] < cutoff:
            self._errors.popleft()
        self._errors.append(now)
        if len(self._errors) >= self.max_errors:
            self._tripped_at = now

    def is_tripped(self) -> bool:
        if self._tripped_at is None:
            return False
        now = utc_now_ms()
        if now - self._tripped_at > self.cooldown_ms:
            self.reset()
            return False
        return True

    def reset(self) -> None:
        self._errors.clear()
        self._tripped_at = None

    def check(self) -> str:
        if self.is_tripped():
            return f"api error circuit tripped: {len(self._errors)} errors in window"
        return ""
