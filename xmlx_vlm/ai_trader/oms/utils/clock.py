# SPDX-License-Identifier: Apache-2.0
"""
ClockProvider — Deterministic Monotonic Time Provider.

Enforces Rule 09 of Engineering Constitution:
Decouples execution, cooldown, and backtesting logic from the OS wall-clock.
Provides deterministic monotonic ticks for simulations, replay, and live trading.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional


class IClock(ABC):
    """Abstract interface for time providers."""

    @abstractmethod
    def now_ms(self) -> int:
        """Return current timestamp in milliseconds."""
        raise NotImplementedError

    @abstractmethod
    def now_iso(self) -> str:
        """Return current timestamp in ISO 8601 string."""
        raise NotImplementedError

    @abstractmethod
    def monotonic(self) -> float:
        """Return monotonic elapsed seconds (for interval calculations)."""
        raise NotImplementedError


class RealtimeClock(IClock):
    """Production live clock utilizing monotonic timer and UTC system time."""

    def now_ms(self) -> int:
        return int(time.time() * 1000)

    def now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def monotonic(self) -> float:
        return time.monotonic()


class VirtualClock(IClock):
    """Deterministic virtual clock for backtesting, step replay, and property testing."""

    def __init__(self, start_ms: int = 1700000000000):
        self._current_ms: int = start_ms
        self._monotonic_offset: float = 0.0

    def set_time_ms(self, ts_ms: int) -> None:
        """Advance or set the virtual timestamp."""
        self._current_ms = ts_ms

    def advance_ms(self, delta_ms: int) -> None:
        """Advance time by delta milliseconds."""
        self._current_ms += delta_ms
        self._monotonic_offset += (delta_ms / 1000.0)

    def now_ms(self) -> int:
        return self._current_ms

    def now_iso(self) -> str:
        return datetime.fromtimestamp(self._current_ms / 1000.0, tz=timezone.utc).isoformat()

    def monotonic(self) -> float:
        return self._monotonic_offset


# Global clock provider instance (Default: RealtimeClock)
_GLOBAL_CLOCK: IClock = RealtimeClock()


def get_clock() -> IClock:
    """Get the active global clock instance."""
    return _GLOBAL_CLOCK


def set_clock(clock: IClock) -> None:
    """Set the active global clock instance (e.g. for testing or backtesting)."""
    global _GLOBAL_CLOCK
    _GLOBAL_CLOCK = clock


def reset_clock() -> None:
    """Reset clock provider back to default RealtimeClock."""
    global _GLOBAL_CLOCK
    _GLOBAL_CLOCK = RealtimeClock()
