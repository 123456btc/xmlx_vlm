"""急停开关."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from xmlx_vlm.ai_trader.oms.events.bus import EventBus
from xmlx_vlm.ai_trader.oms.events.types import KillSwitchEvent

logger = logging.getLogger(__name__)


class KillSwitch:
    """全局急停开关.

    触发后：
    - 锁定所有新订单
    - 可选市价平掉所有持仓
    - 发布 KillSwitchEvent
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        self._locked = False
        self._reason: str = ""
        self._triggered_by: str = ""
        self._event_bus = event_bus
        self._handlers: List[Callable[[KillSwitchEvent], Any]] = []

    @property
    def is_locked(self) -> bool:
        return self._locked

    @property
    def reason(self) -> str:
        return self._reason

    def trigger(
        self,
        triggered_by: str,
        reason: str,
        flatten_positions: bool = True,
    ) -> None:
        if self._locked:
            return
        self._locked = True
        self._triggered_by = triggered_by
        self._reason = reason
        event = KillSwitchEvent(
            triggered_by=triggered_by,
            reason=reason,
            flatten_positions=flatten_positions,
        )
        logger.critical(
            "KILL SWITCH triggered by %s: %s (flatten=%s)",
            triggered_by,
            reason,
            flatten_positions,
        )
        if self._event_bus:
            self._event_bus.publish(event)
        for handler in self._handlers:
            try:
                handler(event)
            except Exception:
                logger.exception("kill switch handler failed")

    def reset(self, reset_by: str) -> None:
        self._locked = False
        self._reason = ""
        self._triggered_by = ""
        logger.warning("KILL SWITCH reset by %s", reset_by)

    def add_handler(self, handler: Callable[[KillSwitchEvent], Any]) -> None:
        self._handlers.append(handler)

    def check(self) -> str:
        if self._locked:
            return f"kill switch active: {self._reason}"
        return ""
