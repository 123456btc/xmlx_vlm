"""事件总线实现."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Type

from xmlx_vlm.ai_trader.oms.events.types import BaseEvent

logger = logging.getLogger(__name__)

EventHandler = Callable[[BaseEvent], Any]


class EventBus:
    """事件总线抽象基类."""

    def subscribe(self, event_type: Type[BaseEvent], handler: EventHandler) -> None:
        raise NotImplementedError

    def publish(self, event: BaseEvent) -> None:
        raise NotImplementedError

    def unsubscribe(self, event_type: Type[BaseEvent], handler: EventHandler) -> None:
        raise NotImplementedError


class SyncEventBus(EventBus):
    """同步事件总线：发布即在同一线程顺序调用处理器."""

    def __init__(self):
        self._handlers: Dict[Type[BaseEvent], List[EventHandler]] = defaultdict(list)

    def subscribe(self, event_type: Type[BaseEvent], handler: EventHandler) -> None:
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: Type[BaseEvent], handler: EventHandler) -> None:
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def publish(self, event: BaseEvent) -> None:
        event_type = type(event)
        handlers = self._handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception:
                logger.exception("Event handler failed for %s", event_type.__name__)

    def clear(self) -> None:
        self._handlers.clear()
