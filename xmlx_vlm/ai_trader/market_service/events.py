"""事件总线与行情事件定义.

采用发布-订阅模型：行情服务生产事件，AI Agent / 策略订阅事件。
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .models import BookSnapshot, FundingRate, OISnapshot, Tick, Trade

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketEvent:
    """所有行情事件的基类."""

    symbol: str
    timestamp_ms: int


@dataclass(frozen=True)
class PriceUpdateEvent(MarketEvent):
    price: float
    source: str = "mark"  # "mark" | "oracle" | "mid"


@dataclass(frozen=True)
class BookUpdateEvent(MarketEvent):
    book: BookSnapshot


@dataclass(frozen=True)
class TradeEvent(MarketEvent):
    trade: Trade


@dataclass(frozen=True)
class FundingUpdateEvent(MarketEvent):
    funding: FundingRate


@dataclass(frozen=True)
class OIUpdateEvent(MarketEvent):
    oi: OISnapshot


@dataclass(frozen=True)
class BarClosedEvent(MarketEvent):
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class IndicatorAlertEvent(MarketEvent):
    """技术指标触发警报，例如突破、OI 异动、Funding 反转等."""

    alert_type: str
    payload: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ConnectionStateEvent:
    """WebSocket 连接状态事件，不含 symbol."""

    state: str  # "connecting" | "connected" | "disconnected" | "error"
    message: str = ""
    timestamp_ms: int = 0


EventHandler = Callable[[object], None]


class EventBus:
    """线程安全的事件总线.

    支持按事件类型订阅，也支持通配符 ``object`` 订阅所有事件。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handlers: Dict[type, List[EventHandler]] = {}

    def subscribe(self, event_type: type, handler: EventHandler) -> None:
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: type, handler: EventHandler) -> None:
        with self._lock:
            handlers = self._handlers.get(event_type, [])
            if handler in handlers:
                handlers.remove(handler)

    def publish(self, event: object) -> None:
        """同步调用所有匹配的处理函数."""
        with self._lock:
            # 按精确类型 + 所有事件的通配符订阅分发
            candidates: List[EventHandler] = []
            candidates.extend(self._handlers.get(type(event), []))
            candidates.extend(self._handlers.get(object, []))
        for handler in candidates:
            try:
                handler(event)
            except Exception:
                logger.exception("Event handler failed for %s", type(event).__name__)
