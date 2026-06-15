"""事件类型定义."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional

from xmlx_vlm.ai_trader.oms.constants import EventType, OrderSide, OrderState, RiskDecisionType
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms


@dataclass
class BaseEvent:
    """事件基类."""

    event_type: EventType
    timestamp_ms: int = field(default_factory=utc_now_ms)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderEvent(BaseEvent):
    """订单生命周期事件."""

    client_order_id: str = ""
    order_id: Optional[str] = None
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    state: OrderState = OrderState.DRAFT
    qty: Decimal = field(default=ZERO)
    filled_qty: Decimal = field(default=ZERO)
    price: Optional[Decimal] = None
    reason: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.qty = to_decimal(self.qty)
        self.filled_qty = to_decimal(self.filled_qty)
        if self.price is not None:
            self.price = to_decimal(self.price)
        if isinstance(self.side, str):
            self.side = OrderSide(self.side.lower())
        if isinstance(self.state, str):
            self.state = OrderState(self.state)


@dataclass
class FillEvent(BaseEvent):
    """成交事件."""

    fill_id: str = ""
    order_id: str = ""
    client_order_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    qty: Decimal = field(default=ZERO)
    price: Decimal = field(default=ZERO)
    fee: Decimal = field(default=ZERO)
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.qty = to_decimal(self.qty)
        self.price = to_decimal(self.price)
        self.fee = to_decimal(self.fee)
        if isinstance(self.side, str):
            self.side = OrderSide(self.side.lower())


@dataclass
class RiskEvent(BaseEvent):
    """风控事件."""

    decision: RiskDecisionType = RiskDecisionType.PASS
    rule_name: str = ""
    client_order_id: Optional[str] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.decision, str):
            self.decision = RiskDecisionType(self.decision)


@dataclass
class CircuitEvent(BaseEvent):
    """熔断事件."""

    event_type: EventType = EventType.CIRCUIT_TRIPPED
    circuit_name: str = ""
    reason: str = ""
    reset_after_ms: Optional[int] = None


@dataclass
class KillSwitchEvent(BaseEvent):
    """急停事件."""

    event_type: EventType = EventType.KILL_SWITCH_TRIGGERED
    triggered_by: str = ""
    reason: str = ""
    flatten_positions: bool = True


@dataclass
class PortfolioEvent(BaseEvent):
    """仓位/账户同步事件."""

    event_subtype: str = "sync"
    data: Dict[str, Any] = field(default_factory=dict)
