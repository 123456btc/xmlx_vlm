"""执行算法基类与 ParentOrder."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms


@dataclass
class ParentOrder:
    """算法单（父订单）."""

    symbol: str
    side: OrderSide
    total_qty: Decimal
    algo_type: str  # twap / vwap / pov / is / iceberg / sniping / liquidity_seek
    params: Dict[str, Any] = field(default_factory=dict)
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    state: OrderState = OrderState.DRAFT
    child_orders: List[Order] = field(default_factory=list)
    filled_qty: Decimal = field(default=ZERO)
    avg_fill_price: Decimal = field(default=ZERO)
    remaining_qty: Decimal = field(default=ZERO)
    created_at_ms: int = field(default_factory=utc_now_ms)
    updated_at_ms: int = field(default_factory=utc_now_ms)
    reject_reason: Optional[str] = None

    def __post_init__(self):
        self.symbol = self.symbol.upper()
        self.total_qty = abs(to_decimal(self.total_qty))
        self.remaining_qty = self.total_qty - self.filled_qty
        if self.remaining_qty < ZERO:
            self.remaining_qty = ZERO
        if isinstance(self.side, str):
            self.side = OrderSide(self.side.lower())

    def apply_child_fill(self, child: Order) -> None:
        """汇总 child 成交到 parent."""
        if child.filled_qty <= ZERO:
            return
        total_cost = self.avg_fill_price * self.filled_qty + child.avg_fill_price * child.filled_qty
        self.filled_qty += child.filled_qty
        self.remaining_qty = self.total_qty - self.filled_qty
        if self.filled_qty > ZERO:
            self.avg_fill_price = total_cost / self.filled_qty
        if self.remaining_qty <= ZERO:
            self.state = OrderState.FILLED
        elif self.filled_qty > ZERO:
            self.state = OrderState.PARTIAL_FILLED
        self.updated_at_ms = utc_now_ms()

    def is_done(self) -> bool:
        return self.state in {
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "total_qty": str(self.total_qty),
            "algo_type": self.algo_type,
            "params": self.params,
            "state": self.state.value,
            "filled_qty": str(self.filled_qty),
            "avg_fill_price": str(self.avg_fill_price),
            "remaining_qty": str(self.remaining_qty),
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
            "reject_reason": self.reject_reason,
        }


class ExecutionAlgorithm(ABC):
    """执行算法抽象基类."""

    def __init__(self, algo_id: Optional[str] = None):
        self._algo_id = algo_id or uuid.uuid4().hex[:16]
        self._parent: Optional[ParentOrder] = None
        self._router: Optional[Any] = None
        self._cancelled = False

    @property
    def algo_id(self) -> str:
        return self._algo_id

    @property
    @abstractmethod
    def name(self) -> str:
        """算法名称."""
        ...

    @abstractmethod
    async def start(
        self,
        parent: ParentOrder,
        router: Any,
        on_child_update: Optional[Any] = None,
    ) -> None:
        """启动算法."""
        ...

    def cancel(self) -> None:
        """请求取消算法."""
        self._cancelled = True

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def is_done(self) -> bool:
        if self._parent is None:
            return False
        return self._parent.is_done() or self._cancelled

    def _create_child(
        self,
        qty: Decimal,
        price: Optional[Decimal] = None,
        order_type: str = "limit",
        time_in_force: str = "GTC",
        urgency: str = "normal",
    ) -> Order:
        """生成一张 child order."""
        assert self._parent is not None
        child = Order(
            symbol=self._parent.symbol,
            side=self._parent.side,
            qty=to_decimal(qty),
            order_type=order_type,
            price=to_decimal(price) if price is not None else None,
            time_in_force=time_in_force,
            parent_order_id=self._parent.order_id,
            algo_id=self._algo_id,
        )
        child.algo_params["urgency"] = urgency
        return child
