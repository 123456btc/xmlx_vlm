"""订单实体与状态机."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState, OrderType, TimeInForce
from xmlx_vlm.ai_trader.oms.exceptions import OrderStateError
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO, quantize_price, quantize_qty
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms


@dataclass
class Fill:
    """一次成交记录."""

    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    qty: Decimal
    price: Decimal
    fee: Decimal = field(default=ZERO)
    timestamp_ms: int = field(default_factory=utc_now_ms)
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        from xmlx_vlm.ai_trader.oms.utils.symbol import normalize_symbol
        if self.symbol:
            self.symbol = normalize_symbol(self.symbol)
        if isinstance(self.side, str):
            self.side = OrderSide(self.side.lower())
        self.qty = to_decimal(self.qty)
        self.price = to_decimal(self.price)
        self.fee = to_decimal(self.fee)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fill_id": self.fill_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "qty": str(self.qty),
            "price": str(self.price),
            "fee": str(self.fee),
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class Order:
    """订单实体."""

    symbol: str
    side: OrderSide
    qty: Decimal
    order_type: OrderType = OrderType.MARKET
    price: Optional[Decimal] = None
    stop_px: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    reduce_only: bool = False

    # 标识 (使用标准 32-char UUID hex，加上 0x 前缀正好符合 Hyperliquid 34-char 128-bit Cloid 规范)
    client_order_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    algo_id: Optional[str] = None

    # 状态
    state: OrderState = OrderState.DRAFT
    filled_qty: Decimal = field(default=ZERO)
    avg_fill_price: Decimal = field(default=ZERO)
    remaining_qty: Decimal = field(default=ZERO)

    # 元数据
    created_at_ms: int = field(default_factory=utc_now_ms)
    updated_at_ms: int = field(default_factory=utc_now_ms)
    exchange: Optional[str] = None
    raw_request: Optional[Dict[str, Any]] = None
    raw_response: Optional[Dict[str, Any]] = None
    reject_reason: Optional[str] = None
    fills: List[Fill] = field(default_factory=list)
    algo_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        from xmlx_vlm.ai_trader.oms.utils.symbol import normalize_symbol
        if self.symbol:
            self.symbol = normalize_symbol(self.symbol)
        self.qty = to_decimal(self.qty)
        self.remaining_qty = self.qty - self.filled_qty
        if self.price is not None:
            self.price = to_decimal(self.price)
        if self.stop_px is not None:
            self.stop_px = to_decimal(self.stop_px)
        if isinstance(self.side, str):
            self.side = OrderSide(self.side.lower())
        if isinstance(self.order_type, str):
            self.order_type = OrderType(self.order_type.lower())
        if isinstance(self.time_in_force, str):
            self.time_in_force = TimeInForce(self.time_in_force.upper())
        if self.remaining_qty < ZERO:
            self.remaining_qty = ZERO

    def notional(self) -> Decimal:
        """订单名义金额（按限价或平均成交价估算）."""
        px = self.price or self.avg_fill_price or ZERO
        return self.qty * px

    def filled_notional(self) -> Decimal:
        """已成交名义金额."""
        return self.filled_qty * self.avg_fill_price if self.avg_fill_price else ZERO

    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY

    def is_sell(self) -> bool:
        return self.side == OrderSide.SELL

    def is_done(self) -> bool:
        return self.state in {
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
        }

    def apply_fill(self, fill: Fill) -> None:
        """应用一次成交，更新状态机."""
        if self.is_done():
            raise OrderStateError(f"Cannot fill order in state {self.state}")

        fill_qty = to_decimal(fill.qty)
        fill_price = to_decimal(fill.price)

        if fill_qty <= ZERO:
            return

        # 更新均价
        total_cost = self.avg_fill_price * self.filled_qty + fill_price * fill_qty
        self.filled_qty += fill_qty
        if self.filled_qty > ZERO:
            self.avg_fill_price = total_cost / self.filled_qty
        self.remaining_qty = self.qty - self.filled_qty
        if self.remaining_qty < ZERO:
            self.remaining_qty = ZERO

        self.fills.append(fill)
        self.updated_at_ms = utc_now_ms()

        if self.remaining_qty == ZERO:
            self.state = OrderState.FILLED
        else:
            self.state = OrderState.PARTIAL_FILLED

    def transition_to(self, new_state: OrderState, reason: Optional[str] = None) -> None:
        """状态转换，非法转换抛出 OrderStateError."""
        allowed = _STATE_TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise OrderStateError(
                f"Invalid transition from {self.state.value} to {new_state.value}"
            )
        self.state = new_state
        if reason:
            self.reject_reason = reason
        self.updated_at_ms = utc_now_ms()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "order_id": self.order_id,
            "parent_order_id": self.parent_order_id,
            "algo_id": self.algo_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "qty": str(self.qty),
            "price": str(self.price) if self.price is not None else None,
            "stop_px": str(self.stop_px) if self.stop_px is not None else None,
            "time_in_force": self.time_in_force.value,
            "reduce_only": self.reduce_only,
            "state": self.state.value,
            "filled_qty": str(self.filled_qty),
            "avg_fill_price": str(self.avg_fill_price),
            "remaining_qty": str(self.remaining_qty),
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
            "exchange": self.exchange,
            "reject_reason": self.reject_reason,
            "algo_params": self.algo_params,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Order":
        return cls(
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["order_type"]),
            qty=Decimal(data["qty"]),
            price=Decimal(data["price"]) if data.get("price") else None,
            stop_px=Decimal(data["stop_px"]) if data.get("stop_px") else None,
            time_in_force=TimeInForce(data["time_in_force"]),
            reduce_only=bool(data.get("reduce_only", False)),
            client_order_id=data.get("client_order_id", uuid.uuid4().hex),
            order_id=data.get("order_id"),
            parent_order_id=data.get("parent_order_id"),
            algo_id=data.get("algo_id"),
            state=OrderState(data.get("state", "draft")),
            filled_qty=Decimal(data.get("filled_qty", "0")),
            avg_fill_price=Decimal(data.get("avg_fill_price", "0")),
            created_at_ms=data.get("created_at_ms", utc_now_ms()),
            updated_at_ms=data.get("updated_at_ms", utc_now_ms()),
            exchange=data.get("exchange"),
            reject_reason=data.get("reject_reason"),
            algo_params=data.get("algo_params", {}),
        )


# 合法状态转换表
_STATE_TRANSITIONS = {
    OrderState.DRAFT: {
        OrderState.PRE_TRADE_OK,
        OrderState.SUBMITTED,
        OrderState.SENT,
        OrderState.ACKNOWLEDGED,
        OrderState.FILLED,
        OrderState.REJECTED,
        OrderState.CANCELLED,
    },
    OrderState.PRE_TRADE_OK: {
        OrderState.SENT,
        OrderState.SUBMITTED,
        OrderState.ACKNOWLEDGED,
        OrderState.FILLED,
        OrderState.REJECTED,
        OrderState.CANCELLED,
    },
    OrderState.SENT: {
        OrderState.SUBMITTED,
        OrderState.ACKNOWLEDGED,
        OrderState.FILLED,
        OrderState.PARTIAL_FILLED,
        OrderState.REJECTED,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
    },
    OrderState.SUBMITTED: {
        OrderState.ACKNOWLEDGED,
        OrderState.FILLED,
        OrderState.PARTIAL_FILLED,
        OrderState.REJECTED,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
    },
    OrderState.ACKNOWLEDGED: {
        OrderState.PARTIAL_FILLED,
        OrderState.FILLED,
        OrderState.CANCEL_REQUESTED,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
    },
    OrderState.PARTIAL_FILLED: {
        OrderState.FILLED,
        OrderState.CANCEL_REQUESTED,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
    },
    OrderState.CANCEL_REQUESTED: {OrderState.CANCELLED, OrderState.FILLED, OrderState.PARTIAL_FILLED},
    OrderState.FILLED: set(),
    OrderState.REJECTED: set(),
    OrderState.CANCELLED: set(),
    OrderState.EXPIRED: set(),
}
