"""成交实体."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict

from xmlx_vlm.ai_trader.oms.constants import OrderSide
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms


@dataclass
class Trade:
    """标准化成交记录."""

    trade_id: str
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    qty: Decimal
    price: Decimal
    fee: Decimal = field(default=ZERO)
    timestamp_ms: int = field(default_factory=utc_now_ms)
    exchange: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        from xmlx_vlm.ai_trader.oms.utils.symbol import normalize_symbol
        if self.symbol:
            self.symbol = normalize_symbol(self.symbol)
        self.qty = to_decimal(self.qty)
        self.price = to_decimal(self.price)
        self.fee = to_decimal(self.fee)
        if isinstance(self.side, str):
            self.side = OrderSide(self.side.lower())

    def notional(self) -> Decimal:
        return self.qty * self.price

    def net_proceeds(self) -> Decimal:
        """扣除手续费后的净额."""
        return self.notional() - self.fee

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "qty": str(self.qty),
            "price": str(self.price),
            "fee": str(self.fee),
            "timestamp_ms": self.timestamp_ms,
            "exchange": self.exchange,
        }
