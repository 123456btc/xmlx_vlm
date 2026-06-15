"""持仓实体."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional

from xmlx_vlm.ai_trader.oms.constants import PositionSide
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO, HUNDRED
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms


@dataclass
class Position:
    """单个交易品种持仓."""

    symbol: str
    side: PositionSide
    qty: Decimal = field(default=ZERO)
    avg_entry_price: Decimal = field(default=ZERO)
    realized_pnl: Decimal = field(default=ZERO)
    unrealized_pnl: Decimal = field(default=ZERO)
    mark_price: Decimal = field(default=ZERO)
    leverage: int = field(default=1)
    margin_type: str = field(default="cross")
    liq_price: Decimal = field(default=ZERO)
    updated_at_ms: int = field(default_factory=utc_now_ms)

    def __post_init__(self):
        self.qty = to_decimal(self.qty)
        self.avg_entry_price = to_decimal(self.avg_entry_price)
        self.realized_pnl = to_decimal(self.realized_pnl)
        self.unrealized_pnl = to_decimal(self.unrealized_pnl)
        self.mark_price = to_decimal(self.mark_price)
        self.liq_price = to_decimal(self.liq_price)
        if isinstance(self.side, str):
            self.side = PositionSide(self.side.lower())
        if self.qty == ZERO:
            self.side = PositionSide.FLAT

    def is_long(self) -> bool:
        return self.side == PositionSide.LONG

    def is_short(self) -> bool:
        return self.side == PositionSide.SHORT

    def is_flat(self) -> bool:
        return self.side == PositionSide.FLAT or self.qty == ZERO

    def notional(self) -> Decimal:
        """按标记价计算的名义持仓价值."""
        return self.qty * self.mark_price

    def update_mark_price(self, mark_price: Decimal) -> None:
        """更新标记价并重新计算未实现盈亏."""
        self.mark_price = to_decimal(mark_price)
        if self.is_flat():
            self.unrealized_pnl = ZERO
            return
        if self.avg_entry_price == ZERO:
            self.unrealized_pnl = ZERO
            return
        if self.is_long():
            self.unrealized_pnl = (self.mark_price - self.avg_entry_price) * self.qty
        else:
            self.unrealized_pnl = (self.avg_entry_price - self.mark_price) * self.qty

    def increase(self, qty: Decimal, price: Decimal) -> None:
        """增加持仓（同向加仓）."""
        qty = to_decimal(qty)
        price = to_decimal(price)
        if qty <= ZERO:
            return
        incoming_side = PositionSide.LONG if self.side in (PositionSide.LONG, PositionSide.FLAT) else PositionSide.SHORT
        if self.side == PositionSide.FLAT:
            self.side = incoming_side

        total_cost = self.avg_entry_price * self.qty + price * qty
        self.qty += qty
        if self.qty > ZERO:
            self.avg_entry_price = total_cost / self.qty
        self.updated_at_ms = utc_now_ms()

    def decrease(self, qty: Decimal, price: Decimal) -> Decimal:
        """减少持仓，返回已实现盈亏."""
        qty = to_decimal(qty)
        price = to_decimal(price)
        if qty <= ZERO:
            return ZERO
        close_qty = min(qty, self.qty)
        if close_qty == ZERO:
            return ZERO

        if self.is_long():
            pnl = (price - self.avg_entry_price) * close_qty
        else:
            pnl = (self.avg_entry_price - price) * close_qty

        self.realized_pnl += pnl
        self.qty -= close_qty
        if self.qty == ZERO:
            self.side = PositionSide.FLAT
            self.avg_entry_price = ZERO
        self.updated_at_ms = utc_now_ms()
        return pnl

    def apply_fill(self, side: str, qty: Decimal, price: Decimal) -> None:
        """根据成交方向与数量更新持仓.

        - 同向：加仓
        - 反向且 qty <= 当前持仓：减仓
        - 反向且 qty > 当前持仓：先平仓再反向开仓
        """
        qty = to_decimal(qty)
        price = to_decimal(price)
        fill_side = PositionSide.LONG if side.lower() == "buy" else PositionSide.SHORT

        if self.is_flat():
            self.side = fill_side
            self.increase(qty, price)
            return

        if fill_side == self.side:
            self.increase(qty, price)
            return

        # 反向成交
        if qty <= self.qty:
            self.decrease(qty, price)
        else:
            remaining = qty - self.qty
            self.decrease(self.qty, price)
            self.side = fill_side
            self.increase(remaining, price)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "qty": str(self.qty),
            "avg_entry_price": str(self.avg_entry_price),
            "mark_price": str(self.mark_price),
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "leverage": self.leverage,
            "margin_type": self.margin_type,
            "liq_price": str(self.liq_price),
            "updated_at_ms": self.updated_at_ms,
        }
