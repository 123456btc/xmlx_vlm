"""账户快照实体."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict

from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms


@dataclass
class AccountSnapshot:
    """账户权益与保证金快照."""

    equity: Decimal = field(default=ZERO)
    available_margin: Decimal = field(default=ZERO)
    used_margin: Decimal = field(default=ZERO)
    total_position_value: Decimal = field(default=ZERO)
    cash: Decimal = field(default=ZERO)
    timestamp_ms: int = field(default_factory=utc_now_ms)
    mode: Optional[str] = field(default=None)
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.equity = to_decimal(self.equity)
        self.available_margin = to_decimal(self.available_margin)
        self.used_margin = to_decimal(self.used_margin)
        self.total_position_value = to_decimal(self.total_position_value)
        self.cash = to_decimal(self.cash)

    def margin_utilization_pct(self) -> Decimal:
        if self.equity == ZERO:
            return ZERO
        return self.used_margin / self.equity * Decimal("100")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "equity": str(self.equity),
            "available_margin": str(self.available_margin),
            "used_margin": str(self.used_margin),
            "total_position_value": str(self.total_position_value),
            "cash": str(self.cash),
            "timestamp_ms": self.timestamp_ms,
            "mode": self.mode,
        }
