"""连续亏损熔断器."""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from xmlx_vlm.ai_trader.oms.circuit.breaker import CircuitBreaker
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO


class ConsecutiveLossCircuitBreaker(CircuitBreaker):
    """连续 N 笔亏损达到阈值时熔断."""

    def __init__(
        self,
        max_consecutive_losses: int = 3,
        min_loss_amount: Optional[Decimal] = None,
    ):
        self.max_consecutive_losses = max_consecutive_losses
        self.min_loss_amount = to_decimal(min_loss_amount) if min_loss_amount else ZERO
        self._consecutive_losses = 0
        self._tripped = False

    @property
    def name(self) -> str:
        return "consecutive_loss"

    def record_trade_pnl(self, realized_pnl: Decimal) -> None:
        pnl = to_decimal(realized_pnl)
        if pnl < -self.min_loss_amount:
            self._consecutive_losses += 1
        elif pnl > ZERO:
            self._consecutive_losses = 0

        if self._consecutive_losses >= self.max_consecutive_losses:
            self._tripped = True

    def is_tripped(self) -> bool:
        return self._tripped

    def reset(self) -> None:
        self._consecutive_losses = 0
        self._tripped = False

    def check(self) -> str:
        if self._tripped:
            return f"consecutive loss circuit tripped: {self._consecutive_losses} losses"
        return ""
