"""日亏损熔断器."""

from __future__ import annotations

from decimal import Decimal

from xmlx_vlm.ai_trader.oms.circuit.breaker import CircuitBreaker
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO, HUNDRED


class DailyLossCircuitBreaker(CircuitBreaker):
    """当日亏损达到阈值时熔断."""

    def __init__(self, max_daily_loss_pct: Decimal = Decimal("3.0")):
        self.max_daily_loss_pct = to_decimal(max_daily_loss_pct)
        self._tripped = False
        self._reason = ""

    @property
    def name(self) -> str:
        return "daily_loss"

    def update(self, starting_equity: Decimal, current_equity: Decimal) -> None:
        if starting_equity <= ZERO:
            return
        drawdown_pct = (starting_equity - current_equity) / starting_equity * HUNDRED
        if drawdown_pct >= self.max_daily_loss_pct:
            self._tripped = True
            self._reason = (
                f"daily drawdown {drawdown_pct:.2f}% exceeds "
                f"{self.max_daily_loss_pct:.2f}%"
            )

    def is_tripped(self) -> bool:
        return self._tripped

    def reset(self) -> None:
        self._tripped = False
        self._reason = ""

    def check(self) -> str:
        if self._tripped:
            return self._reason
        return ""
