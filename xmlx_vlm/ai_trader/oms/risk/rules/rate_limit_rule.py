"""下单频率限制规则."""

from __future__ import annotations

from collections import deque
from decimal import Decimal
from typing import Optional

from xmlx_vlm.ai_trader.oms.constants import RiskDecisionType
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.interfaces.risk_engine import RiskContext, RiskDecision
from xmlx_vlm.ai_trader.oms.risk.rules.base import RiskRule
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms


class RateLimitRule(RiskRule):
    """滑动窗口限制单位时间下单次数."""

    def __init__(
        self,
        max_orders_per_minute: int = 12,
        max_orders_per_second: int = 3,
    ):
        self.max_orders_per_minute = max_orders_per_minute
        self.max_orders_per_second = max_orders_per_second
        self._minute_window: deque = deque()
        self._second_window: deque = deque()

    @property
    def name(self) -> str:
        return "rate_limit"

    def pre_trade(self, order: Order, context: RiskContext) -> RiskDecision:
        now = utc_now_ms()
        cutoff_minute = now - 60_000
        cutoff_second = now - 1_000

        # 清理过期时间戳
        while self._minute_window and self._minute_window[0] < cutoff_minute:
            self._minute_window.popleft()
        while self._second_window and self._second_window[0] < cutoff_second:
            self._second_window.popleft()

        if len(self._second_window) >= self.max_orders_per_second:
            return RiskDecision(
                decision=RiskDecisionType.REJECT,
                rule_name=self.name,
                reason=(
                    f"order rate {len(self._second_window)}/s exceeds "
                    f"limit {self.max_orders_per_second}"
                ),
                metadata={"orders_last_second": len(self._second_window)},
            )

        if len(self._minute_window) >= self.max_orders_per_minute:
            return RiskDecision(
                decision=RiskDecisionType.REJECT,
                rule_name=self.name,
                reason=(
                    f"order rate {len(self._minute_window)}/min exceeds "
                    f"limit {self.max_orders_per_minute}"
                ),
                metadata={"orders_last_minute": len(self._minute_window)},
            )

        self._minute_window.append(now)
        self._second_window.append(now)
        return RiskDecision(
            decision=RiskDecisionType.PASS,
            rule_name=self.name,
            reason="rate within limits",
            metadata={
                "orders_last_second": len(self._second_window),
                "orders_last_minute": len(self._minute_window),
            },
        )

    def reset(self):
        self._minute_window.clear()
        self._second_window.clear()
