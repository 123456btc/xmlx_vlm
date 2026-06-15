"""日亏损上限规则."""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from xmlx_vlm.ai_trader.oms.constants import RiskDecisionType
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.core.portfolio import Portfolio
from xmlx_vlm.ai_trader.oms.core.trade import Trade
from xmlx_vlm.ai_trader.oms.interfaces.risk_engine import RiskContext, RiskDecision
from xmlx_vlm.ai_trader.oms.risk.rules.base import RiskRule
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO, HUNDRED
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms


class DailyLossRule(RiskRule):
    """限制单日已实现亏损占账户权益的比例."""

    def __init__(self, max_daily_loss_pct: Decimal = Decimal("3.0")):
        self.max_daily_loss_pct = to_decimal(max_daily_loss_pct)
        self._daily_realized_pnl = ZERO
        self._starting_equity = ZERO
        self._last_date = self._today()

    @property
    def name(self) -> str:
        return "daily_loss"

    def pre_trade(self, order: Order, context: RiskContext) -> RiskDecision:
        self._reset_if_new_day()
        equity = to_decimal(context.portfolio.account.equity)
        if equity <= ZERO:
            return RiskDecision(
                decision=RiskDecisionType.PASS,
                rule_name=self.name,
                reason="account equity unknown",
            )

        if self._starting_equity == ZERO:
            self._starting_equity = equity

        loss_pct = abs(self._daily_realized_pnl) / self._starting_equity * HUNDRED
        if loss_pct >= self.max_daily_loss_pct:
            return RiskDecision(
                decision=RiskDecisionType.REJECT,
                rule_name=self.name,
                reason=(
                    f"daily loss {loss_pct:.2f}% exceeds limit "
                    f"{self.max_daily_loss_pct:.2f}%"
                ),
                metadata={
                    "daily_realized_pnl": str(self._daily_realized_pnl),
                    "starting_equity": str(self._starting_equity),
                    "loss_pct": str(loss_pct),
                },
            )
        return RiskDecision(
            decision=RiskDecisionType.PASS,
            rule_name=self.name,
            reason="daily loss within limit",
            metadata={"loss_pct": str(loss_pct)},
        )

    def post_trade(self, trade: Trade, portfolio: Portfolio) -> Optional[RiskDecision]:
        self._reset_if_new_day()
        # trade 本身不直接携带 realized_pnl，这里用 portfolio 累计
        self._daily_realized_pnl = portfolio.total_realized_pnl()
        return None

    def reset(self):
        self._daily_realized_pnl = ZERO
        self._starting_equity = ZERO
        self._last_date = self._today()

    def _today(self) -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _reset_if_new_day(self):
        today = self._today()
        if today != self._last_date:
            self.reset()
