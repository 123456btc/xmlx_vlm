"""保证金规则."""

from __future__ import annotations

from decimal import Decimal

from xmlx_vlm.ai_trader.oms.constants import RiskDecisionType
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.interfaces.risk_engine import RiskContext, RiskDecision
from xmlx_vlm.ai_trader.oms.risk.rules.base import RiskRule
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO, HUNDRED


class MarginRule(RiskRule):
    """限制下单后最低可用保证金比例."""

    def __init__(self, min_available_margin_pct: Decimal = Decimal("20.0")):
        self.min_available_margin_pct = to_decimal(min_available_margin_pct)

    @property
    def name(self) -> str:
        return "margin"

    def pre_trade(self, order: Order, context: RiskContext) -> RiskDecision:
        equity = to_decimal(context.portfolio.account.equity)
        available = to_decimal(context.portfolio.account.available_margin)
        if equity <= ZERO:
            return RiskDecision(
                decision=RiskDecisionType.PASS,
                rule_name=self.name,
                reason="account equity unknown",
            )

        order_notional = order.notional()
        projected_available = available - order_notional
        projected_available_pct = projected_available / equity * HUNDRED

        if projected_available_pct < self.min_available_margin_pct:
            return RiskDecision(
                decision=RiskDecisionType.REJECT,
                rule_name=self.name,
                reason=(
                    f"projected available margin {projected_available_pct:.2f}% "
                    f"below limit {self.min_available_margin_pct:.2f}%"
                ),
                metadata={
                    "available_margin": str(available),
                    "projected_available": str(projected_available),
                    "projected_available_pct": str(projected_available_pct),
                },
            )
        return RiskDecision(
            decision=RiskDecisionType.PASS,
            rule_name=self.name,
            reason="margin sufficient",
            metadata={"projected_available_pct": str(projected_available_pct)},
        )
