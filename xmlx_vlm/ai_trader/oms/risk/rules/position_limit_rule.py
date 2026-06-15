"""仓位上限规则."""

from __future__ import annotations

from decimal import Decimal

from xmlx_vlm.ai_trader.oms.constants import RiskDecisionType
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.interfaces.risk_engine import RiskContext, RiskDecision
from xmlx_vlm.ai_trader.oms.risk.rules.base import RiskRule
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO, HUNDRED


class PositionLimitRule(RiskRule):
    """限制单品种与总仓位占账户权益的比例."""

    def __init__(
        self,
        max_single_position_pct: Decimal = Decimal("20.0"),
        max_total_position_pct: Decimal = Decimal("50.0"),
    ):
        self.max_single_position_pct = to_decimal(max_single_position_pct)
        self.max_total_position_pct = to_decimal(max_total_position_pct)

    @property
    def name(self) -> str:
        return "position_limit"

    def pre_trade(self, order: Order, context: RiskContext) -> RiskDecision:
        equity = to_decimal(context.portfolio.account.equity)
        if equity <= ZERO:
            return RiskDecision(
                decision=RiskDecisionType.PASS,
                rule_name=self.name,
                reason="account equity unknown",
            )

        order_notional = order.notional()
        current_single = context.portfolio.position_notional(order.symbol)
        current_total = context.portfolio.gross_exposure()

        new_single = current_single + order_notional
        new_total = current_total + order_notional

        single_pct = new_single / equity * HUNDRED
        total_pct = new_total / equity * HUNDRED

        if single_pct > self.max_single_position_pct:
            return RiskDecision(
                decision=RiskDecisionType.REJECT,
                rule_name=self.name,
                reason=(
                    f"{order.symbol} position {single_pct:.2f}% exceeds "
                    f"single limit {self.max_single_position_pct:.2f}%"
                ),
                metadata={
                    "symbol": order.symbol,
                    "single_pct": str(single_pct),
                    "total_pct": str(total_pct),
                },
            )

        if total_pct > self.max_total_position_pct:
            return RiskDecision(
                decision=RiskDecisionType.REJECT,
                rule_name=self.name,
                reason=(
                    f"total position {total_pct:.2f}% exceeds "
                    f"limit {self.max_total_position_pct:.2f}%"
                ),
                metadata={
                    "total_pct": str(total_pct),
                    "single_pct": str(single_pct),
                },
            )

        return RiskDecision(
            decision=RiskDecisionType.PASS,
            rule_name=self.name,
            reason="position within limits",
            metadata={"single_pct": str(single_pct), "total_pct": str(total_pct)},
        )
