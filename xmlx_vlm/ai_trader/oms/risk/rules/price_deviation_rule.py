"""价格偏离规则."""

from __future__ import annotations

from decimal import Decimal

from xmlx_vlm.ai_trader.oms.constants import OrderType, RiskDecisionType
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.interfaces.risk_engine import RiskContext, RiskDecision
from xmlx_vlm.ai_trader.oms.risk.rules.base import RiskRule
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO, HUNDRED, pct_change


class PriceDeviationRule(RiskRule):
    """限制下单价格相对标记价/ oracle 价的偏离幅度，防止模型幻觉价格."""

    def __init__(self, max_price_deviation_pct: Decimal = Decimal("1.0")):
        self.max_price_deviation_pct = to_decimal(max_price_deviation_pct)

    @property
    def name(self) -> str:
        return "price_deviation"

    def pre_trade(self, order: Order, context: RiskContext) -> RiskDecision:
        if order.order_type == OrderType.MARKET:
            return RiskDecision(
                decision=RiskDecisionType.PASS,
                rule_name=self.name,
                reason="market order skip price deviation check",
            )

        if order.price is None:
            return RiskDecision(
                decision=RiskDecisionType.PASS,
                rule_name=self.name,
                reason="no price provided",
            )

        reference = context.mark_price or context.oracle_price
        if reference is None:
            return RiskDecision(
                decision=RiskDecisionType.WARNING,
                rule_name=self.name,
                reason="no reference price available",
            )

        reference = to_decimal(reference)
        if reference <= ZERO:
            return RiskDecision(
                decision=RiskDecisionType.WARNING,
                rule_name=self.name,
                reason="reference price invalid",
            )

        deviation = abs(pct_change(order.price, reference))
        if deviation > self.max_price_deviation_pct:
            return RiskDecision(
                decision=RiskDecisionType.REJECT,
                rule_name=self.name,
                reason=(
                    f"order price {order.price} deviates {deviation:.2f}% "
                    f"from reference {reference} (limit {self.max_price_deviation_pct:.2f}%)"
                ),
                metadata={
                    "order_price": str(order.price),
                    "reference_price": str(reference),
                    "deviation_pct": str(deviation),
                },
            )
        return RiskDecision(
            decision=RiskDecisionType.PASS,
            rule_name=self.name,
            reason="price deviation within limit",
            metadata={"deviation_pct": str(deviation)},
        )
