"""订单金额与数量限制规则."""

from __future__ import annotations

from decimal import Decimal

from xmlx_vlm.ai_trader.oms.constants import RiskDecisionType
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.interfaces.risk_engine import RiskContext, RiskDecision
from xmlx_vlm.ai_trader.oms.risk.rules.base import RiskRule
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO


class OrderSizeRule(RiskRule):
    """限制单笔订单名义金额与最小数量."""

    def __init__(
        self,
        max_single_order_notional: Decimal = Decimal("5000"),
        min_order_notional: Decimal = Decimal("10"),
    ):
        self.max_single_order_notional = to_decimal(max_single_order_notional)
        self.min_order_notional = to_decimal(min_order_notional)

    @property
    def name(self) -> str:
        return "order_size"

    def pre_trade(self, order: Order, context: RiskContext) -> RiskDecision:
        notional = order.notional()
        # 市价单无 price 时，用 mark/oracle 价格估算名义金额
        if notional <= ZERO and context.mark_price is not None:
            notional = order.qty * context.mark_price
        if notional <= ZERO and context.oracle_price is not None:
            notional = order.qty * context.oracle_price
        if notional <= ZERO:
            return RiskDecision(
                decision=RiskDecisionType.REJECT,
                rule_name=self.name,
                reason="order notional must be positive",
                metadata={"notional": str(notional)},
            )
        if notional > self.max_single_order_notional:
            return RiskDecision(
                decision=RiskDecisionType.REJECT,
                rule_name=self.name,
                reason=(
                    f"order notional {notional} exceeds max "
                    f"{self.max_single_order_notional}"
                ),
                metadata={"notional": str(notional)},
            )
        if notional < self.min_order_notional:
            return RiskDecision(
                decision=RiskDecisionType.REJECT,
                rule_name=self.name,
                reason=(
                    f"order notional {notional} below min "
                    f"{self.min_order_notional}"
                ),
                metadata={"notional": str(notional)},
            )
        return RiskDecision(
            decision=RiskDecisionType.PASS,
            rule_name=self.name,
            reason="order size within limits",
            metadata={"notional": str(notional)},
        )
