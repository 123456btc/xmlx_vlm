"""风控引擎实现."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.oms.constants import RiskDecisionType
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.core.portfolio import Portfolio
from xmlx_vlm.ai_trader.oms.core.trade import Trade
from xmlx_vlm.ai_trader.oms.events.types import FillEvent
from xmlx_vlm.ai_trader.oms.interfaces.risk_engine import (
    RiskContext,
    RiskDecision,
    RiskEngine,
    RiskRule,
)
from xmlx_vlm.ai_trader.oms.risk.rules.base import RiskRule as BaseRiskRule
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal


class RiskManager(RiskEngine):
    """默认风控引擎：串行执行所有规则，任一拒绝即整体拒绝."""

    def __init__(self, rules: Optional[List[RiskRule]] = None):
        self._rules: List[RiskRule] = list(rules or [])

    @property
    def rules(self) -> List[RiskRule]:
        return self._rules

    def add_rule(self, rule: RiskRule) -> None:
        self._rules.append(rule)

    def remove_rule(self, rule_name: str) -> bool:
        for idx, r in enumerate(self._rules):
            if r.name == rule_name:
                self._rules.pop(idx)
                return True
        return False

    def evaluate_pre_trade(self, order: Order, context: RiskContext) -> RiskDecision:
        warnings: List[str] = []
        for rule in self._rules:
            decision = rule.pre_trade(order, context)
            if decision.rejected:
                return decision
            if decision.warning:
                warnings.append(f"{rule.name}: {decision.reason}")

        reason = "all rules passed"
        if warnings:
            reason += "; warnings: " + "; ".join(warnings)
        return RiskDecision(
            decision=RiskDecisionType.PASS,
            rule_name="risk_manager",
            reason=reason,
        )

    def evaluate_in_flight(
        self, order: Order, fill: FillEvent, context: RiskContext
    ) -> Optional[RiskDecision]:
        for rule in self._rules:
            decision = rule.in_flight(order, fill, context)
            if decision is not None and decision.rejected:
                return decision
        return None

    def evaluate_post_trade(self, trade: Trade, portfolio: Portfolio) -> Optional[RiskDecision]:
        for rule in self._rules:
            decision = rule.post_trade(trade, portfolio)
            if decision is not None and decision.rejected:
                return decision
        return None

    def list_rules(self) -> List[RiskRule]:
        return self._rules

    def reset(self):
        for rule in self._rules:
            rule.reset()

    def status(self) -> List[Dict[str, Any]]:
        return [{"name": r.name} for r in self._rules]

    @classmethod
    def from_profile(cls, profile: Dict[str, Decimal]) -> "RiskManager":
        """从风控配置模板创建默认规则集合."""
        from xmlx_vlm.ai_trader.oms.risk.rules import (
            DailyLossRule,
            PositionLimitRule,
            OrderSizeRule,
            PriceDeviationRule,
            RateLimitRule,
            MarginRule,
        )

        return cls(
            rules=[
                DailyLossRule(
                    max_daily_loss_pct=to_decimal(profile.get("max_daily_loss_pct", "3.0"))
                ),
                PositionLimitRule(
                    max_single_position_pct=to_decimal(
                        profile.get("max_single_position_pct", "20.0")
                    ),
                    max_total_position_pct=to_decimal(
                        profile.get("max_total_position_pct", "50.0")
                    ),
                ),
                OrderSizeRule(
                    max_single_order_notional=to_decimal(
                        profile.get("max_single_order_notional", "5000")
                    ),
                    min_order_notional=to_decimal(profile.get("min_order_notional", "10")),
                ),
                PriceDeviationRule(
                    max_price_deviation_pct=to_decimal(
                        profile.get("max_price_deviation_pct", "1.0")
                    )
                ),
                RateLimitRule(
                    max_orders_per_minute=int(profile.get("max_orders_per_minute", 12)),
                    max_orders_per_second=int(profile.get("max_orders_per_second", 3)),
                ),
                MarginRule(
                    min_available_margin_pct=to_decimal(
                        profile.get("min_available_margin_pct", "20.0")
                    )
                ),
            ]
        )
