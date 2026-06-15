"""风控规则基类."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.core.portfolio import Portfolio
from xmlx_vlm.ai_trader.oms.core.trade import Trade
from xmlx_vlm.ai_trader.oms.events.types import FillEvent
from xmlx_vlm.ai_trader.oms.interfaces.risk_engine import RiskContext, RiskDecision


class RiskRule(ABC):
    """单条风控规则抽象基类."""

    @property
    @abstractmethod
    def name(self) -> str:
        """规则唯一名称."""
        ...

    def pre_trade(self, order: Order, context: RiskContext) -> RiskDecision:
        """事前风控检查."""
        from xmlx_vlm.ai_trader.oms.constants import RiskDecisionType

        return RiskDecision(
            decision=RiskDecisionType.PASS,
            rule_name=self.name,
            reason="not implemented",
        )

    def in_flight(
        self, order: Order, fill: FillEvent, context: RiskContext
    ) -> Optional[RiskDecision]:
        """事中风控检查（滑点、异常成交等）."""
        return None

    def post_trade(self, trade: Trade, portfolio: Portfolio) -> Optional[RiskDecision]:
        """事后风控检查（更新累计值、触发熔断等）."""
        return None

    def reset(self):
        """重置规则累计状态."""
        pass
