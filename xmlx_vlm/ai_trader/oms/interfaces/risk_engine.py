"""风控引擎抽象接口."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.oms.constants import RiskDecisionType
from xmlx_vlm.ai_trader.oms.events.types import FillEvent


@dataclass
class RiskContext:
    """风控决策上下文."""

    portfolio: "Portfolio"
    mark_price: Optional[Any] = None
    oracle_price: Optional[Any] = None
    account_equity: Optional[Any] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskDecision:
    """风控决策结果."""

    decision: RiskDecisionType
    rule_name: str
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.decision == RiskDecisionType.PASS

    @property
    def rejected(self) -> bool:
        return self.decision == RiskDecisionType.REJECT

    @property
    def warning(self) -> bool:
        return self.decision == RiskDecisionType.WARNING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "rule_name": self.rule_name,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class RiskRule(ABC):
    """单条风控规则抽象基类."""

    @property
    @abstractmethod
    def name(self) -> str:
        """规则唯一名称."""
        ...

    def pre_trade(self, order: "Order", context: RiskContext) -> RiskDecision:
        """事前风控检查."""
        return RiskDecision(
            decision=RiskDecisionType.PASS,
            rule_name=self.name,
            reason="not implemented",
        )

    def in_flight(self, order: "Order", fill: FillEvent, context: RiskContext) -> Optional[RiskDecision]:
        """事中风控检查（滑点、异常成交等）."""
        return None

    def post_trade(self, trade: "Trade", portfolio: "Portfolio") -> Optional[RiskDecision]:
        """事后风控检查（更新累计值、触发熔断等）."""
        return None

    def reset(self):
        """重置规则累计状态（如跨日）."""
        pass


class RiskEngine(ABC):
    """风控引擎抽象基类."""

    @abstractmethod
    def evaluate_pre_trade(self, order: "Order", context: RiskContext) -> RiskDecision:
        """事前风控统一入口，任一规则拒绝则整体拒绝."""
        ...

    @abstractmethod
    def evaluate_in_flight(
        self, order: "Order", fill: FillEvent, context: RiskContext
    ) -> Optional[RiskDecision]:
        """事中风控统一入口."""
        ...

    @abstractmethod
    def evaluate_post_trade(self, trade: "Trade", portfolio: "Portfolio") -> Optional[RiskDecision]:
        """事后风控统一入口."""
        ...

    @abstractmethod
    def list_rules(self) -> List[RiskRule]:
        """返回所有规则."""
        ...

    @abstractmethod
    def reset(self):
        """重置所有规则状态."""
        ...
