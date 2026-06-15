"""Agent 目标函数与约束配置."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional

from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO


@dataclass
class RiskBudget:
    """单品种风险预算.

    每笔交易允许承担的最大亏损，通常以账户权益百分比或固定金额表示。
    """

    max_risk_pct_per_trade: Decimal = Decimal("1.0")  # 占权益 1%
    max_risk_usd_per_trade: Optional[Decimal] = None

    def __post_init__(self):
        self.max_risk_pct_per_trade = to_decimal(self.max_risk_pct_per_trade)
        self.max_risk_usd_per_trade = (
            to_decimal(self.max_risk_usd_per_trade)
            if self.max_risk_usd_per_trade is not None
            else None
        )

    def effective_risk_usd(self, equity: Decimal) -> Decimal:
        """返回实际美元风险额度（取较小值）."""
        equity = to_decimal(equity)
        risk_from_pct = equity * self.max_risk_pct_per_trade / Decimal("100")
        if self.max_risk_usd_per_trade is not None:
            return min(risk_from_pct, self.max_risk_usd_per_trade)
        return risk_from_pct


@dataclass
class PositionConstraint:
    """仓位约束."""

    max_position_size_usd: Decimal = Decimal("10000")
    max_leverage: int = 5
    max_positions: int = 5
    max_margin_utilization_pct: Decimal = Decimal("80.0")
    min_confidence: int = 60
    min_risk_reward_ratio: Decimal = Decimal("1.5")

    def __post_init__(self):
        self.max_position_size_usd = to_decimal(self.max_position_size_usd)
        self.max_margin_utilization_pct = to_decimal(self.max_margin_utilization_pct)
        self.min_risk_reward_ratio = to_decimal(self.min_risk_reward_ratio)
        self.min_confidence = max(0, min(100, int(self.min_confidence)))


@dataclass
class AgentObjective:
    """Agent 目标函数.

    明确告诉 AI 追求什么、限制什么，而不是无目标运行。
    """

    daily_volatility_target_pct: Decimal = Decimal("2.0")
    max_drawdown_pct: Decimal = Decimal("5.0")
    sharpe_target: Decimal = Decimal("1.0")
    max_open_positions: int = 5
    preferred_timeframe: str = "1h"
    risk_budget: RiskBudget = field(default_factory=RiskBudget)
    position_constraint: PositionConstraint = field(default_factory=PositionConstraint)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.daily_volatility_target_pct = to_decimal(self.daily_volatility_target_pct)
        self.max_drawdown_pct = to_decimal(self.max_drawdown_pct)
        self.sharpe_target = to_decimal(self.sharpe_target)
        if isinstance(self.risk_budget, dict):
            self.risk_budget = RiskBudget(**self.risk_budget)
        if isinstance(self.position_constraint, dict):
            self.position_constraint = PositionConstraint(**self.position_constraint)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "daily_volatility_target_pct": str(self.daily_volatility_target_pct),
            "max_drawdown_pct": str(self.max_drawdown_pct),
            "sharpe_target": str(self.sharpe_target),
            "max_open_positions": self.max_open_positions,
            "preferred_timeframe": self.preferred_timeframe,
            "risk_budget": {
                "max_risk_pct_per_trade": str(self.risk_budget.max_risk_pct_per_trade),
                "max_risk_usd_per_trade": (
                    str(self.risk_budget.max_risk_usd_per_trade)
                    if self.risk_budget.max_risk_usd_per_trade is not None
                    else None
                ),
            },
            "position_constraint": {
                "max_position_size_usd": str(self.position_constraint.max_position_size_usd),
                "max_leverage": self.position_constraint.max_leverage,
                "max_positions": self.position_constraint.max_positions,
                "max_margin_utilization_pct": str(self.position_constraint.max_margin_utilization_pct),
                "min_confidence": self.position_constraint.min_confidence,
                "min_risk_reward_ratio": str(self.position_constraint.min_risk_reward_ratio),
            },
        }
