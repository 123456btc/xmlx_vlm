"""风控决策结果."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from xmlx_vlm.ai_trader.oms.constants import RiskDecisionType


@dataclass
class RiskDecision:
    """单条风控规则或风控引擎的决策结果."""

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
