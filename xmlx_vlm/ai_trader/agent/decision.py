"""Agent 决策数据模型."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO


class ActionType(str, Enum):
    """Agent 决策动作类型."""

    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    HOLD = "hold"
    WAIT = "wait"


@dataclass
class SignalEvaluation:
    """信号评估结果.

    包含置信度、风险收益比、建议止损/止盈等。
    """

    signal_type: str
    symbol: str
    confidence: int = 0  # 0-100
    risk_reward_ratio: Decimal = ZERO
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    expected_return_pct: Decimal = ZERO
    expected_risk_pct: Decimal = ZERO
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        self.confidence = max(0, min(100, int(self.confidence)))
        self.risk_reward_ratio = to_decimal(self.risk_reward_ratio)
        self.expected_return_pct = to_decimal(self.expected_return_pct)
        self.expected_risk_pct = to_decimal(self.expected_risk_pct)
        if self.stop_loss is not None:
            self.stop_loss = to_decimal(self.stop_loss)
        if self.take_profit is not None:
            self.take_profit = to_decimal(self.take_profit)

    def passes(self, min_confidence: int, min_rr: Decimal) -> bool:
        """检查是否通过最低门槛."""
        if self.confidence < min_confidence:
            return False
        if self.risk_reward_ratio < to_decimal(min_rr):
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "symbol": self.symbol,
            "confidence": self.confidence,
            "risk_reward_ratio": str(self.risk_reward_ratio),
            "stop_loss": str(self.stop_loss) if self.stop_loss is not None else None,
            "take_profit": str(self.take_profit) if self.take_profit is not None else None,
            "expected_return_pct": str(self.expected_return_pct),
            "expected_risk_pct": str(self.expected_risk_pct),
            "notes": self.notes,
            "metadata": self.metadata,
            "evaluated_at": self.evaluated_at.isoformat(),
        }


@dataclass
class TradeProposal:
    """交易提案.

    由 Agent 生成，等待根据运行模式决定如何处理。
    """

    action: ActionType
    symbol: str
    size_usd: Decimal = ZERO
    leverage: int = 1
    confidence: int = 0
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    expected_return_pct: Decimal = ZERO
    expected_risk_pct: Decimal = ZERO
    risk_reward_ratio: Decimal = ZERO
    reason: str = ""
    variant_id: str = "default"
    verification_info: Optional[Dict[str, Any]] = None
    entry_price: Optional[Decimal] = None

    def __post_init__(self):
        if isinstance(self.action, str):
            self.action = ActionType(self.action.lower())
        self.symbol = self.symbol.upper().strip()
        self.size_usd = to_decimal(self.size_usd)
        self.confidence = max(0, min(100, int(self.confidence)))
        self.expected_return_pct = to_decimal(self.expected_return_pct)
        self.expected_risk_pct = to_decimal(self.expected_risk_pct)
        self.risk_reward_ratio = to_decimal(self.risk_reward_ratio)
        if self.entry_price is not None:
            self.entry_price = to_decimal(self.entry_price)
        if self.stop_loss is not None:
            self.stop_loss = to_decimal(self.stop_loss)
        if self.take_profit is not None:
            self.take_profit = to_decimal(self.take_profit)

    @property
    def side(self) -> Optional[str]:
        """返回 OMS 侧 buy/sell."""
        if self.action in (ActionType.OPEN_LONG, ActionType.CLOSE_SHORT):
            return "buy"
        if self.action in (ActionType.OPEN_SHORT, ActionType.CLOSE_LONG):
            return "sell"
        return None

    @property
    def is_open(self) -> bool:
        return self.action in (ActionType.OPEN_LONG, ActionType.OPEN_SHORT)

    @property
    def is_close(self) -> bool:
        return self.action in (ActionType.CLOSE_LONG, ActionType.CLOSE_SHORT)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "symbol": self.symbol,
            "side": self.side,
            "size_usd": str(self.size_usd),
            "entry_price": str(self.entry_price) if self.entry_price is not None else None,
            "leverage": self.leverage,
            "confidence": self.confidence,
            "stop_loss": str(self.stop_loss) if self.stop_loss is not None else None,
            "take_profit": str(self.take_profit) if self.take_profit is not None else None,
            "expected_return_pct": str(self.expected_return_pct),
            "expected_risk_pct": str(self.expected_risk_pct),
            "risk_reward_ratio": str(self.risk_reward_ratio),
            "reason": self.reason,
            "variant_id": self.variant_id,
            "verification_info": self.verification_info,
        }


@dataclass
class AgentDecision:
    """一次完整的 Agent 决策记录."""

    decision_id: str
    symbol: str
    proposal: TradeProposal
    evaluation: SignalEvaluation
    mode: str
    executed: bool = False
    execution_result: Optional[Dict[str, Any]] = None
    rationale: Optional[Dict[str, Any]] = None
    rejected_reason: Optional[str] = None
    variant_id: str = "default"
    shadow: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "proposal": self.proposal.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "mode": self.mode,
            "executed": self.executed,
            "execution_result": self.execution_result,
            "rationale": self.rationale,
            "rejected_reason": self.rejected_reason,
            "variant_id": self.variant_id,
            "shadow": self.shadow,
            "created_at": self.created_at.isoformat(),
        }
