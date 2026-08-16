"""AI 交易决策数据模型."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO


@dataclass
class Decision:
    """单个 AI 交易决策.

    action 取值：
    - open_long / open_short：开新仓
    - close_long / close_short：平仓
    - hold：维持当前持仓
    - wait：不操作
    """

    action: str
    symbol: str
    position_size_usd: Optional[Decimal] = None
    leverage: Optional[int] = None
    price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    confidence: int = 0
    reasoning: str = ""

    def __post_init__(self):
        from xmlx_vlm.ai_trader.oms.utils.symbol import normalize_symbol
        self.action = self.action.lower().strip()
        if self.symbol:
            self.symbol = normalize_symbol(self.symbol)
        if self.position_size_usd is not None:
            self.position_size_usd = to_decimal(self.position_size_usd)
        if self.price is not None:
            self.price = to_decimal(self.price)
        if self.stop_loss is not None:
            self.stop_loss = to_decimal(self.stop_loss)
        if self.take_profit is not None:
            self.take_profit = to_decimal(self.take_profit)
        self.confidence = max(0, min(100, int(self.confidence or 0)))

    @property
    def is_open(self) -> bool:
        return self.action in {"open_long", "open_short"}

    @property
    def is_close(self) -> bool:
        return self.action in {"close_long", "close_short"}

    @property
    def side(self) -> Optional[str]:
        """返回 buy/sell 方向（开仓时）."""
        if self.action == "open_long":
            return "buy"
        if self.action == "open_short":
            return "sell"
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "symbol": self.symbol,
            "position_size_usd": str(self.position_size_usd) if self.position_size_usd is not None else None,
            "leverage": self.leverage,
            "price": str(self.price) if self.price is not None else None,
            "stop_loss": str(self.stop_loss) if self.stop_loss is not None else None,
            "take_profit": str(self.take_profit) if self.take_profit is not None else None,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Decision":
        return cls(
            action=data.get("action", "wait"),
            symbol=data.get("symbol", ""),
            position_size_usd=data.get("position_size_usd"),
            leverage=data.get("leverage"),
            price=data.get("price"),
            stop_loss=data.get("stop_loss"),
            take_profit=data.get("take_profit"),
            confidence=data.get("confidence", 0),
            reasoning=data.get("reasoning", ""),
        )


@dataclass
class FullDecision:
    """一次完整决策记录，包含输入、输出、延迟."""

    decisions: List[Decision] = field(default_factory=list)
    system_prompt: str = ""
    user_prompt: str = ""
    cot_trace: str = ""
    raw_response: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: int = 0
    cycle_number: int = 0
    trader_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trader_id": self.trader_id,
            "cycle_number": self.cycle_number,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "cot_trace": self.cot_trace,
            "raw_response": self.raw_response,
            "decisions": [d.to_dict() for d in self.decisions],
        }
