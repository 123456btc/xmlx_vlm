"""持久化抽象接口."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO

if TYPE_CHECKING:
    from xmlx_vlm.ai_trader.decision.decision import FullDecision


@dataclass
class EquitySnapshot:
    """账户权益快照."""

    trader_id: str
    timestamp_ms: int
    total_equity: Decimal = ZERO
    available_margin: Decimal = ZERO
    unrealized_pnl: Decimal = ZERO
    realized_pnl: Decimal = ZERO
    margin_used_pct: Decimal = ZERO
    position_count: int = 0

    def __post_init__(self):
        self.total_equity = to_decimal(self.total_equity)
        self.available_margin = to_decimal(self.available_margin)
        self.unrealized_pnl = to_decimal(self.unrealized_pnl)
        self.realized_pnl = to_decimal(self.realized_pnl)
        self.margin_used_pct = to_decimal(self.margin_used_pct)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trader_id": self.trader_id,
            "timestamp_ms": self.timestamp_ms,
            "total_equity": str(self.total_equity),
            "available_margin": str(self.available_margin),
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "margin_used_pct": str(self.margin_used_pct),
            "position_count": self.position_count,
        }


class DecisionStore(ABC):
    """决策与权益快照持久化抽象."""

    @abstractmethod
    def save_decision(self, record: FullDecision) -> None:
        """保存一次完整决策记录."""
        ...

    @abstractmethod
    def save_equity_snapshot(self, snapshot: EquitySnapshot) -> None:
        """保存账户权益快照."""
        ...

    @abstractmethod
    def list_decisions(
        self,
        trader_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[FullDecision]:
        """查询某策略的决策记录."""
        ...

    @abstractmethod
    def list_equity_snapshots(
        self,
        trader_id: str,
        limit: int = 1000,
        offset: int = 0,
    ) -> List[EquitySnapshot]:
        """查询某策略的权益快照."""
        ...

    @abstractmethod
    def get_latest_equity_snapshot(self, trader_id: str) -> Optional[EquitySnapshot]:
        """获取最新权益快照."""
        ...

    @abstractmethod
    def close(self) -> None:
        """释放资源."""
        ...
