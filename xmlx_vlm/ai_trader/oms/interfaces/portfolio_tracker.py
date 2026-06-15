"""仓位跟踪器抽象接口."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class PortfolioTracker(ABC):
    """仓位与账户跟踪器抽象基类."""

    @abstractmethod
    def update_with_trade(self, trade: "Trade") -> None:
        """根据成交更新仓位."""
        ...

    @abstractmethod
    def sync_positions(self, positions: Dict[str, "Position"]) -> None:
        """用交易所持仓覆盖本地."""
        ...

    @abstractmethod
    def sync_account(self, account: "AccountSnapshot") -> None:
        """同步账户快照."""
        ...

    @abstractmethod
    def get_position(self, symbol: str) -> Optional["Position"]:
        """获取单个品种持仓."""
        ...

    @abstractmethod
    def list_positions(self) -> List["Position"]:
        """获取全部持仓."""
        ...

    @abstractmethod
    def summary(self) -> Dict[str, Any]:
        """返回汇总信息."""
        ...
