"""网格策略状态机."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO


@dataclass
class GridLevelInfo:
    """单个网格档位信息."""

    index: int
    price: Decimal
    buy_order_id: Optional[str] = None
    sell_order_id: Optional[str] = None
    filled: bool = False
    realized_pnl: Decimal = ZERO

    def __post_init__(self):
        self.price = to_decimal(self.price)
        self.realized_pnl = to_decimal(self.realized_pnl)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "price": str(self.price),
            "buy_order_id": self.buy_order_id,
            "sell_order_id": self.sell_order_id,
            "filled": self.filled,
            "realized_pnl": str(self.realized_pnl),
        }


@dataclass
class GridState:
    """网格策略运行时状态."""

    symbol: str
    upper_price: Decimal
    lower_price: Decimal
    grid_count: int
    total_investment: Decimal
    max_drawdown_pct: Decimal
    daily_loss_limit_pct: Decimal

    levels: List[GridLevelInfo] = field(default_factory=list)
    is_paused: bool = False
    is_initialized: bool = False

    total_profit: Decimal = ZERO
    total_trades: int = 0
    winning_trades: int = 0
    max_drawdown: Decimal = ZERO
    peak_equity: Decimal = ZERO
    daily_pnl: Decimal = ZERO
    last_daily_reset: str = ""

    current_direction: str = "neutral"
    position_reduction_pct: Decimal = ZERO
    breakout_level: Optional[str] = None
    breakout_direction: Optional[str] = None

    def __post_init__(self):
        self.upper_price = to_decimal(self.upper_price)
        self.lower_price = to_decimal(self.lower_price)
        self.total_investment = to_decimal(self.total_investment)
        self.max_drawdown_pct = to_decimal(self.max_drawdown_pct)
        self.daily_loss_limit_pct = to_decimal(self.daily_loss_limit_pct)
        if not self.levels:
            self._build_levels()

    def _build_levels(self) -> None:
        """等差构建网格档位."""
        if self.grid_count <= 0 or self.upper_price <= self.lower_price:
            return
        step = (self.upper_price - self.lower_price) / self.grid_count
        self.levels = [
            GridLevelInfo(
                index=i,
                price=self.lower_price + step * i,
            )
            for i in range(self.grid_count + 1)
        ]

    @property
    def grid_spacing(self) -> Decimal:
        if len(self.levels) < 2:
            return ZERO
        return self.levels[1].price - self.levels[0].price

    def level_for_price(self, price: Decimal) -> Optional[GridLevelInfo]:
        """找到价格最接近的网格档位."""
        price = to_decimal(price)
        if not self.levels:
            return None
        return min(self.levels, key=lambda lvl: abs(lvl.price - price))

    def check_breakout(self, price: Decimal) -> Optional[str]:
        """检测价格是否突破网格区间."""
        price = to_decimal(price)
        if price > self.upper_price:
            return "upper"
        if price < self.lower_price:
            return "lower"
        return None

    def update_drawdown(self, current_equity: Decimal) -> Decimal:
        """更新最大回撤，返回当前回撤百分比."""
        current_equity = to_decimal(current_equity)
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        if self.peak_equity <= ZERO:
            return ZERO
        drawdown = (self.peak_equity - current_equity) / self.peak_equity * Decimal("100")
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
        return drawdown

    def check_max_drawdown(self, current_equity: Decimal) -> bool:
        drawdown = self.update_drawdown(current_equity)
        return self.max_drawdown_pct > ZERO and drawdown >= self.max_drawdown_pct

    def update_daily_pnl(self, realized_pnl: Decimal) -> None:
        realized_pnl = to_decimal(realized_pnl)
        self.daily_pnl += realized_pnl
        self.total_profit += realized_pnl
        if realized_pnl > ZERO:
            self.winning_trades += 1
        self.total_trades += 1

    def check_daily_loss_limit(self) -> bool:
        if self.daily_loss_limit_pct <= ZERO or self.total_investment <= ZERO:
            return False
        daily_loss_pct = (-self.daily_pnl) / self.total_investment * Decimal("100")
        return self.daily_pnl < ZERO and daily_loss_pct >= self.daily_loss_limit_pct

    def reset_daily(self, date_str: str) -> None:
        if self.last_daily_reset != date_str:
            self.daily_pnl = ZERO
            self.last_daily_reset = date_str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "upper_price": str(self.upper_price),
            "lower_price": str(self.lower_price),
            "grid_count": self.grid_count,
            "grid_spacing": str(self.grid_spacing),
            "total_investment": str(self.total_investment),
            "max_drawdown_pct": str(self.max_drawdown_pct),
            "daily_loss_limit_pct": str(self.daily_loss_limit_pct),
            "levels": [lvl.to_dict() for lvl in self.levels],
            "is_paused": self.is_paused,
            "is_initialized": self.is_initialized,
            "total_profit": str(self.total_profit),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "max_drawdown": str(self.max_drawdown),
            "peak_equity": str(self.peak_equity),
            "daily_pnl": str(self.daily_pnl),
            "current_direction": self.current_direction,
        }
