"""路由上下文."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, Optional

from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO


@dataclass
class RoutingContext:
    """影响订单路由决策的上下文."""

    mark_price: Optional[Decimal] = None
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    spread_pct: Optional[Decimal] = None
    book_depth: Optional[Decimal] = None      # 目标价方向盘口深度
    recent_volume: Optional[Decimal] = None    # 最近 interval 成交量
    volatility: Optional[Decimal] = None       # 日波动率
    urgency: str = "normal"                    # passive / normal / aggressive
    max_slippage_pct: Decimal = Decimal("0.5")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.mark_price is not None:
            self.mark_price = to_decimal(self.mark_price)
        if self.bid is not None:
            self.bid = to_decimal(self.bid)
        if self.ask is not None:
            self.ask = to_decimal(self.ask)
        if self.spread_pct is not None:
            self.spread_pct = to_decimal(self.spread_pct)
        if self.book_depth is not None:
            self.book_depth = to_decimal(self.book_depth)
        if self.recent_volume is not None:
            self.recent_volume = to_decimal(self.recent_volume)
        if self.volatility is not None:
            self.volatility = to_decimal(self.volatility)
        self.max_slippage_pct = to_decimal(self.max_slippage_pct)
