"""AI 决策所需上下文."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.market_service.models import MarketSummary
from xmlx_vlm.ai_trader.oms.core.account import AccountSnapshot
from xmlx_vlm.ai_trader.oms.core.position import Position
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO


@dataclass
class TradingStats:
    """历史交易统计."""

    total_trades: int = 0
    win_rate: Decimal = ZERO
    profit_factor: Decimal = ZERO
    sharpe_ratio: Decimal = ZERO
    total_pnl: Decimal = ZERO
    avg_win: Decimal = ZERO
    avg_loss: Decimal = ZERO
    max_drawdown_pct: Decimal = ZERO

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "win_rate": str(self.win_rate),
            "profit_factor": str(self.profit_factor),
            "sharpe_ratio": str(self.sharpe_ratio),
            "total_pnl": str(self.total_pnl),
            "avg_win": str(self.avg_win),
            "avg_loss": str(self.avg_loss),
            "max_drawdown_pct": str(self.max_drawdown_pct),
        }


@dataclass
class RecentOrder:
    """最近已完成订单."""

    symbol: str
    side: str
    entry_price: Decimal
    exit_price: Decimal
    realized_pnl: Decimal
    pnl_pct: Decimal
    entry_time: str
    exit_time: str
    hold_duration: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": str(self.entry_price),
            "exit_price": str(self.exit_price),
            "realized_pnl": str(self.realized_pnl),
            "pnl_pct": str(self.pnl_pct),
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "hold_duration": self.hold_duration,
        }


@dataclass
class TradingContext:
    """单次决策所需的完整上下文."""

    current_time: str
    runtime_minutes: int
    cycle_number: int
    account: AccountSnapshot
    positions: List[Position] = field(default_factory=list)
    candidate_symbols: List[str] = field(default_factory=list)
    market_data: Dict[str, MarketSummary] = field(default_factory=dict)
    trading_stats: Optional[TradingStats] = None
    recent_orders: List[RecentOrder] = field(default_factory=list)
    prompt_variant: str = "default"
    trader_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_time": self.current_time,
            "runtime_minutes": self.runtime_minutes,
            "cycle_number": self.cycle_number,
            "trader_id": self.trader_id,
            "account": self.account.to_dict(),
            "positions": [p.to_dict() for p in self.positions],
            "candidate_symbols": self.candidate_symbols,
            "market_data": {k: self._market_summary_to_dict(v) for k, v in self.market_data.items()},
            "trading_stats": self.trading_stats.to_dict() if self.trading_stats else None,
            "recent_orders": [o.to_dict() for o in self.recent_orders],
            "prompt_variant": self.prompt_variant,
        }

    @staticmethod
    def _market_summary_to_dict(summary: MarketSummary) -> Dict[str, Any]:
        return {
            "symbol": summary.symbol,
            "mark_price": summary.mark_price,
            "oracle_price": summary.oracle_price,
            "basis_pct": summary.basis_pct,
            "bid": summary.bid,
            "ask": summary.ask,
            "spread": summary.spread,
            "change_24h_pct": summary.change_24h_pct,
            "volume_24h": summary.volume_24h,
            "atr14": summary.atr14,
            "rsi14": summary.rsi14,
            "ema20": summary.ema20,
            "ema50": summary.ema50,
            "open_interest": summary.open_interest,
            "oi_change_1h_pct": summary.oi_change_1h_pct,
            "oi_change_24h_pct": summary.oi_change_24h_pct,
            "funding_rate": summary.funding_rate,
            "depth_imbalance": summary.depth_imbalance,
            "cvd_15m": summary.cvd_15m,
            "cvd_1h": summary.cvd_1h,
            "cvd_4h": summary.cvd_4h,
        }
