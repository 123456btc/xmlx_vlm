"""仓位簿与账户聚合."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.oms.constants import PositionSide
from xmlx_vlm.ai_trader.oms.core.account import AccountSnapshot
from xmlx_vlm.ai_trader.oms.core.position import Position
from xmlx_vlm.ai_trader.oms.core.trade import Trade
from xmlx_vlm.ai_trader.oms.interfaces.portfolio_tracker import PortfolioTracker
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms


class Portfolio(PortfolioTracker):
    """本地仓位簿与账户聚合."""

    def __init__(self):
        self._positions: Dict[str, Position] = {}
        self._account: AccountSnapshot = AccountSnapshot()
        self._last_sync_ms: int = 0

    # ── PortfolioTracker 接口 ──
    def update_with_trade(self, trade: Trade) -> None:
        pos = self._positions.setdefault(trade.symbol, Position(symbol=trade.symbol, side=PositionSide.FLAT))
        fill_side = "buy" if trade.side.value == "buy" else "sell"
        pos.apply_fill(fill_side, trade.qty, trade.price)
        if pos.is_flat():
            self._positions.pop(trade.symbol, None)
        self._update_account_with_trade(trade)

    def sync_positions(self, positions: Dict[str, Position]) -> None:
        self._positions = {p.symbol: p for p in positions.values()}
        self._last_sync_ms = utc_now_ms()

    def sync_account(self, account: AccountSnapshot) -> None:
        self._account = account
        self._last_sync_ms = utc_now_ms()

    def get_position(self, symbol: str) -> Optional[Position]:
        """获取指定标的的持仓（支持标准对 'BTC/USDC' 或裸币种 'BTC'，大小写不敏感兼容）."""
        from xmlx_vlm.ai_trader.oms.utils.symbol import normalize_symbol, symbol_matches
        if not symbol:
            return None
        try:
            canonical = normalize_symbol(symbol)
            if canonical in self._positions:
                return self._positions[canonical]
        except Exception:
            pass

        # 若未直接命中，遍历匹配标的
        for pos_sym, pos in self._positions.items():
            if symbol_matches(pos_sym, symbol):
                return pos
        return None

    def list_positions(self) -> List[Position]:
        return list(self._positions.values())

    def summary(self) -> Dict[str, Any]:
        gross = self.gross_exposure()
        net = self.net_exposure()
        unrealized = sum((p.unrealized_pnl for p in self._positions.values()), ZERO)
        realized = sum((p.realized_pnl for p in self._positions.values()), ZERO)
        return {
            "account": self._account.to_dict(),
            "positions": [p.to_dict() for p in self._positions.values()],
            "gross_exposure": str(gross),
            "net_exposure": str(net),
            "unrealized_pnl": str(unrealized),
            "realized_pnl": str(realized),
            "margin_utilization_pct": str(self._account.margin_utilization_pct()),
            "last_sync_ms": self._last_sync_ms,
        }

    # ── 业务方法 ──
    def update_mark_prices(self, prices: Dict[str, Decimal]) -> None:
        """用最新价格更新所有持仓未实现盈亏."""
        for symbol, price in prices.items():
            pos = self.get_position(symbol)
            if pos:
                pos.update_mark_price(to_decimal(price))

    def gross_exposure(self) -> Decimal:
        """总名义敞口（多空绝对值之和）."""
        return sum((p.notional() for p in self._positions.values()), ZERO)

    def net_exposure(self) -> Decimal:
        """净名义敞口（多 - 空）."""
        total = ZERO
        for p in self._positions.values():
            if p.is_long():
                total += p.notional()
            elif p.is_short():
                total -= p.notional()
        return total

    def position_notional(self, symbol: str) -> Decimal:
        pos = self.get_position(symbol)
        return pos.notional() if pos else ZERO

    def total_realized_pnl(self) -> Decimal:
        return sum((p.realized_pnl for p in self._positions.values()), ZERO)

    def total_unrealized_pnl(self) -> Decimal:
        return sum((p.unrealized_pnl for p in self._positions.values()), ZERO)

    @property
    def account(self) -> AccountSnapshot:
        return self._account

    def _update_account_with_trade(self, trade: Trade) -> None:
        """根据成交简单更新账户现金（仅适用于现货/保证金简化模型）."""
        notional = trade.notional()
        if trade.side.value == "buy":
            self._account.cash -= notional
        else:
            self._account.cash += notional
        self._account.timestamp_ms = utc_now_ms()
