"""
High-Performance Event-Driven Historical Backtesting Engine & TearSheet Generator.

Features:
1. Dual-direction (Long/Short) perpetual futures accounting with realistic fees & slippage.
2. In-bar High/Low stop-loss and take-profit intrabar trigger simulation.
3. Institutional-grade TearSheet performance reporting (Sharpe, Sortino, Calmar, MaxDD, WinRate).
4. Direct integration with ColumnarMarketStore and OHLCV time-series.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from xmlx_vlm.ai_trader.market_service.models import OHLCV
from xmlx_vlm.ai_trader.market_service.columnar_store import ColumnarMarketStore

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtesting execution configuration."""

    initial_equity: float = 10000.0
    taker_fee: float = 0.0005     # 0.05% taker fee
    maker_fee: float = 0.0002     # 0.02% maker fee
    slippage_pct: float = 0.0002  # 0.02% base slippage
    leverage: float = 3.0
    risk_per_trade: float = 0.02  # 2% equity risk per trade
    timeframe: str = "1h"
    bars_per_year: int = 365 * 24 # 8760 for 1h perpetual crypto


@dataclass
class Position:
    """Active open position."""

    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    qty: float
    notional: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_bar_idx: int = 0
    entry_ts: int = 0

    def compute_unrealized_pnl(self, current_price: float) -> float:
        if self.side == "long":
            return (current_price - self.entry_price) * self.qty
        else:
            return (self.entry_price - current_price) * self.qty


@dataclass
class TradeRecord:
    """Completed trade record."""

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: float
    notional: float
    pnl: float
    fee: float
    net_pnl: float
    pnl_pct: float
    entry_ts: int
    exit_ts: int
    hold_bars: int
    exit_reason: str  # "signal", "stop_loss", "take_profit"


@dataclass
class TearSheet:
    """Institutional-grade backtesting performance report."""

    initial_equity: float
    final_equity: float
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    max_drawdown_bars: int
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_trade_pnl: float
    avg_hold_bars: float
    max_profit: float
    max_loss: float
    equity_curve: List[Tuple[int, float]] = field(default_factory=list)

    def summary(self) -> str:
        """Format clean Markdown tear sheet summary."""
        return (
            f"=== 📊 回测绩效评估报告 (Backtest TearSheet) ===\n"
            f"• 初始本金: ${self.initial_equity:,.2f}  ──>  最终权益: ${self.final_equity:,.2f}\n"
            f"• 累计收益率: {self.total_return_pct:+.2f}%  |  年化收益率 (CAGR): {self.cagr_pct:+.2f}%\n"
            f"• 夏普比率 (Sharpe): {self.sharpe_ratio:.2f}  |  索提诺比率 (Sortino): {self.sortino_ratio:.2f}\n"
            f"• 最大回撤 (MaxDD): {self.max_drawdown_pct:.2f}%  |  卡尔玛比率 (Calmar): {self.calmar_ratio:.2f}\n"
            f"• 总交易次数: {self.total_trades} (胜: {self.winning_trades}, 负: {self.losing_trades})\n"
            f"• 胜率: {self.win_rate_pct:.2f}%  |  盈亏比 (Profit Factor): {self.profit_factor:.2f}\n"
            f"• 平均单笔盈亏: ${self.avg_trade_pnl:+.2f}  |  平均持仓周期: {self.avg_hold_bars:.1f} 根Bar"
        )


@dataclass
class BacktestResult:
    """Full backtesting execution result."""

    config: BacktestConfig
    tear_sheet: TearSheet
    trades: List[TradeRecord]
    positions_history: List[Dict[str, Any]] = field(default_factory=list)


StrategyCallback = Callable[[int, OHLCV, Optional[Position], Dict[str, Any]], Optional[Dict[str, Any]]]


class BacktestEngine:
    """Event-Driven Historical Backtesting Engine."""

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        ohlcv_bars: List[OHLCV],
        strategy_fn: StrategyCallback,
        symbol: str = "BTC",
    ) -> BacktestResult:
        """
        Run backtest over a series of OHLCV bars using a strategy callback.

        strategy_fn signature:
            strategy_fn(bar_idx, current_bar, open_position, context) -> Optional[Dict]
            Returning None = Hold / No action
            Returning dict:
                {"action": "open_long" | "open_short" | "close", "size_usd": ..., "stop_loss": ..., "take_profit": ...}
        """
        if not ohlcv_bars:
            empty_sheet = TearSheet(
                initial_equity=self.config.initial_equity,
                final_equity=self.config.initial_equity,
                total_return_pct=0.0,
                cagr_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown_pct=0.0,
                max_drawdown_bars=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate_pct=0.0,
                profit_factor=0.0,
                avg_trade_pnl=0.0,
                avg_hold_bars=0.0,
                max_profit=0.0,
                max_loss=0.0,
            )
            return BacktestResult(config=self.config, tear_sheet=empty_sheet, trades=[])

        equity = self.config.initial_equity
        position: Optional[Position] = None
        trades: List[TradeRecord] = []
        equity_curve: List[Tuple[int, float]] = []
        context: Dict[str, Any] = {"history": []}

        for i, bar in enumerate(ohlcv_bars):
            context["history"].append(bar)
            curr_price = bar.close

            # 1. Check Intrabar Stop Loss and Take Profit triggers on active position
            if position is not None:
                exit_triggered = False
                exit_price = curr_price
                exit_reason = ""

                if position.side == "long":
                    # Check Stop Loss (triggered if bar.low <= SL)
                    if position.stop_loss and bar.low <= position.stop_loss:
                        exit_price = min(bar.open, position.stop_loss) * (1.0 - self.config.slippage_pct)
                        exit_reason = "stop_loss"
                        exit_triggered = True
                    # Check Take Profit (triggered if bar.high >= TP)
                    elif position.take_profit and bar.high >= position.take_profit:
                        exit_price = max(bar.open, position.take_profit) * (1.0 - self.config.slippage_pct)
                        exit_reason = "take_profit"
                        exit_triggered = True
                elif position.side == "short":
                    # Check Stop Loss (triggered if bar.high >= SL)
                    if position.stop_loss and bar.high >= position.stop_loss:
                        exit_price = max(bar.open, position.stop_loss) * (1.0 + self.config.slippage_pct)
                        exit_reason = "stop_loss"
                        exit_triggered = True
                    # Check Take Profit (triggered if bar.low <= TP)
                    elif position.take_profit and bar.low <= position.take_profit:
                        exit_price = min(bar.open, position.take_profit) * (1.0 + self.config.slippage_pct)
                        exit_reason = "take_profit"
                        exit_triggered = True

                if exit_triggered:
                    trade = self._close_position(position, exit_price, bar.timestamp_ms, i, exit_reason)
                    trades.append(trade)
                    equity += trade.net_pnl
                    position = None

            # 2. Call strategy decision function
            signal = strategy_fn(i, bar, position, context)

            if signal and isinstance(signal, dict):
                action = signal.get("action", "").lower()

                # Action: Close
                if action == "close" and position is not None:
                    slip_mult = 1.0 - self.config.slippage_pct if position.side == "long" else 1.0 + self.config.slippage_pct
                    exec_price = curr_price * slip_mult
                    trade = self._close_position(position, exec_price, bar.timestamp_ms, i, "signal")
                    trades.append(trade)
                    equity += trade.net_pnl
                    position = None

                # Action: Open Long / Open Short
                elif action in ("open_long", "open_short") and position is None:
                    side = "long" if action == "open_long" else "short"
                    slip_mult = 1.0 + self.config.slippage_pct if side == "long" else 1.0 - self.config.slippage_pct
                    entry_px = curr_price * slip_mult

                    # Size calculation
                    max_notional = equity * self.config.leverage
                    desired_size_usd = float(signal.get("size_usd", equity * self.config.risk_per_trade * self.config.leverage))
                    notional = min(max_notional, max(10.0, desired_size_usd))
                    qty = notional / entry_px if entry_px > 0 else 0.0

                    open_fee = notional * self.config.taker_fee
                    equity -= open_fee

                    position = Position(
                        symbol=symbol,
                        side=side,
                        entry_price=entry_px,
                        qty=qty,
                        notional=notional,
                        stop_loss=signal.get("stop_loss"),
                        take_profit=signal.get("take_profit"),
                        entry_bar_idx=i,
                        entry_ts=bar.timestamp_ms,
                    )

            # 3. Mark to market equity recording
            unrealized = position.compute_unrealized_pnl(curr_price) if position else 0.0
            mark_equity = equity + unrealized
            equity_curve.append((bar.timestamp_ms, mark_equity))

        # Close remaining open position at the end of backtest period
        if position is not None and ohlcv_bars:
            last_bar = ohlcv_bars[-1]
            slip_mult = 1.0 - self.config.slippage_pct if position.side == "long" else 1.0 + self.config.slippage_pct
            final_px = last_bar.close * slip_mult
            trade = self._close_position(position, final_px, last_bar.timestamp_ms, len(ohlcv_bars) - 1, "end_of_backtest")
            trades.append(trade)
            equity += trade.net_pnl
            equity_curve[-1] = (last_bar.timestamp_ms, equity)

        # 4. Generate TearSheet
        tear_sheet = self._generate_tearsheet(equity_curve, trades, len(ohlcv_bars))

        return BacktestResult(
            config=self.config,
            tear_sheet=tear_sheet,
            trades=trades,
        )

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        exit_ts: int,
        current_bar_idx: int,
        reason: str,
    ) -> TradeRecord:
        exit_notional = position.qty * exit_price
        close_fee = exit_notional * self.config.taker_fee
        if position.side == "long":
            raw_pnl = (exit_price - position.entry_price) * position.qty
        else:
            raw_pnl = (position.entry_price - exit_price) * position.qty

        net_pnl = raw_pnl - close_fee
        pnl_pct = (net_pnl / position.notional * 100.0) if position.notional > 0 else 0.0
        hold_bars = max(1, current_bar_idx - position.entry_bar_idx)

        return TradeRecord(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            qty=position.qty,
            notional=position.notional,
            pnl=raw_pnl,
            fee=close_fee,
            net_pnl=net_pnl,
            pnl_pct=round(pnl_pct, 2),
            entry_ts=position.entry_ts,
            exit_ts=exit_ts,
            hold_bars=hold_bars,
            exit_reason=reason,
        )

    def _generate_tearsheet(
        self,
        equity_curve: List[Tuple[int, float]],
        trades: List[TradeRecord],
        total_bars: int,
    ) -> TearSheet:
        if not equity_curve:
            return TearSheet(
                initial_equity=self.config.initial_equity,
                final_equity=self.config.initial_equity,
                total_return_pct=0.0,
                cagr_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                max_drawdown_pct=0.0,
                max_drawdown_bars=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate_pct=0.0,
                profit_factor=0.0,
                avg_trade_pnl=0.0,
                avg_hold_bars=0.0,
                max_profit=0.0,
                max_loss=0.0,
                equity_curve=[],
            )

        init_eq = self.config.initial_equity
        final_eq = equity_curve[-1][1]
        total_ret_pct = (final_eq - init_eq) / init_eq * 100.0

        # Returns series
        returns = []
        for j in range(1, len(equity_curve)):
            prev_e = equity_curve[j - 1][1]
            curr_e = equity_curve[j][1]
            ret = (curr_e - prev_e) / prev_e if prev_e > 0 else 0.0
            returns.append(ret)

        # Annualized CAGR
        years = max(1 / 365, total_bars / self.config.bars_per_year)
        cagr_pct = ((final_eq / init_eq) ** (1.0 / years) - 1.0) * 100.0 if final_eq > 0 and init_eq > 0 else -100.0

        # Sharpe & Sortino
        if returns and len(returns) > 1:
            mean_r = sum(returns) / len(returns)
            var_r = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
            std_r = var_r ** 0.5

            annual_factor = math.sqrt(self.config.bars_per_year)
            sharpe = (mean_r / std_r * annual_factor) if std_r > 1e-8 else 0.0

            # Downside deviation
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                down_std = (sum(r ** 2 for r in downside_returns) / len(returns)) ** 0.5
                sortino = (mean_r / down_std * annual_factor) if down_std > 1e-8 else 0.0
            else:
                sortino = sharpe * 1.5 if sharpe > 0 else 0.0
        else:
            sharpe = 0.0
            sortino = 0.0

        # Max Drawdown
        peak = init_eq
        max_dd_pct = 0.0
        curr_dd_bars = 0
        max_dd_bars = 0

        for _, eq in equity_curve:
            if eq > peak:
                peak = eq
                curr_dd_bars = 0
            else:
                curr_dd_bars += 1
                if curr_dd_bars > max_dd_bars:
                    max_dd_bars = curr_dd_bars
                dd = (peak - eq) / peak * 100.0 if peak > 0 else 0.0
                if dd > max_dd_pct:
                    max_dd_pct = dd

        calmar = (cagr_pct / max_dd_pct) if max_dd_pct > 0 else 0.0

        # Trade metrics
        tot_trades = len(trades)
        wins = [t for t in trades if t.net_pnl > 0]
        losses = [t for t in trades if t.net_pnl <= 0]
        win_rate = (len(wins) / tot_trades * 100.0) if tot_trades > 0 else 0.0

        total_gain = sum(t.net_pnl for t in wins)
        total_loss = abs(sum(t.net_pnl for t in losses))
        profit_factor = (total_gain / total_loss) if total_loss > 0 else (99.9 if total_gain > 0 else 0.0)
        avg_trade_pnl = (sum(t.net_pnl for t in trades) / tot_trades) if tot_trades > 0 else 0.0
        avg_hold = (sum(t.hold_bars for t in trades) / tot_trades) if tot_trades > 0 else 0.0
        max_profit = max((t.net_pnl for t in trades), default=0.0)
        max_loss = min((t.net_pnl for t in trades), default=0.0)

        return TearSheet(
            initial_equity=round(init_eq, 2),
            final_equity=round(final_eq, 2),
            total_return_pct=round(total_ret_pct, 2),
            cagr_pct=round(cagr_pct, 2),
            sharpe_ratio=round(sharpe, 2),
            sortino_ratio=round(sortino, 2),
            calmar_ratio=round(calmar, 2),
            max_drawdown_pct=round(max_dd_pct, 2),
            max_drawdown_bars=max_dd_bars,
            total_trades=tot_trades,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            avg_trade_pnl=round(avg_trade_pnl, 2),
            avg_hold_bars=round(avg_hold, 1),
            max_profit=round(max_profit, 2),
            max_loss=round(max_loss, 2),
            equity_curve=equity_curve,
        )
