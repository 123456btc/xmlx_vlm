"""Unit tests for Event-Driven Backtesting Engine and TearSheet generator."""

import pytest
from xmlx_vlm.ai_trader.market_service.models import OHLCV
from xmlx_vlm.ai_trader.research.backtest import BacktestConfig, BacktestEngine, Position, TearSheet


class TestBacktestEngine:
    """Test suite for BacktestEngine execution, SL/TP triggers, and TearSheet generation."""

    def _generate_synthetic_trend_bars(self, n: int = 100) -> list[OHLCV]:
        bars = []
        base_price = 50000.0
        base_ts = 1700000000000

        for i in range(n):
            # Upward trending price with some oscillation
            p = base_price + i * 100.0 + (50.0 if i % 2 == 0 else -50.0)
            bars.append(
                OHLCV(
                    timestamp_ms=base_ts + i * 3600000,
                    open=p - 20.0,
                    high=p + 80.0,
                    low=p - 60.0,
                    close=p + 10.0,
                    volume=100.0,
                )
            )
        return bars

    def test_backtest_simple_trend_strategy(self):
        bars = self._generate_synthetic_trend_bars(100)
        config = BacktestConfig(initial_equity=10000.0, leverage=2.0)
        engine = BacktestEngine(config)

        # Simple strategy: buy at bar 10, close at bar 80
        def sample_strategy(bar_idx, bar, position, context):
            if bar_idx == 10 and position is None:
                return {
                    "action": "open_long",
                    "size_usd": 5000.0,
                    "stop_loss": 48000.0,
                    "take_profit": 65000.0,
                }
            elif bar_idx == 80 and position is not None:
                return {"action": "close"}
            return None

        result = engine.run(bars, sample_strategy, symbol="BTC")
        sheet = result.tear_sheet

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.side == "long"
        assert trade.net_pnl > 0  # Profitable in an uptrend
        assert sheet.final_equity > sheet.initial_equity
        assert sheet.total_trades == 1
        assert sheet.win_rate_pct == 100.0
        assert sheet.sharpe_ratio > 0.0
        assert len(sheet.equity_curve) == 100

    def test_backtest_stop_loss_trigger(self):
        bars = []
        base_ts = 1700000000000
        # Bar 0: price 50000
        bars.append(OHLCV(base_ts, 50000.0, 50100.0, 49900.0, 50000.0, 10.0))
        # Bar 1: Flash crash down to 47000 (triggering SL at 48500)
        bars.append(OHLCV(base_ts + 3600000, 49900.0, 50000.0, 47000.0, 47200.0, 500.0))

        engine = BacktestEngine(BacktestConfig(initial_equity=10000.0))

        def buy_with_tight_sl(bar_idx, bar, position, context):
            if bar_idx == 0:
                return {"action": "open_long", "size_usd": 2000.0, "stop_loss": 48500.0}
            return None

        result = engine.run(bars, buy_with_tight_sl, symbol="BTC")
        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.exit_reason == "stop_loss"
        assert trade.net_pnl < 0
        assert result.tear_sheet.losing_trades == 1

    def test_backtest_short_position_profit(self):
        bars = []
        base_ts = 1700000000000
        # Downward price series: 50000 -> 40000
        for i in range(20):
            p = 50000.0 - i * 500.0
            bars.append(OHLCV(base_ts + i * 3600000, p, p + 50.0, p - 50.0, p - 20.0, 10.0))

        engine = BacktestEngine(BacktestConfig(initial_equity=10000.0))

        def short_strategy(bar_idx, bar, position, context):
            if bar_idx == 2 and position is None:
                return {"action": "open_short", "size_usd": 4000.0, "take_profit": 42000.0}
            return None

        result = engine.run(bars, short_strategy, symbol="BTC")
        assert len(result.trades) >= 1
        trade = result.trades[0]
        assert trade.side == "short"
        assert trade.net_pnl > 0  # Profited from shorting downward market
        assert result.tear_sheet.total_return_pct > 0
