"""行情警报引擎.

订阅原始行情事件，按阈值判断后产出高阶警报事件，
避免 AI 被每笔 tick 淹没。
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from .events import (
    BookUpdateEvent,
    FundingUpdateEvent,
    IndicatorAlertEvent,
    OIUpdateEvent,
    PriceUpdateEvent,
    TradeEvent,
)
from .models import BookSnapshot, FundingRate, OISnapshot, Trade
from .state import MarketState

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """警报阈值配置."""

    # 价格突破：最近 N 根 K 线高低点
    breakout_lookback_bars: int = 20
    breakout_min_volume_ratio: float = 1.2  # 当前成交量 / 前 N 根平均成交量

    # OI 异动
    oi_1h_change_threshold_pct: float = 5.0
    oi_24h_change_threshold_pct: float = 15.0

    # 大单集群
    large_trade_notional: float = 50_000.0
    large_trade_window_sec: float = 60.0
    large_trade_cluster_count: int = 3
    large_trade_cluster_notional: float = 200_000.0

    # Funding 反转
    funding_flip_threshold: float = 0.0001

    # 盘口失衡
    book_imbalance_threshold: float = 0.6

    # 波动率扩张：当前 5m 波幅相对 ATR(14) 的倍数
    volatility_expansion_ratio: float = 2.0


@dataclass
class _RecentWindow:
    """用于记录最近窗口内的数据."""

    max_age_sec: float
    items: Deque[tuple[float, object]] = field(default_factory=deque)

    def add(self, ts_sec: float, item: object) -> None:
        self.items.append((ts_sec, item))
        cutoff = ts_sec - self.max_age_sec
        while self.items and self.items[0][0] < cutoff:
            self.items.popleft()

    def __len__(self) -> int:
        return len(self.items)


class AlertEngine:
    """基于阈值的事件驱动警报引擎."""

    def __init__(
        self,
        market_state: MarketState,
        event_bus,
        config: Optional[AlertConfig] = None,
    ) -> None:
        self.state = market_state
        self.bus = event_bus
        self.config = config or AlertConfig()

        self._last_funding: Dict[str, FundingRate] = {}
        self._last_oi: Dict[str, OISnapshot] = {}
        self._recent_trades: Dict[str, _RecentWindow] = {}
        self._last_book_imbalance: Dict[str, float] = {}
        self._cooldown_until: Dict[str, float] = {}
        self._cooldown_sec = 60.0  # 同类型同币种警报最小间隔

        self._subscribe()

    def _subscribe(self) -> None:
        self.bus.subscribe(PriceUpdateEvent, self._on_price)
        self.bus.subscribe(TradeEvent, self._on_trade)
        self.bus.subscribe(BookUpdateEvent, self._on_book)
        self.bus.subscribe(FundingUpdateEvent, self._on_funding)
        self.bus.subscribe(OIUpdateEvent, self._on_oi)

    # ── 冷却保护 ──
    def _cool(self, key: str) -> bool:
        now = time.time()
        if now < self._cooldown_until.get(key, 0):
            return False
        self._cooldown_until[key] = now + self._cooldown_sec
        return True

    def _emit(self, alert_type: str, symbol: str, payload: Dict[str, object]) -> None:
        key = f"{alert_type}:{symbol}"
        if not self._cool(key):
            return
        self.bus.publish(
            IndicatorAlertEvent(
                symbol=symbol,
                timestamp_ms=int(time.time() * 1000),
                alert_type=alert_type,
                payload=payload,
            )
        )
        logger.info("Alert emitted: %s %s", alert_type, symbol)

    # ── 事件处理 ──
    def _on_price(self, event: PriceUpdateEvent) -> None:
        self._check_breakout(event.symbol)
        self._check_volatility_expansion(event.symbol)

    def _on_trade(self, event: TradeEvent) -> None:
        trade = event.trade
        if trade.price * trade.size < self.config.large_trade_notional:
            return
        window = self._recent_trades.setdefault(
            trade.symbol, _RecentWindow(self.config.large_trade_window_sec)
        )
        window.add(event.timestamp_ms / 1000.0, trade)
        self._check_large_order_cluster(trade.symbol)

    def _on_book(self, event: BookUpdateEvent) -> None:
        self._check_book_imbalance(event.book)

    def _on_funding(self, event: FundingUpdateEvent) -> None:
        self._check_funding_flip(event.funding)
        self._last_funding[event.funding.symbol] = event.funding

    def _on_oi(self, event: OIUpdateEvent) -> None:
        self._last_oi[event.oi.symbol] = event.oi
        self._check_oi_spike(event.oi.symbol)

    # ── 具体策略 ──
    def _check_breakout(self, symbol: str) -> None:
        ohlcv = self.state.get(symbol, create=False).get_ohlcv("15m", limit=50)
        if len(ohlcv) < self.config.breakout_lookback_bars + 1:
            return
        lookback = ohlcv[-(self.config.breakout_lookback_bars + 1) : -1]
        current = ohlcv[-1]
        high = max(c.high for c in lookback)
        low = min(c.low for c in lookback)
        avg_vol = sum(c.volume for c in lookback) / len(lookback)

        if current.volume < avg_vol * self.config.breakout_min_volume_ratio:
            return

        if current.close > high:
            self._emit(
                "price_breakout",
                symbol,
                {
                    "direction": "up",
                    "break_level": high,
                    "current_price": current.close,
                    "volume_ratio": current.volume / avg_vol if avg_vol else None,
                },
            )
        elif current.close < low:
            self._emit(
                "price_breakout",
                symbol,
                {
                    "direction": "down",
                    "break_level": low,
                    "current_price": current.close,
                    "volume_ratio": current.volume / avg_vol if avg_vol else None,
                },
            )

    def _check_oi_spike(self, symbol: str) -> None:
        sym_state = self.state.get(symbol, create=False)
        delta_1h = sym_state.oi_delta_pct(60)
        delta_24h = sym_state.oi_delta_pct(24 * 60)
        triggered = False
        payload: Dict[str, object] = {}
        if delta_1h is not None and abs(delta_1h) >= self.config.oi_1h_change_threshold_pct:
            triggered = True
            payload["oi_change_1h_pct"] = delta_1h
        if delta_24h is not None and abs(delta_24h) >= self.config.oi_24h_change_threshold_pct:
            triggered = True
            payload["oi_change_24h_pct"] = delta_24h
        if triggered:
            self._emit("oi_spike", symbol, payload)

    def _check_large_order_cluster(self, symbol: str) -> None:
        window = self._recent_trades.get(symbol)
        if window is None or len(window) < self.config.large_trade_cluster_count:
            return
        by_side: Dict[str, List[Trade]] = {"buy": [], "sell": []}
        for _, item in window.items:
            trade = item  # type: ignore[assignment]
            by_side.setdefault(trade.side, []).append(trade)

        for side, trades in by_side.items():
            count = len(trades)
            notional = sum(t.price * t.size for t in trades)
            if (
                count >= self.config.large_trade_cluster_count
                and notional >= self.config.large_trade_cluster_notional
            ):
                self._emit(
                    "large_order_cluster",
                    symbol,
                    {
                        "side": side,
                        "count": count,
                        "notional": notional,
                        "window_sec": self.config.large_trade_window_sec,
                    },
                )
                # 清空该方向避免重复
                for t in trades:
                    try:
                        window.items.remove((t.timestamp_ms / 1000.0, t))
                    except ValueError:
                        pass

    def _check_funding_flip(self, funding: FundingRate) -> None:
        prev = self._last_funding.get(funding.symbol)
        if prev is None:
            return
        if abs(funding.rate) < self.config.funding_flip_threshold:
            return
        if prev.rate * funding.rate < 0:
            self._emit(
                "funding_flip",
                funding.symbol,
                {
                    "previous_rate": prev.rate,
                    "current_rate": funding.rate,
                },
            )

    def _check_book_imbalance(self, book: BookSnapshot) -> None:
        bids = book.bids[:20]
        asks = book.asks[:20]
        bid_qty = sum(l.size for l in bids)
        ask_qty = sum(l.size for l in asks)
        total = bid_qty + ask_qty
        if total == 0:
            return
        imbalance = (bid_qty - ask_qty) / total
        prev = self._last_book_imbalance.get(book.symbol)
        self._last_book_imbalance[book.symbol] = imbalance
        if abs(imbalance) < self.config.book_imbalance_threshold:
            return
        if prev is not None and abs(prev) >= self.config.book_imbalance_threshold:
            # 已经处于极端状态，避免连续警报
            return
        self._emit(
            "book_imbalance_spike",
            book.symbol,
            {
                "imbalance": imbalance,
                "bid_qty_top": bid_qty,
                "ask_qty_top": ask_qty,
            },
        )

    def _check_volatility_expansion(self, symbol: str) -> None:
        ohlcv = self.state.get(symbol, create=False).get_ohlcv("5m", limit=50)
        if len(ohlcv) < 20:
            return
        from .indicators import atr

        atr_values = atr(ohlcv, 14)
        if not atr_values:
            return
        current = ohlcv[-1]
        current_range = current.high - current.low
        latest_atr = atr_values[-1]
        if latest_atr <= 0:
            return
        ratio = current_range / latest_atr
        if ratio >= self.config.volatility_expansion_ratio:
            self._emit(
                "volatility_expansion",
                symbol,
                {
                    "range": current_range,
                    "atr14": latest_atr,
                    "ratio": ratio,
                },
            )
