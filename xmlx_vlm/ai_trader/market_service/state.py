"""内存状态机.

维护每个交易对的实时状态：最新 tick、订单簿、成交、资金费率、持仓量、
K 线序列与技术指标。所有读写通过锁保证线程安全。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

from .columnar_store import ColumnarMarketStore
from .indicators import adx, atr, ema, rsi, volume_profile
from .kline_db import KlineDB
from .models import (
    Bar,
    BookLevel,
    BookSnapshot,
    FundingRate,
    OHLCV,
    OISnapshot,
    Quote,
    Tick,
    Trade,
)

logger = logging.getLogger(__name__)

# 保留最近 N 条原始数据
TRADE_RING_SIZE = 10_000
FUNDING_RING_SIZE = 288  # 8h * 30 天 ≈ 90，留足余量
OI_RING_SIZE = 1_500     # 1 分钟一条可存 25h

_TIME_FRAME_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
}


def _floor_ms(ts_ms: int, interval_ms: int) -> int:
    return (ts_ms // interval_ms) * interval_ms


class _RingBuffer:
    def __init__(self, maxlen: int) -> None:
        self._deque: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, item: object) -> None:
        with self._lock:
            self._deque.append(item)

    def slice(self, count: int) -> List[object]:
        with self._lock:
            return list(self._deque)[-count:]

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)


class SymbolState:
    """单个交易对的状态."""

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._lock = threading.RLock()

        self.latest_tick: Optional[Tick] = None
        self.latest_quote: Optional[Quote] = None
        self.latest_book: Optional[BookSnapshot] = None
        self.trades = _RingBuffer(TRADE_RING_SIZE)
        self.funding = _RingBuffer(FUNDING_RING_SIZE)
        self.oi = _RingBuffer(OI_RING_SIZE)

        # 按 timeframe 聚合的 K 线
        self._bars: Dict[str, List[Bar]] = defaultdict(list)
        # 正在构建中的当前 K 线
        self._current_bar: Dict[str, Bar | None] = {}

        self._kline_db = KlineDB()
        self._has_candle_stream = False
        self.load_from_db()

        # Performance caches
        self._cvd_cache: Dict[int, float] = {}
        self._cvd_cache_time: Dict[int, float] = {}
        self._oi_delta_cache: Dict[int, float | None] = {}
        self._oi_delta_cache_time: Dict[int, float] = {}
        self._indicator_cache: Dict[str, Dict[str, Any]] = {}
        self._indicator_cache_time: Dict[str, float] = {}

    # ── 更新接口 ──
    def update_tick(self, tick: Tick) -> None:
        with self._lock:
            self.latest_tick = tick
            self.latest_quote = Quote(
                symbol=tick.symbol,
                bid=tick.price,
                ask=tick.price,
                timestamp_ms=tick.timestamp_ms,
            )
            self._aggregate_bar(tick.price, 0.0, tick.timestamp_ms, "", 0.0)

    def update_book(self, book: BookSnapshot) -> None:
        with self._lock:
            self.latest_book = book
            if book.bids and book.asks:
                self.latest_quote = Quote(
                    symbol=book.symbol,
                    bid=book.bids[0].price,
                    ask=book.asks[0].price,
                    bid_size=book.bids[0].size,
                    ask_size=book.asks[0].size,
                    timestamp_ms=book.timestamp_ms,
                )

    def add_trade(self, trade: Trade) -> None:
        with self._lock:
            self.trades.append(trade)
            self._aggregate_bar(
                trade.price,
                trade.price * trade.size,
                trade.timestamp_ms,
                trade.side,
                trade.size,
            )

    def add_funding(self, funding: FundingRate) -> None:
        with self._lock:
            self.funding.append(funding)

    def add_oi(self, oi: OISnapshot) -> None:
        with self._lock:
            self.oi.append(oi)

    # ── 查询接口 ──
    def get_quote(self) -> Optional[Quote]:
        with self._lock:
            return self.latest_quote

    def get_book(self) -> Optional[BookSnapshot]:
        with self._lock:
            return self.latest_book

    def get_ohlcv(self, timeframe: str, limit: int = 100) -> List[OHLCV]:
        with self._lock:
            bars = list(self._bars.get(timeframe, []))[-limit:]
            current = self._current_bar.get(timeframe)
            if current is not None:
                bars = bars + [current]
            return [b.to_ohlcv() for b in bars]

    def recent_trades(self, count: int = 1000) -> List[Trade]:
        trades = self.trades.slice(count)
        return [t for t in trades if isinstance(t, Trade)]

    def recent_funding(self, count: int = 288) -> List[FundingRate]:
        items = self.funding.slice(count)
        return [f for f in items if isinstance(f, FundingRate)]

    def recent_oi(self, count: int = OI_RING_SIZE) -> List[OISnapshot]:
        items = self.oi.slice(count)
        return [x for x in items if isinstance(x, OISnapshot)]

    def oi_delta_pct(self, minutes: int) -> Optional[float]:
        now = time.time()
        with self._lock:
            if minutes in self._oi_delta_cache and now - self._oi_delta_cache_time.get(minutes, 0.0) < 2.0:
                return self._oi_delta_cache[minutes]
        
        val = self._calculate_oi_delta_pct(minutes)
        with self._lock:
            self._oi_delta_cache[minutes] = val
            self._oi_delta_cache_time[minutes] = now
        return val

    def _calculate_oi_delta_pct(self, minutes: int) -> Optional[float]:
        rows = self.recent_oi(OI_RING_SIZE)
        if len(rows) < 2:
            return None
        now_ms = int(time.time() * 1000)
        threshold = now_ms - minutes * 60_000
        past = [r for r in rows if r.timestamp_ms <= threshold]
        if not past:
            return None
        current = rows[-1]
        old = past[-1]
        if old.open_interest == 0:
            return None
        return (current.open_interest - old.open_interest) / old.open_interest * 100.0

    def cvd_window(self, minutes: int) -> Optional[float]:
        now = time.time()
        with self._lock:
            if minutes in self._cvd_cache and now - self._cvd_cache_time.get(minutes, 0.0) < 2.0:
                return self._cvd_cache[minutes]
                
        val = self._calculate_cvd_window(minutes)
        with self._lock:
            self._cvd_cache[minutes] = val
            self._cvd_cache_time[minutes] = now
        return val

    def _calculate_cvd_window(self, minutes: int) -> Optional[float]:
        trades = self.recent_trades(TRADE_RING_SIZE)
        if not trades:
            return None
        now_ms = int(time.time() * 1000)
        threshold = now_ms - minutes * 60_000
        window = [t for t in trades if t.timestamp_ms >= threshold]
        if not window:
            return None
        total = 0.0
        for t in window:
            val = t.price * t.size
            total += val if t.side.lower() == "buy" else -val
        return total

    def indicators(self, timeframe: str) -> Dict[str, object]:
        now = time.time()
        with self._lock:
            if timeframe in self._indicator_cache and now - self._indicator_cache_time.get(timeframe, 0.0) < 5.0:
                return self._indicator_cache[timeframe]
                
        val = self._calculate_indicators(timeframe)
        with self._lock:
            self._indicator_cache[timeframe] = val
            self._indicator_cache_time[timeframe] = now
        return val

    def _calculate_indicators(self, timeframe: str) -> Dict[str, object]:
        ohlcv = self.get_ohlcv(timeframe, limit=200)
        if len(ohlcv) < 26:
            return {}
        closes = [c.close for c in ohlcv]
        highs = [c.high for c in ohlcv]
        lows = [c.low for c in ohlcv]

        ema20 = ema(closes, 20)
        ema50 = ema(closes, 50)
        rsi14 = rsi(closes, 14)
        atr14 = atr(ohlcv, 14)
        adx14, plus_di, minus_di = adx(ohlcv, 14)
        vp = volume_profile(ohlcv, bins=24)

        latest = closes[-1]
        return {
            "ema20": ema20[-1] if ema20 else None,
            "ema50": ema50[-1] if ema50 else None,
            "rsi14": rsi14[-1] if rsi14 else None,
            "atr14": atr14[-1] if atr14 else None,
            "atr_pct": (atr14[-1] / latest * 100.0) if atr14 and latest else None,
            "adx14": adx14[-1] if adx14 else None,
            "plus_di14": plus_di[-1] if plus_di else None,
            "minus_di14": minus_di[-1] if minus_di else None,
            "poc": vp.get("poc"),
            "vah": vp.get("vah"),
            "val": vp.get("val"),
        }

    def update_candle(self, bar: Bar) -> None:
        with self._lock:
            self._has_candle_stream = True
            tf = bar.timeframe
            # We only do incremental aggregation from 1m bars
            if tf == "1m":
                current = self._current_bar.get("1m")
                if current is None:
                    self._current_bar["1m"] = bar
                elif current.timestamp_ms != bar.timestamp_ms:
                    # The previous 1m bar has closed!
                    # Append it to _bars
                    self._bars["1m"].append(current)
                    if len(self._bars["1m"]) > 200:
                        self._bars["1m"].pop(0)
                    # Save it to SQLite
                    self._kline_db.save_bar(current)
                    # Trigger incremental aggregation
                    self._aggregate_higher_timeframe(current)
                    # Set the new active bar
                    self._current_bar["1m"] = bar
                else:
                    # Just update the current active bar
                    self._current_bar["1m"] = bar
            else:
                # If we receive higher timeframe bars directly (e.g. from bootstrapping), just save them
                self._current_bar[tf] = bar

    def _aggregate_higher_timeframe(self, m1_bar: Bar) -> None:
        # Aggregate 1m bar into 5m, 15m, 1h, 4h
        for tf in ("5m", "15m", "1h", "4h"):
            interval_ms = _TIME_FRAME_MS[tf]
            aligned_ts = _floor_ms(m1_bar.timestamp_ms, interval_ms)
            current = self._current_bar.get(tf)
            
            if current is None or current.timestamp_ms != aligned_ts:
                # The previous timeframe bar has closed!
                if current is not None:
                    self._bars[tf].append(current)
                    if len(self._bars[tf]) > 200:
                        self._bars[tf].pop(0)
                    self._kline_db.save_bar(current)
                    
                    # Also persist to Columnar Store
                    try:
                        latest_oi = self.latest_oi.open_interest if self.latest_oi else 0.0
                        latest_funding = self.latest_funding.funding_rate if self.latest_funding else 0.0
                        cvd_val = self.get_cvd("1h")
                        imbalance_val = self.book_imbalance(10)
                        ColumnarMarketStore.get_instance().append_bar(
                            symbol=self.symbol,
                            timeframe=tf,
                            ts=current.timestamp_ms,
                            o=current.open,
                            h=current.high,
                            l=current.low,
                            c=current.close,
                            v=current.volume,
                            cvd=cvd_val,
                            oi=latest_oi,
                            funding=latest_funding,
                            imbalance=imbalance_val,
                        )
                    except Exception as exc:
                        logger.debug("Failed to record columnar bar: %s", exc)
                # Start new partial bar
                current = Bar(
                    symbol=self.symbol,
                    timeframe=tf,
                    open=m1_bar.open,
                    high=m1_bar.high,
                    low=m1_bar.low,
                    close=m1_bar.close,
                    volume=m1_bar.volume,
                    timestamp_ms=aligned_ts,
                    buy_volume=m1_bar.buy_volume,
                    sell_volume=m1_bar.sell_volume
                )
            else:
                # Merge into existing partial bar
                current.high = max(current.high, m1_bar.high)
                current.low = min(current.low, m1_bar.low)
                current.close = m1_bar.close
                current.volume += m1_bar.volume
                current.buy_volume += m1_bar.buy_volume
                current.sell_volume += m1_bar.sell_volume
                
            self._current_bar[tf] = current

    def load_from_db(self) -> None:
        with self._lock:
            for tf in ("1m", "5m", "15m", "1h", "4h"):
                bars = self._kline_db.load_bars(self.symbol, tf, limit=150)
                if bars:
                    interval_ms = _TIME_FRAME_MS.get(tf)
                    if interval_ms:
                        now_ms = int(time.time() * 1000)
                        current_bar_ts = _floor_ms(now_ms, interval_ms)
                        latest_bar = bars[-1]
                        if latest_bar.timestamp_ms == current_bar_ts:
                            self._current_bar[tf] = latest_bar
                            self._bars[tf] = bars[:-1]
                        else:
                            self._current_bar[tf] = None
                            self._bars[tf] = bars
                    else:
                        self._bars[tf] = bars

    # ── 内部：K 线聚合 ──
    def _aggregate_bar(
        self,
        price: float,
        notional: float,
        timestamp_ms: int,
        side: str,
        size: float,
    ) -> None:
        if getattr(self, "_has_candle_stream", False):
            return
        for tf, interval_ms in _TIME_FRAME_MS.items():
            bar_ts = _floor_ms(timestamp_ms, interval_ms)
            current = self._current_bar.get(tf)
            if current is None or current.timestamp_ms != bar_ts:
                if current is not None:
                    self._bars[tf].append(current)
                current = Bar(
                    symbol=self.symbol,
                    timeframe=tf,
                    open=price,
                    high=price,
                    low=price,
                    close=price,
                    volume=notional,
                    timestamp_ms=bar_ts,
                    buy_volume=notional if side.lower() == "buy" else 0.0,
                    sell_volume=notional if side.lower() == "sell" else 0.0,
                )
                self._current_bar[tf] = current
            else:
                current.high = max(current.high, price)
                current.low = min(current.low, price)
                current.close = price
                current.volume += notional
                if side.lower() == "buy":
                    current.buy_volume += notional
                elif side.lower() == "sell":
                    current.sell_volume += notional


class MarketState:
    """所有交易对的全局状态."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._symbols: Dict[str, SymbolState] = {}

    def get(self, symbol: str, create: bool = True) -> SymbolState:
        with self._lock:
            if symbol not in self._symbols and create:
                self._symbols[symbol] = SymbolState(symbol)
            return self._symbols[symbol]

    def symbols(self) -> List[str]:
        with self._lock:
            return list(self._symbols.keys())

    def snapshot(self) -> Dict[str, dict]:
        """导出只读快照，用于调试或日志."""
        with self._lock:
            return {
                sym: {
                    "tick": state.latest_tick,
                    "quote": state.latest_quote,
                    "trade_count": len(state.trades),
                    "funding_count": len(state.funding),
                    "oi_count": len(state.oi),
                }
                for sym, state in self._symbols.items()
            }
