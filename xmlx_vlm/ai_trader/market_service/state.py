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
from .indicators import (
    adx,
    atr,
    bollinger_bands,
    bollinger_squeeze,
    candle_efficiency,
    cvd_price_divergence,
    ema,
    funding_rate_zscore,
    oi_price_regime,
    pinbar_liquidity_sweep,
    rsi,
    volume_profile,
)
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

        # 6 大币圈实战高阶因子计算
        # 1. 爆仓清算针与插针吸收
        pinbar_info = pinbar_liquidity_sweep(ohlcv)

        # 2. 布林带与挤压突破
        bb_info = bollinger_bands(closes, 20, 2.0)
        # 计算过去多根 K 线的带宽历史以求挤压分位数
        bw_hist = []
        if len(closes) >= 30:
            step = max(1, len(closes) // 50)
            for i in range(20, len(closes) + 1, step):
                bw_hist.append(bollinger_bands(closes[:i], 20, 2.0)["bandwidth"])
        squeeze_info = bollinger_squeeze(bw_hist if bw_hist else [bb_info["bandwidth"]])

        # 3. K 线实体推进效率比
        eff_info = candle_efficiency(ohlcv)

        # 4. CVD 与价格背离 (若有 CVD 历史)
        recent_trades = self.recent_trades(500)
        cvd_vals = []
        if recent_trades:
            # 简易窗口 CVD
            cum = 0.0
            for t in recent_trades:
                val = t.price * t.size
                cum += val if t.side.lower() == "buy" else -val
                cvd_vals.append(cum)
            trade_pxs = [t.price for t in recent_trades]
            cvd_div = cvd_price_divergence(trade_pxs, cvd_vals, lookback=30)
        else:
            cvd_div = {"divergence_type": "neutral", "correlation": 0.0, "is_divergence": False}

        # 5. OI 价格共振 4 象限
        oi_snaps = self.recent_oi(100)
        if len(oi_snaps) >= 4 and len(closes) >= 4:
            oi_vals = [o.open_interest for o in oi_snaps]
            oi_pxs = [o.mark_price if o.mark_price > 0 else closes[-1] for o in oi_snaps]
            oi_reg = oi_price_regime(oi_pxs, oi_vals, lookback=min(30, len(oi_vals)))
        else:
            oi_reg = {"regime": "neutral", "regime_desc": "中性平衡", "price_change_pct": 0.0, "oi_change_pct": 0.0}

        # 6. 资金费率 Z-Score 与拥挤度
        funding_rates = [f.rate for f in self.recent_funding(100)]
        funding_z = funding_rate_zscore(funding_rates)

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
            # 6 大高阶因子输出
            "bb_upper": bb_info.get("upper"),
            "bb_middle": bb_info.get("middle"),
            "bb_lower": bb_info.get("lower"),
            "bb_bandwidth": bb_info.get("bandwidth"),
            "bb_percent_b": bb_info.get("percent_b"),
            "squeeze_score": squeeze_info.get("squeeze_score"),
            "is_squeezed": squeeze_info.get("is_squeezed", False),
            "candle_efficiency": eff_info.get("efficiency"),
            "is_high_efficiency": eff_info.get("is_high_efficiency", False),
            "is_fakeout_risk": eff_info.get("is_fakeout_risk", False),
            "pinbar_type": pinbar_info.get("sweep_type", "none"),
            "pinbar_is_sweep": pinbar_info.get("is_sweep", False),
            "pinbar_wick_ratio": pinbar_info.get("wick_ratio", 0.0),
            "cvd_divergence": cvd_div.get("divergence_type", "neutral"),
            "cvd_correlation": cvd_div.get("correlation", 0.0),
            "oi_regime": oi_reg.get("regime", "neutral"),
            "oi_regime_desc": oi_reg.get("regime_desc", "中性平衡"),
            "funding_zscore": funding_z.get("zscore", 0.0),
            "funding_crowding": funding_z.get("crowding_status", "normal"),
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
                        latest_oi_rows = self.recent_oi(1)
                        latest_oi = latest_oi_rows[-1].open_interest if latest_oi_rows else 0.0
                        latest_funding_rows = self.recent_funding(1)
                        latest_funding = latest_funding_rows[-1].rate if latest_funding_rows else 0.0
                        cvd_val = self.cvd_window(60) or 0.0
                        imbalance_val = 0.0
                        if self.latest_book and (self.latest_book.bids or self.latest_book.asks):
                            b_sz = sum(lvl.size for lvl in self.latest_book.bids[:10])
                            a_sz = sum(lvl.size for lvl in self.latest_book.asks[:10])
                            tot = b_sz + a_sz
                            if tot > 0:
                                imbalance_val = (b_sz - a_sz) / tot
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
