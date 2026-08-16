"""行情领域模型.

所有时间戳统一为毫秒级 UNIX epoch，与 Hyperliquid API 保持一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Tick:
    symbol: str
    price: float
    timestamp_ms: int


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    timestamp_ms: int = 0


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: str  # "buy" | "sell"
    price: float
    size: float
    timestamp_ms: int


@dataclass(frozen=True)
class BookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class BookSnapshot:
    symbol: str
    bids: List[BookLevel]
    asks: List[BookLevel]
    timestamp_ms: int


@dataclass(frozen=True)
class FundingRate:
    symbol: str
    rate: float
    timestamp_ms: int


@dataclass(frozen=True)
class OISnapshot:
    symbol: str
    open_interest: float
    mark_price: float
    funding_rate: float = 0.0
    timestamp_ms: int = 0


@dataclass
class Bar:
    symbol: str
    timeframe: str  # e.g. "1m", "5m", "15m", "1h"
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp_ms: int
    buy_volume: float = 0.0
    sell_volume: float = 0.0

    def to_ohlcv(self) -> "OHLCV":
        return OHLCV(
            timestamp_ms=self.timestamp_ms,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


@dataclass
class OHLCV:
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Indicators:
    """某个时间周期上的技术指标与高阶策略因子快照."""

    timeframe: str
    ema20: float | None = None
    ema50: float | None = None
    rsi14: float | None = None
    atr14: float | None = None
    adx14: float | None = None
    plus_di14: float | None = None
    minus_di14: float | None = None
    poc: float | None = None
    vah: float | None = None
    val: float | None = None
    # 6 大币圈实战高阶量化因子
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_bandwidth: float | None = None
    bb_percent_b: float | None = None
    squeeze_score: float | None = None
    is_squeezed: bool | None = None
    candle_efficiency: float | None = None
    pinbar_type: str | None = None
    pinbar_wick_ratio: float | None = None
    cvd_divergence: str | None = None
    oi_regime: str | None = None
    funding_zscore: float | None = None


@dataclass
class MarketSummary:
    symbol: str
    mark_price: float
    oracle_price: float | None
    basis_pct: float
    bid: float
    ask: float
    spread: float
    high_24h: float | None
    low_24h: float | None
    change_24h_pct: float
    volume_24h: float
    atr14: float | None
    atr_pct: float | None
    adx14: float | None
    rsi14: float | None
    ema20: float | None
    ema50: float | None
    volume_profile: Dict[str, float] = field(default_factory=dict)
    open_interest: float | None = None
    oi_change_1h_pct: float | None = None
    oi_change_24h_pct: float | None = None
    funding_rate: float | None = None
    avg_funding_24h: float | None = None
    funding_trend: str = "stable"
    depth_imbalance: float = 0.0
    bid_qty_top: float = 0.0
    ask_qty_top: float = 0.0
    cvd_15m: float | None = None
    cvd_1h: float | None = None
    cvd_4h: float | None = None
    # 6 大币圈实战高阶因子汇总
    bb_bandwidth: float | None = None
    squeeze_score: float | None = None
    is_squeezed: bool = False
    candle_efficiency: float | None = None
    pinbar_type: str = "none"
    cvd_divergence: str = "neutral"
    oi_regime: str = "neutral"
    funding_zscore: float | None = None

