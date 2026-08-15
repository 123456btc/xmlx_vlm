"""市场状态识别分类器 (Market Regime Detector).

通过 ADX、DMI 方向指标、实现波动率 (ATR 比率) 与筹码价值区分布 (Volume Profile)，
量化识别当前标的/大盘的市场状态（单边趋势、高波动恐慌、低波震荡、流动性不足），
为 AI Agent 动态选择策略（趋势跟踪 vs 区间网格 vs 空仓防守）提供确定性依据。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.market_service import indicators
from xmlx_vlm.ai_trader.market_service.models import OHLCV

logger = logging.getLogger(__name__)


class MarketRegime(str, Enum):
    """市场状态分类."""

    TREND_BULLISH = "trend_bullish"          # 强多头单边趋势 (顺势做多/金字塔加仓)
    TREND_BEARISH = "trend_bearish"          # 强空头单边趋势 (顺势做空)
    RANGE_BOUND = "range_bound"              # 低波区间震荡 (网格/布林带均值回归)
    HIGH_VOL_PANIC = "high_vol_panic"        # 极端高波动/流动性挤兑 (防守观望/严控敞口)
    LOW_LIQUIDITY = "low_liquidity"          # 低流动性死盘 (禁止开仓)
    UNKNOWN = "unknown"


@dataclass
class RegimeAnalysis:
    """市场状态分析输出."""

    symbol: str
    regime: MarketRegime
    confidence: float                        # 0.0 ~ 1.0
    suggested_strategy: str                  # "trend_following" | "grid_mean_reversion" | "defensive_cash"
    metrics: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""


class MarketRegimeDetector:
    """市场状态量化检测器."""

    def __init__(
        self,
        trend_adx_threshold: float = 25.0,
        range_adx_threshold: float = 20.0,
        panic_atr_multiplier: float = 2.2,
    ):
        self.trend_adx_threshold = trend_adx_threshold
        self.range_adx_threshold = range_adx_threshold
        self.panic_atr_multiplier = panic_atr_multiplier

    def detect_regime(
        self,
        symbol: str,
        ohlcv_candles: List[OHLCV],
        timeframe: str = "1h",
    ) -> RegimeAnalysis:
        """根据历史 K 线序列判断市场状态."""
        if not ohlcv_candles or len(ohlcv_candles) < 28:
            return RegimeAnalysis(
                symbol=symbol.upper(),
                regime=MarketRegime.UNKNOWN,
                confidence=0.5,
                suggested_strategy="defensive_cash",
                summary="K线数据样本不足 (要求 ≥ 28 根)，保持保守观望",
            )

        closes = [c.close for c in ohlcv_candles]
        curr_price = closes[-1]

        # 1. 计算 ADX 与 DMI
        adx_series, plus_di, minus_di = indicators.adx(ohlcv_candles, period=14)
        curr_adx = adx_series[-1] if adx_series else 0.0
        curr_plus = plus_di[-1] if plus_di else 0.0
        curr_minus = minus_di[-1] if minus_di else 0.0

        # 2. 计算 ATR 及波动率扩张比 (Volatility Expansion Ratio)
        atr_series = indicators.atr(ohlcv_candles, period=14)
        curr_atr = atr_series[-1] if atr_series else 0.0
        # 基准 ATR: 过去 30 根均值
        baseline_atr = sum(atr_series[-30:]) / len(atr_series[-30:]) if len(atr_series) >= 30 else curr_atr
        vol_ratio = (curr_atr / baseline_atr) if baseline_atr > 0 else 1.0

        # 3. 计算均线排列 (EMA9, EMA21)
        ema_fast = indicators.ema(closes, 9)
        ema_slow = indicators.ema(closes, 21)
        fast_val = ema_fast[-1] if ema_fast else curr_price
        slow_val = ema_slow[-1] if ema_slow else curr_price

        # 4. 计算 Volume Profile 筹码分布
        vp = indicators.volume_profile(ohlcv_candles[-48:])  # 最近 48 根 K 线的筹码分布
        poc = vp.get("poc")
        vah = vp.get("vah")
        val = vp.get("val")

        metrics = {
            "adx": round(curr_adx, 2),
            "plus_di": round(curr_plus, 2),
            "minus_di": round(curr_minus, 2),
            "current_atr": round(curr_atr, 4),
            "volatility_ratio": round(vol_ratio, 2),
            "ema9": round(fast_val, 2),
            "ema21": round(slow_val, 2),
            "poc": poc,
            "vah": vah,
            "val": val,
        }

        # 状态判定决策树 (Priority Decision Tree)

        # 规则 A: 极端恐慌/剧烈插针 (Panic / Volatility Shock)
        if vol_ratio >= self.panic_atr_multiplier:
            return RegimeAnalysis(
                symbol=symbol.upper(),
                regime=MarketRegime.HIGH_VOL_PANIC,
                confidence=0.88,
                suggested_strategy="defensive_cash",
                metrics=metrics,
                summary=f"波动率剧烈扩张 (当前 ATR 为历史基准的 {vol_ratio:.2f} 倍)，触发极端防守，暂停高频开仓",
            )

        # 规则 B: 强单边趋势 (Trend Expansion)
        if curr_adx >= self.trend_adx_threshold:
            if curr_plus > curr_minus and fast_val > slow_val and curr_price > slow_val:
                return RegimeAnalysis(
                    symbol=symbol.upper(),
                    regime=MarketRegime.TREND_BULLISH,
                    confidence=min(0.95, 0.65 + (curr_adx - self.trend_adx_threshold) * 0.01),
                    suggested_strategy="trend_following",
                    metrics=metrics,
                    summary=f"处于明确多头单边趋势 (ADX={curr_adx:.1f}, +DI>{curr_minus:.1f}, 均线多头排列)，建议顺势做多/移动止损",
                )
            elif curr_minus > curr_plus and fast_val < slow_val and curr_price < slow_val:
                return RegimeAnalysis(
                    symbol=symbol.upper(),
                    regime=MarketRegime.TREND_BEARISH,
                    confidence=min(0.95, 0.65 + (curr_adx - self.trend_adx_threshold) * 0.01),
                    suggested_strategy="trend_following",
                    metrics=metrics,
                    summary=f"处于明确空头单边趋势 (ADX={curr_adx:.1f}, -DI>{curr_plus:.1f}, 均线空头排列)，建议顺势做空",
                )

        # 规则 C: 低波区间震荡 (Range-Bound Consolidation)
        if curr_adx <= self.range_adx_threshold:
            return RegimeAnalysis(
                symbol=symbol.upper(),
                regime=MarketRegime.RANGE_BOUND,
                confidence=0.82,
                suggested_strategy="grid_mean_reversion",
                metrics=metrics,
                summary=f"市场动能减弱处于区间震荡 (ADX={curr_adx:.1f} < {self.range_adx_threshold})，适合网格逢低买入逢高卖出",
            )

        # 默认过渡态
        return RegimeAnalysis(
            symbol=symbol.upper(),
            regime=MarketRegime.RANGE_BOUND,
            confidence=0.60,
            suggested_strategy="grid_mean_reversion",
            metrics=metrics,
            summary=f"市场处于震荡向趋势过渡阶段 (ADX={curr_adx:.1f})，建议控制仓位并收窄止盈",
        )
