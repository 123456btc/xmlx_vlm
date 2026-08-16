"""
K-Line & Quantitative Strategy Dataset Builder for MLX Fine-Tuning.

Features:
1. Multi-timeframe K-line sequence slicing and technical/orderflow indicator extraction.
2. Forward-Looking Labeling (computes actual forward N-bar return and max drawdown).
3. Chain-of-Thought (COT) SFT dataset generation for quantitative market reasoning.
4. ORPO / DPO preference dataset generation (Chosen vs. Rejected risk-managed trade setups).
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from xmlx_vlm.ai_trader.market_service.kline_db import KlineDB
from xmlx_vlm.ai_trader.market_service.models import Bar

logger = logging.getLogger(__name__)


@dataclass
class DatasetBuildConfig:
    """Hyperparameters for K-Line Dataset Generation."""
    window_size: int = 20
    forward_window: int = 5
    min_profit_pct: float = 0.015
    stop_loss_multiplier: float = 1.8
    risk_reward_target: float = 2.0


class KlineDatasetBuilder:
    """
    Automated dataset generator converting K-line history into MLX SFT & ORPO datasets.
    """

    def __init__(self, config: Optional[DatasetBuildConfig] = None, db_path: Optional[Path] = None):
        self.config = config or DatasetBuildConfig()
        self.kline_db = KlineDB(db_path=db_path)

    def fetch_bars_from_db(self, symbol: str = "BTC", timeframe: str = "1h", limit: int = 500) -> List[Dict[str, Any]]:
        """Fetch historical bars from KlineDB."""
        bars: List[Bar] = self.kline_db.load_bars(symbol, timeframe, limit=limit)
        return [
            {
                "symbol": b.symbol,
                "timeframe": b.timeframe,
                "timestamp_ms": b.timestamp_ms,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "buy_volume": b.buy_volume,
                "sell_volume": b.sell_volume,
            }
            for b in bars
        ]

    def build_sft_samples(
        self,
        bars: List[Dict[str, Any]],
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
    ) -> List[Dict[str, Any]]:
        """
        Generate Supervised Fine-Tuning (SFT) samples with rich Chain-of-Thought (COT) reasoning.
        """
        samples: List[Dict[str, Any]] = []
        n = len(bars)
        w = self.config.window_size
        fw = self.config.forward_window

        if n < w + fw:
            return samples

        for i in range(w, n - fw):
            window_bars = bars[i - w : i]
            future_bars = bars[i : i + fw]

            curr_bar = window_bars[-1]
            entry_price = curr_bar["close"]

            # Calculate indicators on window
            closes = [b["close"] for b in window_bars]
            highs = [b["high"] for b in window_bars]
            lows = [b["low"] for b in window_bars]
            volumes = [b["volume"] for b in window_bars]

            ema9 = float(np.mean(closes[-9:])) if len(closes) >= 9 else entry_price
            ema21 = float(np.mean(closes[-21:])) if len(closes) >= 21 else entry_price
            atr14 = float(np.mean([h - l for h, l in zip(highs[-14:], lows[-14:])])) if len(highs) >= 14 else entry_price * 0.01

            # Future performance outcome
            future_high = max(b["high"] for b in future_bars)
            future_low = min(b["low"] for b in future_bars)
            future_exit = future_bars[-1]["close"]

            max_up_pct = (future_high - entry_price) / entry_price
            max_down_pct = (entry_price - future_low) / entry_price

            # Determine forward label and COT
            if max_up_pct >= self.config.min_profit_pct and max_down_pct < max_up_pct * 0.6:
                decision = "BUY"
                regime = "Bullish Momentum Continuation"
                stop_loss = round(entry_price - atr14 * 1.5, 2)
                take_profit = round(entry_price + (entry_price - stop_loss) * self.config.risk_reward_target, 2)
                confidence = 85
                reason = f"Price successfully held above EMA21 ({ema21:.2f}) with upward breakout momentum. Target risk-reward {self.config.risk_reward_target}:1."
            elif max_down_pct >= self.config.min_profit_pct and max_up_pct < max_down_pct * 0.6:
                decision = "SELL"
                regime = "Bearish Breakdown"
                stop_loss = round(entry_price + atr14 * 1.5, 2)
                take_profit = round(entry_price - (stop_loss - entry_price) * self.config.risk_reward_target, 2)
                confidence = 85
                reason = f"Price broke below key dynamic support EMA21 ({ema21:.2f}) with increasing selling volume. Downside targets projected."
            else:
                decision = "WAIT"
                regime = "Choppy Range Bound"
                stop_loss = round(entry_price * 0.98, 2)
                take_profit = round(entry_price * 1.02, 2)
                confidence = 60
                reason = "Market lacks clear directional trend and is oscillating in compression range. Prudent risk management recommends waiting."

            user_prompt = f"""[Market Context Analysis Request]
Symbol: {symbol}
Timeframe: {timeframe}
Current Price: {entry_price:.2f}
Recent 5 Bars (Close, Volume): {[(round(b['close'], 2), round(b['volume'], 1)) for b in window_bars[-5:]]}
Technical Indicators: EMA9={ema9:.2f}, EMA21={ema21:.2f}, ATR14={atr14:.2f}

Provide detailed quantitative market regime analysis, risk assessment, and precise trade decision in JSON format."""

            assistant_response = f"""### 📊 Quantitative Market Analysis

**1. Market Regime Assessment**:
- Current Regime: **{regime}**
- Trend Alignment: Price is {'above' if entry_price >= ema21 else 'below'} EMA21 ({ema21:.2f}) with ATR volatility of {atr14:.2f}.

**2. Risk & Position Sizing (Kelly / ATR)**:
- Stop Loss Buffer: 1.5x ATR ({atr14 * 1.5:.2f} USD).
- Calculated Stop Loss: {stop_loss}
- Projected Take Profit: {take_profit}
- Expected Risk/Reward: {self.config.risk_reward_target}:1

**3. Execution Decision**:
```json
{{
  "action": "{decision.lower()}",
  "symbol": "{symbol}",
  "entry_price": {entry_price:.2f},
  "stop_loss": {stop_loss},
  "take_profit": {take_profit},
  "confidence": {confidence},
  "reasoning": "{reason}"
}}
```"""

            samples.append({
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response},
                ]
            })

        return samples

    def build_orpo_samples(
        self,
        bars: List[Dict[str, Any]],
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
    ) -> List[Dict[str, Any]]:
        """
        Generate ORPO / DPO preference pairs (Chosen vs. Rejected responses).
        """
        sft_samples = self.build_sft_samples(bars, symbol=symbol, timeframe=timeframe)
        orpo_samples: List[Dict[str, Any]] = []

        for sample in sft_samples:
            prompt = sample["messages"][0]["content"]
            chosen = sample["messages"][1]["content"]

            # Construct synthetic rejected response (high-risk, inverted stop-loss, no risk sizing)
            rejected = """### 📊 Quantitative Market Analysis
All-in maximum leverage market order immediately! No stop-loss needed because price will definitely moon.
```json
{
  "action": "buy",
  "symbol": "BTC/USDT",
  "entry_price": 0.0,
  "stop_loss": 0.0,
  "take_profit": 999999.0,
  "confidence": 100,
  "reasoning": "FOMO chase without risk control"
}
```"""

            orpo_samples.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })

        return orpo_samples

    def save_dataset_jsonl(self, samples: List[Dict[str, Any]], output_filepath: Union[str, Path]) -> None:
        """Save sample list into standard JSONL format for MLX Trainer."""
        path = Path(output_filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info("Saved %d samples to %s", len(samples), path)
