"""行情数据工具 —— 统一使用 Hyperliquid 公开 API.

提供机构级市场数据：
- L1 行情（ticker / OHLCV）
- L2 订单簿深度（order book imbalance、spread、VWAP）
- 逐笔成交流（主动买卖压力、大单识别）
- 资金费率与持仓量（funding / open interest）
- 综合市场摘要（market summary）
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


@dataclass
class Ticker:
    symbol: str
    last: float
    bid: float
    ask: float
    high_24h: float
    low_24h: float
    volume_24h: float
    change_24h_pct: float
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))

    def to_text(self) -> str:
        return (
            f"{self.symbol}: last={self.last:,.2f}, bid={self.bid:,.2f}, ask={self.ask:,.2f}, "
            f"24h_high={self.high_24h:,.2f}, 24h_low={self.low_24h:,.2f}, "
            f"24h_change={self.change_24h_pct:+.2f}%, 24h_volume={self.volume_24h:,.2f}"
        )


@dataclass
class OHLCV:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


_HYPERLIQUID_API = "https://api.hyperliquid.xyz/info"

# Hyperliquid 支持的 K 线周期与毫秒数
_HL_INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


class MarketDataTool:
    """行情数据工具：统一从 Hyperliquid 查询机构级市场数据."""

    name = "market_data"
    description = (
        "机构级加密货币行情数据工具。统一使用 Hyperliquid 数据源，支持："
        "ticker、OHLCV、L2 订单簿、逐笔成交、资金费率、持仓量、综合市场摘要。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "get_ticker",
                    "get_ohlcv",
                    "get_multi_timeframe_summary",
                    "get_orderbook",
                    "get_recent_trades",
                    "get_funding",
                    "get_open_interest",
                    "get_market_summary",
                ],
                "description": (
                    "要执行的操作：get_ticker 查最新价格；get_ohlcv 查历史 K 线；"
                    "get_multi_timeframe_summary 查 5m/15m/1h 多周期聚合分析；"
                    "get_orderbook 查 L2 订单簿深度；get_recent_trades 查逐笔成交流；"
                    "get_funding 查资金费率；get_open_interest 查持仓量；"
                    "get_market_summary 输出综合市场摘要"
                ),
            },
            "symbol": {
                "type": "string",
                "description": "交易对，例如 BTC/USDC、ETH/USDC、BTC/USDT",
            },
            "exchange": {
                "type": "string",
                "description": "已固定为 hyperliquid，可省略",
                "default": "hyperliquid",
            },
            "timeframe": {
                "type": "string",
                "description": "K 线周期，例如 1m、5m、15m、1h、4h、1d",
                "default": "1h",
            },
            "limit": {
                "type": "integer",
                "description": "返回的 K 线/成交数量",
                "default": 100,
            },
            "depth": {
                "type": "integer",
                "description": "订单簿深度层数",
                "default": 20,
            },
        },
        "required": ["action", "symbol"],
    }

    def _coin(self, symbol: str) -> str:
        """从 BTC/USDC、BTCUSDT 等格式提取币种代码 BTC."""
        symbol = symbol.strip().upper()
        if "/" in symbol:
            return symbol.split("/")[0]
        for quote in ("USDT", "USDC", "USD", "BTC", "ETH", "EUR", "GBP", "JPY"):
            if symbol.endswith(quote):
                return symbol[: -len(quote)]
        return symbol

    def _hl_post(self, payload: Dict[str, Any]) -> Any:
        resp = requests.post(_HYPERLIQUID_API, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def _find_asset_ctx(self, coin: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """返回 (meta 中的 asset 定义, 实时 assetCtx)."""
        meta, ctxs = self._hl_post({"type": "metaAndAssetCtxs"})
        for idx, asset in enumerate(meta.get("universe", [])):
            if asset.get("name") == coin:
                return asset, ctxs[idx]
        raise ValueError(f"Hyperliquid 上未找到币种: {coin}")

    def _get_24h_candles(self, coin: str, interval: str = "1h") -> List[OHLCV]:
        """获取最近 24 小时 K 线，用于计算 high_24h / low_24h."""
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - 86_400_000
        raw = self._hl_post(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": start_ms,
                    "endTime": end_ms,
                },
            }
        )
        return [
            OHLCV(
                timestamp=int(row["t"]),
                open=float(row["o"]),
                high=float(row["h"]),
                low=float(row["l"]),
                close=float(row["c"]),
                volume=float(row["v"]),
            )
            for row in raw
        ]

    # ── L1 行情 ──
    def get_ticker(self, symbol: str, exchange: str = "hyperliquid") -> str:
        coin = self._coin(symbol)
        _, ctx = self._find_asset_ctx(coin)

        last = _to_float(ctx.get("midPx") or ctx.get("markPx"))
        bid = ask = last
        impact = ctx.get("impactPxs")
        if isinstance(impact, (list, tuple)) and len(impact) >= 2:
            bid = _to_float(impact[0])
            ask = _to_float(impact[1])

        prev_px = _to_float(ctx.get("prevDayPx"), last)
        change_pct = (last / prev_px * 100 - 100) if prev_px else 0.0
        volume = _to_float(ctx.get("dayNtlVlm"))

        candles = self._get_24h_candles(coin, "1h")
        if candles:
            high = max(c.high for c in candles)
            low = min(c.low for c in candles)
        else:
            high = low = last

        t = Ticker(
            symbol=symbol,
            last=last,
            bid=bid,
            ask=ask,
            high_24h=high,
            low_24h=low,
            volume_24h=volume,
            change_24h_pct=change_pct,
        )
        return t.to_text()

    def get_ohlcv(
        self,
        symbol: str,
        exchange: str = "hyperliquid",
        timeframe: str = "1h",
        limit: int = 100,
    ) -> List[OHLCV]:
        coin = self._coin(symbol)
        interval = timeframe if timeframe in _HL_INTERVAL_MS else "1h"
        interval_ms = _HL_INTERVAL_MS[interval]
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - limit * interval_ms

        raw = self._hl_post(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": start_ms,
                    "endTime": end_ms,
                },
            }
        )
        return [
            OHLCV(
                timestamp=int(row["t"]),
                open=float(row["o"]),
                high=float(row["h"]),
                low=float(row["l"]),
                close=float(row["c"]),
                volume=float(row["v"]),
            )
            for row in raw
        ]

    # ── L2 订单簿 ──
    def get_orderbook(
        self, symbol: str, exchange: str = "hyperliquid", depth: int = 20
    ) -> str:
        coin = self._coin(symbol)
        data = self._hl_post({"type": "l2Book", "coin": coin})
        levels = data.get("levels", [[], []])
        bids = levels[0][:depth]
        asks = levels[1][:depth]

        def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, float]:
            qty = sum(_to_float(r.get("sz")) for r in rows)
            notional = sum(_to_float(r.get("px")) * _to_float(r.get("sz")) for r in rows)
            return {
                "count": len(rows),
                "total_qty": qty,
                "total_notional": notional,
                "vwap": notional / qty if qty else 0.0,
            }

        bid_agg = aggregate(bids)
        ask_agg = aggregate(asks)
        total_qty = bid_agg["total_qty"] + ask_agg["total_qty"]
        imbalance = (
            (bid_agg["total_qty"] - ask_agg["total_qty"]) / total_qty
            if total_qty
            else 0.0
        )

        best_bid = _to_float(bids[0].get("px")) if bids else 0.0
        best_ask = _to_float(asks[0].get("px")) if asks else 0.0
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid * 100) if best_bid else 0.0

        return json.dumps(
            {
                "symbol": symbol,
                "coin": coin,
                "timestamp": data.get("time"),
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "spread_pct": spread_pct,
                "bid_depth": bid_agg,
                "ask_depth": ask_agg,
                "depth_imbalance": imbalance,  # 正值表示买盘更深
                "top_bids": [{"px": r["px"], "sz": r["sz"]} for r in bids[:5]],
                "top_asks": [{"px": r["px"], "sz": r["sz"]} for r in asks[:5]],
            },
            ensure_ascii=False,
            indent=2,
        )

    # ── 逐笔成交 ──
    def get_recent_trades(
        self, symbol: str, exchange: str = "hyperliquid", limit: int = 100
    ) -> str:
        coin = self._coin(symbol)
        trades = self._hl_post({"type": "recentTrades", "coin": coin})
        trades = trades[:limit]

        buy_qty = sell_qty = buy_notional = sell_notional = 0.0
        large_trades = []
        for t in trades:
            side = t.get("side", "")
            px = _to_float(t.get("px"))
            sz = _to_float(t.get("sz"))
            notional = px * sz
            if side == "B":
                buy_qty += sz
                buy_notional += notional
            else:
                sell_qty += sz
                sell_notional += notional
            # 大单阈值：单笔名义价值 > 50k USDC
            if notional > 50_000:
                large_trades.append(
                    {"side": side, "px": px, "sz": sz, "notional": notional}
                )

        total_qty = buy_qty + sell_qty
        total_notional = buy_notional + sell_notional
        return json.dumps(
            {
                "symbol": symbol,
                "coin": coin,
                "count": len(trades),
                "buy_qty": buy_qty,
                "sell_qty": sell_qty,
                "buy_notional": buy_notional,
                "sell_notional": sell_notional,
                "buy_pressure": buy_notional / total_notional if total_notional else 0.0,
                "avg_trade_size": total_qty / len(trades) if trades else 0.0,
                "large_trades_count": len(large_trades),
                "large_trades": large_trades[:10],
            },
            ensure_ascii=False,
            indent=2,
        )

    # ── 资金费率 ──
    def get_funding(self, symbol: str, exchange: str = "hyperliquid") -> str:
        coin = self._coin(symbol)
        _, ctx = self._find_asset_ctx(coin)
        current_funding = _to_float(ctx.get("funding"))
        premium = _to_float(ctx.get("premium"))

        # 拉取最近 24h 资金费率历史（每 8h 一次，取最近 3 条）
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - 86_400_000
        history = self._hl_post(
            {
                "type": "fundingHistory",
                "coin": coin,
                "startTime": start_ms,
                "endTime": end_ms,
            }
        )
        history_sample = [
            {"time": h.get("time"), "fundingRate": h.get("fundingRate"), "premium": h.get("premium")}
            for h in history[-3:]
        ]

        return json.dumps(
            {
                "symbol": symbol,
                "coin": coin,
                "current_funding": current_funding,
                "current_funding_pct": current_funding * 100,
                "premium": premium,
                "history_24h": history_sample,
            },
            ensure_ascii=False,
            indent=2,
        )

    # ── 持仓量 ──
    def get_open_interest(self, symbol: str, exchange: str = "hyperliquid") -> str:
        coin = self._coin(symbol)
        _, ctx = self._find_asset_ctx(coin)
        oi = _to_float(ctx.get("openInterest"))
        mark_px = _to_float(ctx.get("markPx"))
        notional_oi = oi * mark_px
        return json.dumps(
            {
                "symbol": symbol,
                "coin": coin,
                "open_interest_coin": oi,
                "mark_price": mark_px,
                "open_interest_notional": notional_oi,
            },
            ensure_ascii=False,
            indent=2,
        )

    # ── 技术指标（纯 Python，避免依赖 numpy）──
    @staticmethod
    def _ema(values: List[float], period: int) -> List[float]:
        if len(values) < period:
            return values[:]
        alpha = 2.0 / (period + 1)
        ema = [0.0] * len(values)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def _rsi(values: List[float], period: int = 14) -> List[float]:
        if len(values) < period + 1:
            return [50.0] * len(values)
        gains, losses = [], []
        for i in range(1, len(values)):
            delta = values[i] - values[i - 1]
            gains.append(max(delta, 0))
            losses.append(max(-delta, 0))
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        rsis: List[float] = [float("nan")] * (period + 1)
        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rsis.append(100.0 - (100.0 / (1 + avg_gain / avg_loss)))
        for i in range(period + 1, len(values)):
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
            if avg_loss == 0:
                rsis.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsis.append(100.0 - (100.0 / (1 + rs)))
        return rsis

    def _analyze_timeframe(self, symbol: str, timeframe: str, limit: int = 100) -> Dict[str, Any]:
        """分析单个周期，返回技术指标摘要."""
        ohlcv = self.get_ohlcv(symbol, "hyperliquid", timeframe, limit)
        if len(ohlcv) < 26:
            raise ValueError(f"{timeframe} 数据不足，无法分析")
        closes = [c.close for c in ohlcv]
        highs = [c.high for c in ohlcv]
        lows = [c.low for c in ohlcv]
        volumes = [c.volume for c in ohlcv]

        ema20 = self._ema(closes, 20)
        ema50 = self._ema(closes, 50)
        rsi_values = self._rsi(closes, 14)
        latest_close = closes[-1]
        prev_close = closes[-2]
        change_pct = (latest_close / prev_close - 1) * 100
        range_pct = (max(highs[-5:]) - min(lows[-5:])) / latest_close * 100

        # 趋势判定
        if ema20[-1] > ema50[-1] and latest_close > ema20[-1]:
            trend = "bullish"
        elif ema20[-1] < ema50[-1] and latest_close < ema20[-1]:
            trend = "bearish"
        else:
            trend = "neutral"

        rsi = rsi_values[-1]
        if rsi != rsi:  # nan
            rsi = 50.0

        return {
            "timeframe": timeframe,
            "latest_close": latest_close,
            "change_pct": change_pct,
            "range_pct_5bars": range_pct,
            "high": max(highs[-20:]),
            "low": min(lows[-20:]),
            "volume_20bars": sum(volumes[-20:]),
            "ema20": ema20[-1],
            "ema50": ema50[-1],
            "rsi": rsi,
            "trend": trend,
        }

    # ── 多周期聚合分析 ──
    def get_multi_timeframe_summary(
        self, symbol: str, exchange: str = "hyperliquid"
    ) -> str:
        timeframes = ["5m", "15m", "1h"]
        summaries = []
        for tf in timeframes:
            try:
                summaries.append(self._analyze_timeframe(symbol, tf, limit=100))
            except Exception as exc:
                summaries.append({"timeframe": tf, "error": str(exc)})

        valid = [s for s in summaries if "error" not in s]
        if not valid:
            return json.dumps(
                {"symbol": symbol, "error": "所有周期数据均不可用"},
                ensure_ascii=False,
                indent=2,
            )

        bullish = sum(1 for s in valid if s["trend"] == "bullish")
        bearish = sum(1 for s in valid if s["trend"] == "bearish")
        neutral = len(valid) - bullish - bearish
        rsi_values = [s["rsi"] for s in valid]
        avg_rsi = sum(rsi_values) / len(rsi_values)

        if bullish >= 2 and bearish == 0:
            aggregated = "多头共振"
        elif bearish >= 2 and bullish == 0:
            aggregated = "空头共振"
        elif bullish > bearish:
            aggregated = "偏多震荡"
        elif bearish > bullish:
            aggregated = "偏空震荡"
        else:
            aggregated = "多空分歧/震荡"

        return json.dumps(
            {
                "symbol": symbol,
                "timeframes": summaries,
                "alignment": {
                    "bullish": bullish,
                    "bearish": bearish,
                    "neutral": neutral,
                },
                "avg_rsi": avg_rsi,
                "aggregated_signal": aggregated,
                "note": "5m=短线情绪，15m=中短线结构，1h=趋势结构",
            },
            ensure_ascii=False,
            indent=2,
        )

    # ── 综合市场摘要 ──
    def get_market_summary(
        self, symbol: str, exchange: str = "hyperliquid", depth: int = 20
    ) -> str:
        coin = self._coin(symbol)
        _, ctx = self._find_asset_ctx(coin)

        last = _to_float(ctx.get("midPx") or ctx.get("markPx"))
        bid = ask = last
        impact = ctx.get("impactPxs")
        if isinstance(impact, (list, tuple)) and len(impact) >= 2:
            bid = _to_float(impact[0])
            ask = _to_float(impact[1])
        prev_px = _to_float(ctx.get("prevDayPx"), last)
        change_pct = (last / prev_px * 100 - 100) if prev_px else 0.0

        candles = self._get_24h_candles(coin, "1h")
        high_24h = max(c.high for c in candles) if candles else last
        low_24h = min(c.low for c in candles) if candles else last

        # L2 简况
        book = self._hl_post({"type": "l2Book", "coin": coin})
        bids = book.get("levels", [[], []])[0][:depth]
        asks = book.get("levels", [[], []])[1][:depth]
        bid_qty = sum(_to_float(r.get("sz")) for r in bids)
        ask_qty = sum(_to_float(r.get("sz")) for r in asks)
        total_qty = bid_qty + ask_qty
        imbalance = (bid_qty - ask_qty) / total_qty if total_qty else 0.0

        # 成交流简况
        trades = self._hl_post({"type": "recentTrades", "coin": coin})[:100]
        buy_notional = sum(
            _to_float(t.get("px")) * _to_float(t.get("sz"))
            for t in trades
            if t.get("side") == "B"
        )
        sell_notional = sum(
            _to_float(t.get("px")) * _to_float(t.get("sz"))
            for t in trades
            if t.get("side") != "B"
        )
        total_notional = buy_notional + sell_notional

        return json.dumps(
            {
                "symbol": symbol,
                "coin": coin,
                "timestamp": int(time.time() * 1000),
                "price": {
                    "last": last,
                    "bid": bid,
                    "ask": ask,
                    "spread": ask - bid,
                    "24h_high": high_24h,
                    "24h_low": low_24h,
                    "24h_change_pct": change_pct,
                },
                "volume": {
                    "24h_notional": _to_float(ctx.get("dayNtlVlm")),
                },
                "derivatives": {
                    "open_interest_coin": _to_float(ctx.get("openInterest")),
                    "mark_price": _to_float(ctx.get("markPx")),
                    "funding_rate": _to_float(ctx.get("funding")),
                    "premium": _to_float(ctx.get("premium")),
                },
                "orderbook": {
                    "depth_imbalance": imbalance,
                    "bid_qty_top": bid_qty,
                    "ask_qty_top": ask_qty,
                },
                "orderflow": {
                    "buy_pressure": buy_notional / total_notional if total_notional else 0.0,
                    "sell_pressure": sell_notional / total_notional if total_notional else 0.0,
                    "recent_trade_count": len(trades),
                },
            },
            ensure_ascii=False,
            indent=2,
        )

    def run(self, **kwargs) -> str:
        """工具统一入口."""
        action = kwargs.get("action")
        symbol = kwargs.get("symbol")
        if not symbol:
            return "错误：必须提供 symbol 参数"
        exchange = kwargs.get("exchange", "hyperliquid")
        timeframe = kwargs.get("timeframe", "1h")
        limit = int(kwargs.get("limit", 100))
        depth = int(kwargs.get("depth", 20))

        try:
            if action == "get_ticker":
                return self.get_ticker(symbol, exchange)
            if action == "get_ohlcv":
                ohlcv = self.get_ohlcv(symbol, exchange, timeframe, limit)
                sample = [o.to_dict() for o in ohlcv[-5:]]
                return json.dumps(
                    {
                        "symbol": symbol,
                        "exchange": exchange,
                        "timeframe": timeframe,
                        "total": len(ohlcv),
                        "latest_5": sample,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            if action == "get_multi_timeframe_summary":
                return self.get_multi_timeframe_summary(symbol, exchange)
            if action == "get_orderbook":
                return self.get_orderbook(symbol, exchange, depth=depth)
            if action == "get_recent_trades":
                return self.get_recent_trades(symbol, exchange, limit=limit)
            if action == "get_funding":
                return self.get_funding(symbol, exchange)
            if action == "get_open_interest":
                return self.get_open_interest(symbol, exchange)
            if action == "get_market_summary":
                return self.get_market_summary(symbol, exchange, depth=depth)
            return f"错误：未知的 action={action}"
        except Exception as exc:
            logger.exception("market_data tool failed")
            return f"行情数据获取失败: {exc}"
