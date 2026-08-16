"""行情数据工具 —— 统一使用 Hyperliquid 公开 API.

提供机构级市场数据：
- L1 行情（ticker / OHLCV）
- L2 订单簿深度（order book imbalance、spread、VWAP、滑点估计）
- 逐笔成交流（CVD 多窗口、大单识别）
- 资金费率与持仓量（含 ΔOI 本地快照）
- ATR / ADX / Volume Profile（POC/VAH/VAL）
- 综合市场摘要
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from xmlx_vlm.ai_trader.config import LOGS_DIR

logger = logging.getLogger(__name__)


# ── 可选：常驻行情服务（WebSocket 内存状态机）──
_SERVICE_ENABLED = os.environ.get("XMLX_VLM_AI_TRADER_WS", "1") == "1"
_SERVICE_INSTANCE: Optional[Any] = None
_SERVICE_LOCK = threading.Lock()


def _get_live_service() -> Optional[Any]:
    """懒加载并启动常驻行情服务；失败时返回 None，工具自动回退到 REST."""
    if not _SERVICE_ENABLED:
        return None
    global _SERVICE_INSTANCE
    if _SERVICE_INSTANCE is not None:
        return _SERVICE_INSTANCE
    with _SERVICE_LOCK:
        if _SERVICE_INSTANCE is not None:
            return _SERVICE_INSTANCE
        try:
            from xmlx_vlm.ai_trader.market_service import MarketDataService

            svc = MarketDataService()
            svc.start()
            _SERVICE_INSTANCE = svc
            logger.info("MarketDataService started for live market data")
            return svc
        except Exception as exc:
            logger.warning("MarketDataService start failed, fallback to REST: %s", exc)
            return None


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
            f"{self.symbol}: mark={self.last:,.2f}, bid={self.bid:,.2f}, ask={self.ask:,.2f}, "
            f"24h_high={self.high_24h:,.2f}, 24h_low={self.low_24h:,.2f}, "
            f"24h_change={self.change_24h_pct:+.2f}%, 24h_volume={_format_notional(self.volume_24h)}"
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


# ── 通用工具函数 ──

def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _format_notional(value: float) -> str:
    """把大数字格式化为 B/M/K，避免模型读错数量级."""
    value = _to_float(value)
    abs_v = abs(value)
    if abs_v >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if abs_v >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs_v >= 1_000:
        return f"{value / 1_000:.2f}K"
    return f"{value:.2f}"


def _price_ref(ctx: Dict[str, Any]) -> float:
    """统一使用 markPx 作为价格基准."""
    return _to_float(ctx.get("markPx") or ctx.get("midPx"))


# ── OI / Funding 本地快照追踪 ──

class _OITracker:
    """本地持久化 OI/funding 快照，用于计算 ΔOI."""

    def __init__(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._path = LOGS_DIR / "oi_tracker.json"
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("OI tracker load failed: %s", exc)
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        try:
            self._path.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("OI tracker save failed: %s", exc)

    def record(self, coin: str, ctx: Dict[str, Any]):
        now = int(time.time() * 1000)
        entry = {
            "t": now,
            "oi": _to_float(ctx.get("openInterest")),
            "mark_px": _to_float(ctx.get("markPx")),
            "funding": _to_float(ctx.get("funding")),
            "premium": _to_float(ctx.get("premium")),
        }
        self._data.setdefault(coin, []).append(entry)
        # 清理 7 天前的快照；兼容旧数据无 t 字段
        cutoff = now - 7 * 86_400_000
        self._data[coin] = [e for e in self._data[coin] if e.get("t", now) > cutoff]
        self._save()

    def delta(self, coin: str, minutes: int) -> Optional[float]:
        """返回相比 N 分钟前 OI 的百分比变化."""
        rows = self._data.get(coin, [])
        if not rows:
            return None
        now = int(time.time() * 1000)
        threshold = now - minutes * 60_000
        # 找到最后一个早于 threshold 的快照；兼容旧数据无 t 字段
        past = [r for r in rows if r.get("t") and r["t"] <= threshold]
        if not past:
            return None
        old = past[-1]
        current = rows[-1]
        if old["oi"] == 0:
            return None
        return (current["oi"] - old["oi"]) / old["oi"] * 100


# ── 技术指标（纯 Python） ──

def _ema(values: List[float], period: int) -> List[float]:
    if len(values) < period:
        return values[:]
    alpha = 2.0 / (period + 1)
    out = [0.0] * len(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < period + 1:
        return [50.0] * len(values)
    gains, losses = [], []
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
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


def _atr(ohlcv: List[OHLCV], period: int = 14) -> List[float]:
    if len(ohlcv) < period + 1:
        return []
    trs = [0.0]
    for i in range(1, len(ohlcv)):
        c = ohlcv[i]
        p = ohlcv[i - 1]
        tr = max(
            c.high - c.low,
            abs(c.high - p.close),
            abs(c.low - p.close),
        )
        trs.append(tr)
    atr = _ema(trs, period)
    return atr


def _adx(ohlcv: List[OHLCV], period: int = 14) -> Tuple[List[float], List[float], List[float]]:
    """返回 (adx, +DI, -DI)."""
    n = len(ohlcv)
    plus_dm = [0.0]
    minus_dm = [0.0]
    trs = [0.0]
    for i in range(1, n):
        up = ohlcv[i].high - ohlcv[i - 1].high
        down = ohlcv[i - 1].low - ohlcv[i].low
        plus_dm.append(max(up, 0) if up > down else 0.0)
        minus_dm.append(max(down, 0) if down > up else 0.0)
        trs.append(
            max(
                ohlcv[i].high - ohlcv[i].low,
                abs(ohlcv[i].high - ohlcv[i - 1].close),
                abs(ohlcv[i].low - ohlcv[i - 1].close),
            )
        )
    atr = _ema(trs, period)
    plus_di = [0.0] * n
    minus_di = [0.0] * n
    for i in range(n):
        if atr[i] != 0:
            plus_di[i] = plus_dm[i] / atr[i] * 100
            minus_di[i] = minus_dm[i] / atr[i] * 100
    dx = [0.0] * n
    for i in range(n):
        s = plus_di[i] + minus_di[i]
        dx[i] = abs(plus_di[i] - minus_di[i]) / s * 100 if s else 0.0
    adx = _ema(dx, period)
    return adx, plus_di, minus_di


def _volume_profile(ohlcv: List[OHLCV], bins: int = 24) -> Dict[str, Any]:
    """基于收盘价分布计算 POC、VAH、VAL（70% 成交量价值区）."""
    if not ohlcv:
        return {"poc": 0.0, "vah": 0.0, "val": 0.0}
    closes = [c.close for c in ohlcv]
    volumes = [c.volume for c in ohlcv]
    min_p, max_p = min(closes), max(closes)
    if min_p == max_p or bins <= 0:
        return {"poc": closes[-1], "vah": closes[-1], "val": closes[-1]}
    bin_edges = [min_p + (max_p - min_p) * i / bins for i in range(bins + 1)]
    bin_volumes = [0.0] * bins
    bin_prices = []
    for i in range(bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        bin_prices.append((lo + hi) / 2)
        for c, v in zip(closes, volumes):
            if (i < bins - 1 and lo <= c < hi) or (i == bins - 1 and lo <= c <= hi):
                bin_volumes[i] += v
    max_vol = max(bin_volumes)
    poc_idx = bin_volumes.index(max_vol)
    poc = bin_prices[poc_idx]

    # 价值区：从 POC 向两边扩展，直到覆盖 70% 成交量
    total_vol = sum(bin_volumes)
    target = total_vol * 0.70
    cum = max_vol
    low_idx = high_idx = poc_idx
    while cum < target and (low_idx > 0 or high_idx < bins - 1):
        left_vol = bin_volumes[low_idx - 1] if low_idx > 0 else 0.0
        right_vol = bin_volumes[high_idx + 1] if high_idx < bins - 1 else 0.0
        if left_vol >= right_vol and low_idx > 0:
            low_idx -= 1
            cum += left_vol
        elif high_idx < bins - 1:
            high_idx += 1
            cum += right_vol
        else:
            break
    return {
        "poc": poc,
        "vah": bin_prices[high_idx],
        "val": bin_prices[low_idx],
        "coverage_pct": cum / total_vol * 100 if total_vol else 0.0,
    }


def _structure(ohlcv: List[OHLCV], lookback: int = 20) -> str:
    """识别更高低点/更低高点结构."""
    recent = ohlcv[-lookback:]
    if len(recent) < 4:
        return "insufficient_data"
    hh_hl = 0
    lh_ll = 0
    for i in range(2, len(recent)):
        if recent[i].high > recent[i - 1].high and recent[i].low > recent[i - 1].low:
            hh_hl += 1
        elif recent[i].high < recent[i - 1].high and recent[i].low < recent[i - 1].low:
            lh_ll += 1
    if hh_hl > lh_ll * 1.5 and hh_hl >= 2:
        return "higher_highs_higher_lows"
    if lh_ll > hh_hl * 1.5 and lh_ll >= 2:
        return "lower_highs_lower_lows"
    return "range"


def _slippage(levels: List[Dict[str, Any]], side: str, notional: float) -> Dict[str, Any]:
    """Walk the book 估算指定名义金额的冲击成本. side='buy' 吃卖盘，'sell' 吃买盘."""
    remaining = notional
    filled_qty = 0.0
    for lvl in levels:
        px = _to_float(lvl.get("px"))
        sz = _to_float(lvl.get("sz"))
        if px <= 0 or sz <= 0:
            continue
        max_notional = px * sz
        take = min(remaining, max_notional)
        filled_qty += take / px
        remaining -= take
        if remaining <= 0:
            break
    if filled_qty <= 0:
        return {"avg_px": 0.0, "slippage_pct": None, "filled": False}
    avg_px = (notional - remaining) / filled_qty
    first_px = _to_float(levels[0].get("px")) if levels else avg_px
    slippage = (avg_px - first_px) / first_px * 100 if first_px and side == "buy" else (first_px - avg_px) / first_px * 100
    return {
        "avg_px": avg_px,
        "slippage_pct": slippage,
        "filled": remaining <= 0,
        "remaining_notional": remaining,
    }


class MarketDataTool:
    """行情数据工具：统一从 Hyperliquid 查询机构级市场数据."""

    name = "market_data"
    description = (
        "机构级加密货币行情数据工具。统一使用 Hyperliquid 数据源，支持："
        "ticker、OHLCV、L2 订单簿（含滑点）、多窗口 CVD 成交流、"
        "资金费率与持仓量（含 ΔOI）、ATR/ADX/Volume Profile、综合市场摘要。"
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
                    "get_trade_flow",
                    "get_funding",
                    "get_open_interest",
                    "get_market_summary",
                ],
                "description": (
                    "要执行的操作：get_ticker 查最新价格；get_ohlcv 查历史 K 线；"
                    "get_multi_timeframe_summary 查 5m/15m/1h 多周期聚合分析；"
                    "get_orderbook 查 L2 订单簿深度与滑点；"
                    "get_recent_trades 查逐笔成交；get_trade_flow 查 15m/1h/4h CVD；"
                    "get_funding 查资金费率趋势；get_open_interest 查持仓量含 ΔOI；"
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

    def __init__(self):
        self._oi_tracker = _OITracker()

    def _coin(self, symbol: str) -> str:
        """从 BTC/USDC、BTCUSDT 等格式提取币种代码 BTC (使用统一的 extract_base_coin SSOT)."""
        from xmlx_vlm.ai_trader.oms.utils.symbol import extract_base_coin
        return extract_base_coin(symbol)

    def _hl_post(self, payload: Dict[str, Any]) -> Any:
        resp = requests.post(_HYPERLIQUID_API, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def _find_asset_ctx(self, coin: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        meta, ctxs = self._hl_post({"type": "metaAndAssetCtxs"})
        for idx, asset in enumerate(meta.get("universe", [])):
            asset_name = asset.get("name", "")
            if asset_name == coin or asset_name.upper() == coin.upper():
                return asset, ctxs[idx]
        raise ValueError(f"Hyperliquid 上未找到币种: {coin}")

    def _get_24h_candles(self, coin: str, interval: str = "1h") -> List[OHLCV]:
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

        # 优先使用常驻行情服务的内存状态
        svc = _get_live_service()
        if svc is not None:
            try:
                summary = svc.get_summary(symbol)
                if summary is not None and summary.mark_price:
                    t = Ticker(
                        symbol=symbol,
                        last=summary.mark_price,
                        bid=summary.bid,
                        ask=summary.ask,
                        high_24h=summary.high_24h or summary.mark_price,
                        low_24h=summary.low_24h or summary.mark_price,
                        volume_24h=summary.volume_24h,
                        change_24h_pct=summary.change_24h_pct,
                    )
                    return t.to_text()
            except Exception as exc:
                logger.debug("Live service ticker failed, fallback to REST: %s", exc)

        _, ctx = self._find_asset_ctx(coin)

        last = _price_ref(ctx)
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

        # 优先使用常驻行情服务的内存订单簿
        svc = _get_live_service()
        if svc is not None:
            try:
                book = svc.state.get(coin).get_book()
                if book is not None:
                    bids = [{"px": l.price, "sz": l.size} for l in book.bids[:depth]]
                    asks = [{"px": l.price, "sz": l.size} for l in book.asks[:depth]]
                    return self._format_orderbook(symbol, coin, bids, asks, book.timestamp_ms)
            except Exception as exc:
                logger.debug("Live service orderbook failed, fallback to REST: %s", exc)

        data = self._hl_post({"type": "l2Book", "coin": coin})
        levels = data.get("levels", [[], []])
        bids = levels[0][:depth]
        asks = levels[1][:depth]
        return self._format_orderbook(symbol, coin, bids, asks, data.get("time"))

    def _format_orderbook(
        self,
        symbol: str,
        coin: str,
        bids: List[Dict[str, Any]],
        asks: List[Dict[str, Any]],
        timestamp: Any,
    ) -> str:
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

        slippage_buy_50k = _slippage(asks, "buy", 50_000)
        slippage_buy_200k = _slippage(asks, "buy", 200_000)
        slippage_sell_50k = _slippage(bids, "sell", 50_000)
        slippage_sell_200k = _slippage(bids, "sell", 200_000)

        return json.dumps(
            {
                "symbol": symbol,
                "coin": coin,
                "timestamp": timestamp,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "spread": spread,
                "spread_pct": spread_pct,
                "bid_depth": bid_agg,
                "ask_depth": ask_agg,
                "depth_imbalance": imbalance,
                "top_bids": [{"px": r["px"], "sz": r["sz"]} for r in bids[:5]],
                "top_asks": [{"px": r["px"], "sz": r["sz"]} for r in asks[:5]],
                "slippage": {
                    "buy_50k": slippage_buy_50k,
                    "buy_200k": slippage_buy_200k,
                    "sell_50k": slippage_sell_50k,
                    "sell_200k": slippage_sell_200k,
                },
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

    # ── CVD 多窗口成交流 ──
    def _cvd_window(self, coin: str, minutes: int, whale_threshold: float = 50_000) -> Dict[str, Any]:
        trades = self._hl_post({"type": "recentTrades", "coin": coin})
        now = int(time.time() * 1000)
        cutoff = now - minutes * 60_000
        filtered = [t for t in trades if t.get("time", 0) >= cutoff]

        buy_qty = sell_qty = buy_notional = sell_notional = 0.0
        whale_buy = whale_sell = 0.0
        for t in filtered:
            side = t.get("side", "")
            px = _to_float(t.get("px"))
            sz = _to_float(t.get("sz"))
            notional = px * sz
            if side == "B":
                buy_qty += sz
                buy_notional += notional
                if notional >= whale_threshold:
                    whale_buy += notional
            else:
                sell_qty += sz
                sell_notional += notional
                if notional >= whale_threshold:
                    whale_sell += notional
        total_notional = buy_notional + sell_notional
        return {
            "window_minutes": minutes,
            "trade_count": len(filtered),
            "buy_notional": buy_notional,
            "sell_notional": sell_notional,
            "cvd_notional": buy_notional - sell_notional,
            "cvd_coin": buy_qty - sell_qty,
            "buy_pressure": buy_notional / total_notional if total_notional else 0.0,
            "sell_pressure": sell_notional / total_notional if total_notional else 0.0,
            "whale_buy": whale_buy,
            "whale_sell": whale_sell,
            "whale_net": whale_buy - whale_sell,
        }

    def get_trade_flow(self, symbol: str, exchange: str = "hyperliquid") -> str:
        coin = self._coin(symbol)
        return json.dumps(
            {
                "symbol": symbol,
                "coin": coin,
                "windows": {
                    "15m": self._cvd_window(coin, 15),
                    "1h": self._cvd_window(coin, 60),
                    "4h": self._cvd_window(coin, 240),
                },
                "note": "CVD = 主动买入 - 主动卖出；whale 阈值 50k USDC",
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
        rates = [_to_float(h.get("fundingRate")) for h in history]
        avg_24h = sum(rates) / len(rates) if rates else current_funding
        vs_avg = current_funding - avg_24h

        trend = "stable"
        if len(rates) >= 2:
            first = rates[0]
            last = rates[-1]
            if last > first * 1.2:
                trend = "rising"
            elif last < first * 0.8:
                trend = "falling"

        history_sample = [
            {"time": h.get("time"), "fundingRate": h.get("fundingRate"), "premium": h.get("premium")}
            for h in history[-3:]
        ]

        return json.dumps(
            {
                "symbol": symbol,
                "coin": coin,
                "current_funding_8h": current_funding,
                "current_funding_8h_pct": current_funding * 100,
                "premium": premium,
                "avg_24h": avg_24h,
                "vs_avg": vs_avg,
                "trend": trend,
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

        self._oi_tracker.record(coin, ctx)
        delta_1h = self._oi_tracker.delta(coin, 60)
        delta_24h = self._oi_tracker.delta(coin, 24 * 60)

        return json.dumps(
            {
                "symbol": symbol,
                "coin": coin,
                "open_interest_coin": oi,
                "mark_price": mark_px,
                "open_interest_notional": notional_oi,
                "open_interest_notional_fmt": _format_notional(notional_oi),
                "oi_change_1h_pct": delta_1h,
                "oi_change_24h_pct": delta_24h,
            },
            ensure_ascii=False,
            indent=2,
        )

    # ── 技术指标 ──
    def _analyze_timeframe(
        self, symbol: str, timeframe: str, limit: int = 100, ohlcv: Optional[List[OHLCV]] = None
    ) -> Dict[str, Any]:
        if ohlcv is None:
            ohlcv = self.get_ohlcv(symbol, "hyperliquid", timeframe, limit)
        if len(ohlcv) < 26:
            raise ValueError(f"{timeframe} 数据不足，无法分析")
        closes = [c.close for c in ohlcv]
        highs = [c.high for c in ohlcv]
        lows = [c.low for c in ohlcv]
        volumes = [c.volume for c in ohlcv]

        ema20 = _ema(closes, 20)
        ema50 = _ema(closes, 50)
        rsi_values = _rsi(closes, 14)
        atr_values = _atr(ohlcv, 14)
        adx_values, plus_di_values, minus_di_values = _adx(ohlcv, 14)
        vp = _volume_profile(ohlcv, bins=24)

        latest_close = closes[-1]
        prev_close = closes[-2]
        change_pct = (latest_close / prev_close - 1) * 100
        atr = atr_values[-1]
        atr_pct = atr / latest_close * 100 if latest_close else 0.0
        adx = adx_values[-1]
        plus_di = plus_di_values[-1]
        minus_di = minus_di_values[-1]
        rsi = rsi_values[-1]
        if rsi != rsi:
            rsi = 50.0

        # 趋势判定：结合 ADX 强度、DI 方向、EMA 排列
        trend = "neutral"
        if adx >= 25:
            if plus_di > minus_di and latest_close > ema20[-1] and ema20[-1] > ema50[-1]:
                trend = "bullish"
            elif minus_di > plus_di and latest_close < ema20[-1] and ema20[-1] < ema50[-1]:
                trend = "bearish"

        structure = _structure(ohlcv, lookback=20)

        return {
            "timeframe": timeframe,
            "latest_close": latest_close,
            "change_pct": change_pct,
            "range_pct_5bars": (max(highs[-5:]) - min(lows[-5:])) / latest_close * 100,
            "high_20bars": max(highs[-20:]),
            "low_20bars": min(lows[-20:]),
            "volume_20bars": sum(volumes[-20:]),
            "ema20": ema20[-1],
            "ema50": ema50[-1],
            "rsi": rsi,
            "atr14": atr,
            "atr_pct": atr_pct,
            "adx14": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "volume_profile": vp,
            "trend": trend,
            "structure": structure,
        }

    # ── 多周期聚合分析 ──
    def get_multi_timeframe_summary(
        self, symbol: str, exchange: str = "hyperliquid"
    ) -> str:
        timeframes = ["5m", "15m", "1h"]
        summaries = []

        # 优先使用常驻行情服务的内存 OHLCV
        svc = _get_live_service()
        service_ohlcv: Dict[str, List[OHLCV]] = {}
        if svc is not None:
            try:
                coin = self._coin(symbol)
                state = svc.state.get(coin)
                for tf in timeframes:
                    svc_bars = state.get_ohlcv(tf, limit=100)
                    if len(svc_bars) >= 26:
                        service_ohlcv[tf] = [
                            OHLCV(
                                timestamp=b.timestamp_ms,
                                open=b.open,
                                high=b.high,
                                low=b.low,
                                close=b.close,
                                volume=b.volume,
                            )
                            for b in svc_bars
                        ]
            except Exception as exc:
                logger.debug("Live service multi-timeframe failed, fallback: %s", exc)

        for tf in timeframes:
            try:
                ohlcv = service_ohlcv.get(tf)
                summaries.append(self._analyze_timeframe(symbol, tf, limit=100, ohlcv=ohlcv))
            except Exception as exc:
                summaries.append({"timeframe": tf, "error": str(exc)})

        valid = [s for s in summaries if "error" not in s]
        if not valid:
            return json.dumps(
                {"symbol": symbol, "error": "所有周期数据均不可用"},
                ensure_ascii=False,
                indent=2,
            )

        # 加权投票：1h=3, 15m=2, 5m=1
        weights = {"1h": 3, "15m": 2, "5m": 1}
        score = 0.0
        total_weight = 0.0
        for s in valid:
            w = weights.get(s["timeframe"], 1)
            total_weight += w
            if s["trend"] == "bullish":
                score += w
            elif s["trend"] == "bearish":
                score -= w

        bullish = sum(1 for s in valid if s["trend"] == "bullish")
        bearish = sum(1 for s in valid if s["trend"] == "bearish")
        neutral = len(valid) - bullish - bearish

        avg_rsi = sum(s["rsi"] for s in valid) / len(valid)
        avg_adx = sum(s["adx14"] for s in valid) / len(valid)
        avg_atr_pct = sum(s["atr_pct"] for s in valid) / len(valid)

        if score >= 2:
            aggregated = "多头共振"
        elif score <= -2:
            aggregated = "空头共振"
        elif score > 0:
            aggregated = "偏多震荡"
        elif score < 0:
            aggregated = "偏空震荡"
        else:
            aggregated = "多空分歧/震荡"

        trend_strength = "strong" if avg_adx >= 25 else "weak"

        return json.dumps(
            {
                "symbol": symbol,
                "timeframes": summaries,
                "alignment": {
                    "bullish": bullish,
                    "bearish": bearish,
                    "neutral": neutral,
                },
                "weighted_score": score,
                "avg_rsi": avg_rsi,
                "avg_adx": avg_adx,
                "avg_atr_pct": avg_atr_pct,
                "trend_strength": trend_strength,
                "aggregated_signal": aggregated,
                "note": "5m=短线情绪，15m=中短线结构，1h=趋势结构；1h 权重最高",
            },
            ensure_ascii=False,
            indent=2,
        )

    def get_summary_object(self, symbol: str) -> Optional[Any]:
        """返回结构化的 MarketSummary 对象（供决策引擎使用）.

        优先使用常驻行情服务；若服务未启动则返回 None。
        """
        svc = _get_live_service()
        if svc is not None:
            try:
                return svc.get_summary(symbol)
            except Exception as exc:
                logger.debug("Live service get_summary failed for %s: %s", symbol, exc)
        return None

    # ── 综合市场摘要 ──
    def get_market_summary(
        self, symbol: str, exchange: str = "hyperliquid", depth: int = 20
    ) -> str:
        coin = self._coin(symbol)

        # 优先读取常驻行情服务的内存状态
        svc = _get_live_service()
        live_quote = live_book = None
        svc_ohlcv_1h: List[OHLCV] = []
        if svc is not None:
            try:
                state = svc.state.get(coin)
                live_quote = state.get_quote()
                live_book = state.get_book()
                svc_bars = state.get_ohlcv("1h", limit=100)
                if len(svc_bars) >= 26:
                    svc_ohlcv_1h = [
                        OHLCV(
                            timestamp=b.timestamp_ms,
                            open=b.open,
                            high=b.high,
                            low=b.low,
                            close=b.close,
                            volume=b.volume,
                        )
                        for b in svc_bars
                    ]
            except Exception as exc:
                logger.debug("Live service summary failed, fallback: %s", exc)

        _, ctx = self._find_asset_ctx(coin)

        mark_px = _price_ref(ctx)
        bid = ask = mark_px
        impact = ctx.get("impactPxs")
        if isinstance(impact, (list, tuple)) and len(impact) >= 2:
            bid = _to_float(impact[0])
            ask = _to_float(impact[1])
        # 用服务报价覆盖 REST 的 impact 价格
        if live_quote is not None:
            bid = live_quote.bid or bid
            ask = live_quote.ask or ask
            mark_px = (bid + ask) / 2 if bid and ask else mark_px
        oracle_px = _to_float(ctx.get("oraclePx"), mark_px)
        basis_pct = (mark_px - oracle_px) / oracle_px * 100 if oracle_px else 0.0
        prev_px = _to_float(ctx.get("prevDayPx"), mark_px)
        change_pct = (mark_px / prev_px * 100 - 100) if prev_px else 0.0
        volume_24h = _to_float(ctx.get("dayNtlVlm"))

        candles = self._get_24h_candles(coin, "1h")
        high_24h = max(c.high for c in candles) if candles else mark_px
        low_24h = min(c.low for c in candles) if candles else mark_px
        # 若服务有更长 1h 数据，用服务数据补 24h 高低点
        if svc_ohlcv_1h:
            high_24h = max(high_24h, max(c.high for c in svc_ohlcv_1h))
            low_24h = min(low_24h, min(c.low for c in svc_ohlcv_1h))

        # 1h 技术指标：优先使用服务内存 OHLCV
        try:
            tf_summary = self._analyze_timeframe(
                symbol, "1h", limit=100, ohlcv=svc_ohlcv_1h or None
            )
        except Exception:
            tf_summary = {}

        # L2：优先使用服务内存订单簿
        if live_book is not None:
            bids = [{"px": l.price, "sz": l.size} for l in live_book.bids[:depth]]
            asks = [{"px": l.price, "sz": l.size} for l in live_book.asks[:depth]]
        else:
            book = self._hl_post({"type": "l2Book", "coin": coin})
            bids = book.get("levels", [[], []])[0][:depth]
            asks = book.get("levels", [[], []])[1][:depth]
        bid_qty = sum(_to_float(r.get("sz")) for r in bids)
        ask_qty = sum(_to_float(r.get("sz")) for r in asks)
        total_qty = bid_qty + ask_qty
        imbalance = (bid_qty - ask_qty) / total_qty if total_qty else 0.0

        # CVD：优先使用服务内存逐笔成交
        cvd_15m = cvd_1h = None
        if svc is not None:
            try:
                state = svc.state.get(coin)
                cvd_15m = state.cvd_window(15)
                cvd_1h = state.cvd_window(60)
            except Exception:
                pass
        if cvd_15m is None:
            cvd_15m = self._cvd_window(coin, 15)
        if cvd_1h is None:
            cvd_1h = self._cvd_window(coin, 60)

        # OI + funding：Hyperliquid WS 不直接推送 OI，仍通过 REST 获取
        oi = _to_float(ctx.get("openInterest"))
        notional_oi = oi * mark_px
        self._oi_tracker.record(coin, ctx)
        oi_delta_1h = self._oi_tracker.delta(coin, 60)
        oi_delta_24h = self._oi_tracker.delta(coin, 24 * 60)

        end_ms = int(time.time() * 1000)
        start_ms = end_ms - 86_400_000
        funding_history = self._hl_post(
            {"type": "fundingHistory", "coin": coin, "startTime": start_ms, "endTime": end_ms}
        )
        rates = [_to_float(h.get("fundingRate")) for h in funding_history]
        avg_funding_24h = sum(rates) / len(rates) if rates else _to_float(ctx.get("funding"))
        funding_trend = "stable"
        if len(rates) >= 2:
            if rates[-1] > rates[0] * 1.2:
                funding_trend = "rising"
            elif rates[-1] < rates[0] * 0.8:
                funding_trend = "falling"

        return json.dumps(
            {
                "symbol": symbol,
                "coin": coin,
                "timestamp": int(time.time() * 1000),
                "price": {
                    "mark_price": mark_px,
                    "oracle_price": oracle_px,
                    "basis_pct": basis_pct,
                    "bid": bid,
                    "ask": ask,
                    "spread": ask - bid,
                    "24h_high": high_24h,
                    "24h_low": low_24h,
                    "24h_change_pct": change_pct,
                },
                "volume": {
                    "24h_notional": volume_24h,
                    "24h_notional_fmt": _format_notional(volume_24h),
                },
                "indicators_1h": {
                    "atr14": tf_summary.get("atr14"),
                    "atr_pct": tf_summary.get("atr_pct"),
                    "adx14": tf_summary.get("adx14"),
                    "rsi": tf_summary.get("rsi"),
                    "ema20": tf_summary.get("ema20"),
                    "ema50": tf_summary.get("ema50"),
                    "volume_profile": tf_summary.get("volume_profile"),
                },
                "derivatives": {
                    "open_interest_coin": oi,
                    "open_interest_notional": notional_oi,
                    "open_interest_notional_fmt": _format_notional(notional_oi),
                    "oi_change_1h_pct": oi_delta_1h,
                    "oi_change_24h_pct": oi_delta_24h,
                    "funding_rate_8h": _to_float(ctx.get("funding")),
                    "avg_funding_24h": avg_funding_24h,
                    "funding_trend": funding_trend,
                    "premium": _to_float(ctx.get("premium")),
                },
                "orderbook": {
                    "depth_imbalance": imbalance,
                    "bid_qty_top": bid_qty,
                    "ask_qty_top": ask_qty,
                },
                "orderflow": {
                    "cvd_15m": cvd_15m,
                    "cvd_1h": cvd_1h,
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
            if action == "get_trade_flow":
                return self.get_trade_flow(symbol, exchange)
            if action == "get_funding":
                return self.get_funding(symbol, exchange)
            if action == "get_open_interest":
                return self.get_open_interest(symbol, exchange)
            if action == "get_market_summary":
                return self.get_market_summary(symbol, exchange, depth=depth)
            if action == "get_columnar_series":
                as_of = kwargs.get("as_of")
                as_of_ms = int(as_of) if as_of is not None else None
                from xmlx_vlm.ai_trader.market_service.columnar_store import ColumnarMarketStore
                coin = self._coin(symbol)
                col_data = ColumnarMarketStore.get_instance().query_columnar(
                    symbol=coin,
                    timeframe=timeframe,
                    limit=limit,
                    as_of_ms=as_of_ms,
                )
                return json.dumps(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "as_of_ms": as_of_ms,
                        "row_count": len(col_data.get("timestamp", [])),
                        "columns": col_data,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            if action == "get_point_in_time_snapshot":
                as_of = kwargs.get("as_of")
                as_of_ms = int(as_of) if as_of is not None else None
                from xmlx_vlm.ai_trader.market_service.columnar_store import ColumnarMarketStore
                coin = self._coin(symbol)
                snap = ColumnarMarketStore.get_instance().get_snapshot_as_of(coin, as_of_ms=as_of_ms)
                return json.dumps(
                    {
                        "symbol": symbol,
                        "as_of_ms": as_of_ms,
                        "snapshot": snap,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            return f"错误：未知的 action={action}"
        except Exception as exc:
            logger.exception("market_data tool failed")
            return f"行情数据获取失败: {exc}"
