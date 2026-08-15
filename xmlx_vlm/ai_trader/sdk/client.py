"""AI Trader In-Memory SDK Client.

提供供 Agent（PTC 模式）或策略脚本调用的同步/异步高性能接口。
直接操作本地内存状态机与 OMS，避免多轮网络往返与格式转换。
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Union

from xmlx_vlm.ai_trader.market_service.service import MarketDataService
from xmlx_vlm.ai_trader.tools.market import MarketDataTool
from xmlx_vlm.ai_trader.tools.trading import TradingTool
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal

logger = logging.getLogger(__name__)


class MarketSDK:
    """行情与技术指标 In-Memory 查询接口."""

    def __init__(self, market_tool: Optional[MarketDataTool] = None):
        self._tool = market_tool or MarketDataTool()

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """获取单个币种的最新 Tick 快照（价格、24h涨跌幅、成交量、资金费率、持仓量等）."""
        # 尝试优先从本地常驻服务内存提取结构化数据
        svc = MarketDataService.get_instance()
        if svc and svc.is_running:
            try:
                state = svc.state.get(symbol, create=False)
                if state and state.latest_tick:
                    tick = state.latest_tick
                    quote = state.latest_quote
                    funding_list = state.recent_funding(1)
                    funding_rate = funding_list[-1].rate if funding_list else 0.0
                    return {
                        "symbol": symbol.upper(),
                        "mark_price": float(tick.price or 0.0),
                        "last_price": float(tick.price or 0.0),
                        "bid": float(quote.bid if quote else tick.price),
                        "ask": float(quote.ask if quote else tick.price),
                        "funding_rate": float(funding_rate),
                        "source": "memory_state",
                    }
            except Exception as exc:
                logger.debug("MarketDataService get_ticker failed: %s", exc)

        # 回退到 Tool 原有解析
        try:
            raw_text = self._tool.get_ticker(symbol)
            return {"symbol": symbol.upper(), "raw_summary": raw_text, "source": "tool_fallback"}
        except Exception as exc:
            logger.debug("get_ticker fallback failed: %s", exc)
            return {
                "symbol": symbol.upper(),
                "mark_price": 60000.0 if symbol.upper() == "BTC" else 3000.0,
                "raw_summary": f"Offline fallback for {symbol} ({exc})",
                "source": "offline_fallback",
            }

    def get_candles(self, symbol: str, timeframe: str = "1h", limit: int = 50) -> List[Dict[str, Any]]:
        """获取 K 线历史数据."""
        svc = MarketDataService.get_instance()
        if svc and svc.is_running:
            try:
                state = svc.state.get(symbol, create=False)
                if state:
                    candles = state.get_ohlcv(timeframe, limit=limit)
                    if candles:
                        return [
                            {
                                "timestamp_ms": c.timestamp_ms,
                                "open": float(c.open),
                                "high": float(c.high),
                                "low": float(c.low),
                                "close": float(c.close),
                                "volume": float(c.volume),
                            }
                            for c in candles
                        ]
            except Exception as exc:
                logger.debug("MarketDataService get_candles failed: %s", exc)

        # 回退到 Tool
        try:
            raw = self._tool.get_candles(symbol, timeframe, limit)
            return [{"raw": raw, "source": "tool_fallback"}]
        except Exception:
            return []

    def get_indicators(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """获取当前技术指标计算结果（RSI, EMA, ATR, ADX, CVD 等）."""
        svc = MarketDataService.get_instance()
        if svc and svc.is_running:
            try:
                state = svc.state.get(symbol, create=False)
                if state:
                    inds = state.indicators(timeframe)
                    if inds:
                        return {
                            "symbol": symbol.upper(),
                            "timeframe": timeframe,
                            "rsi": float(inds.get("rsi14")) if inds.get("rsi14") is not None else None,
                            "atr": float(inds.get("atr14")) if inds.get("atr14") is not None else None,
                            "adx": float(inds.get("adx14")) if inds.get("adx14") is not None else None,
                            "ema_20": float(inds.get("ema20")) if inds.get("ema20") is not None else None,
                            "ema_50": float(inds.get("ema50")) if inds.get("ema50") is not None else None,
                            "poc": inds.get("poc"),
                            "vah": inds.get("vah"),
                            "val": inds.get("val"),
                        }
            except Exception as exc:
                logger.debug("MarketDataService get_indicators failed: %s", exc)

        # 回退
        try:
            raw = self._tool.get_technical_analysis(symbol, [timeframe])
            return {"symbol": symbol.upper(), "raw_analysis": raw}
        except Exception:
            return {"symbol": symbol.upper(), "raw_analysis": ""}

    def get_orderbook(self, symbol: str, depth: int = 5) -> Dict[str, Any]:
        """获取 L2 深度盘口."""
        svc = MarketDataService.get_instance()
        if svc and svc.is_running:
            try:
                state = svc.state.get(symbol, create=False)
                if state and state.latest_book:
                    book = state.latest_book
                    bids = [{"price": float(b.price), "size": float(b.size)} for b in book.bids[:depth]]
                    asks = [{"price": float(a.price), "size": float(a.size)} for a in book.asks[:depth]]
                    return {
                        "symbol": symbol.upper(),
                        "bids": bids,
                        "asks": asks,
                        "spread": asks[0]["price"] - bids[0]["price"] if bids and asks else 0.0,
                    }
            except Exception as exc:
                logger.debug("MarketDataService get_orderbook failed: %s", exc)

        try:
            raw = self._tool.get_orderbook(symbol, depth)
            return {"symbol": symbol.upper(), "raw_book": raw}
        except Exception:
            return {"symbol": symbol.upper(), "bids": [], "asks": [], "spread": 0.0}

    def get_market_regime(self, symbol: str = "BTC", timeframe: str = "1h") -> Dict[str, Any]:
        """获取指定币种或大盘的市场状态量化分类 (趋势/震荡/恐慌/观望)."""
        from xmlx_vlm.ai_trader.market_service.regime import MarketRegimeDetector
        svc = MarketDataService.get_instance()
        detector = MarketRegimeDetector()
        if svc and svc.is_running:
            try:
                state = svc.state.get(symbol, create=False)
                if state:
                    candles = state.get_ohlcv(timeframe, limit=50)
                    if candles and len(candles) >= 28:
                        analysis = detector.detect_regime(symbol, candles, timeframe)
                        return {
                            "symbol": analysis.symbol,
                            "regime": analysis.regime.value,
                            "confidence": analysis.confidence,
                            "suggested_strategy": analysis.suggested_strategy,
                            "summary": analysis.summary,
                            "metrics": analysis.metrics,
                        }
            except Exception as exc:
                logger.debug("MarketDataService regime detection failed: %s", exc)

        # 回退或默认分析
        return {
            "symbol": symbol.upper(),
            "regime": "range_bound",
            "confidence": 0.7,
            "suggested_strategy": "grid_mean_reversion",
            "summary": "默认震荡市，建议控制杠杆与仓位",
            "metrics": {},
        }

    def scan_markets(
        self,
        conditions: Optional[Callable[[Dict[str, Any]], bool]] = None,
        symbols: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """批量快速扫描多个币种的实时状态，并用条件函数过滤.
        
        示例:
        ```python
        # 筛选 RSI < 35 且 24h 成交额大于 10M 的超跌标的
        oversold = sdk.market.scan_markets(
            lambda x: (x.get("rsi") or 50) < 35 and (x.get("volume_24h") or 0) > 10_000_000
        )
        ```
        """
        svc = MarketDataService.get_instance()
        results = []
        target_symbols = symbols
        if not target_symbols:
            if svc and svc.is_running:
                target_symbols = list(svc.state.symbols.keys())
            else:
                target_symbols = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "AVAX", "LINK"]

        for s in target_symbols:
            ticker = self.get_ticker(s)
            inds = self.get_indicators(s, "1h")
            merged = {**ticker, **inds, "symbol": s.upper()}
            if conditions is None or conditions(merged):
                results.append(merged)
        return results


class OMSSDK:
    """订单与账户管理 In-Memory 接口."""

    def __init__(self, trading_tool: Optional[TradingTool] = None, oms: Optional[OMSEngine] = None):
        self._tool = trading_tool or TradingTool(oms=oms)

    @property
    def oms(self) -> OMSEngine:
        return self._tool.oms

    def get_positions(self) -> List[Dict[str, Any]]:
        """获取当前活跃持仓列表."""
        summary = self.oms.portfolio_summary()
        return summary.get("positions", [])

    def get_account_summary(self) -> Dict[str, Any]:
        """获取账户总权益、可用保证金与盈亏概况."""
        return self.oms.portfolio_summary()

    def simulate_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: Optional[float] = None,
        order_type: str = "market",
    ) -> Dict[str, Any]:
        """模拟下单试算（通过 OMS 规则并返回风控与滑点试算结果）."""
        res = self._tool.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            mode="paper",
            order_type=order_type,
            price=price,
        )
        return {"result": res, "status": "simulated"}

    def propose_trade(
        self,
        symbol: str,
        direction: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        confidence: int = 80,
        reason: str = "",
    ) -> Dict[str, Any]:
        """构建标准化交易提案."""
        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        rr = round(reward / risk, 2) if risk > 0 else 0.0

        return {
            "symbol": symbol.upper(),
            "direction": direction.lower(),
            "entry_price": float(entry),
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "risk_reward_ratio": rr,
            "confidence": int(confidence),
            "reason": reason,
        }


class TraderSDK:
    """AI Trader 聚合 In-Memory SDK 入口."""

    def __init__(
        self,
        market_tool: Optional[MarketDataTool] = None,
        trading_tool: Optional[TradingTool] = None,
        oms: Optional[OMSEngine] = None,
    ):
        self.market = MarketSDK(market_tool=market_tool)
        self.oms = OMSSDK(trading_tool=trading_tool, oms=oms)

    def scan_and_rank(self, metric: str = "rsi", reverse: bool = False, top_k: int = 5) -> List[Dict[str, Any]]:
        """快速排序辅助方法."""
        all_data = self.market.scan_markets()
        valid = [x for x in all_data if x.get(metric) is not None]
        valid.sort(key=lambda x: x.get(metric, 0), reverse=reverse)
        return valid[:top_k]
