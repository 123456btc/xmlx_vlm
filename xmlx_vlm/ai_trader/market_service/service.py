"""行情服务 orchestrator.

把 WebSocket 客户端、内存状态机、事件总线拼在一起，
对外提供同步/异步查询接口，供 tools/market.py 与 AI Agent 使用。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Dict, List, Optional

from .alerts import AlertConfig, AlertEngine
from .events import (
    BarClosedEvent,
    BookUpdateEvent,
    EventBus,
    FundingUpdateEvent,
    MarketEvent,
    OIUpdateEvent,
    PriceUpdateEvent,
    TradeEvent,
)
from .market_info import fetch_top_volume_coins
from .models import Bar, FundingRate, MarketSummary, OISnapshot, Quote, Tick
from .state import MarketState
from .ws_client import HyperliquidMessageParser, HyperliquidWSClient

logger = logging.getLogger(__name__)


class MarketDataService:
    """常驻行情服务.

    使用示例：
        service = MarketDataService()
        service.start()
        service.subscribe("BTC")
        service.subscribe("ETH")
        # 稍后...
        summary = service.get_summary("BTC")
    """

    _instance: Optional[MarketDataService] = None

    @classmethod
    def get_instance(cls) -> Optional[MarketDataService]:
        """获取当前活跃的 MarketDataService 单例实例."""
        return cls._instance

    def __init__(
        self,
        url: str = "wss://api.hyperliquid.xyz/ws",
        event_bus: Optional[EventBus] = None,
        top_n: int = 30,
        refresh_interval_sec: int = 60,
        alert_config: Optional[AlertConfig] = None,
        watched_coins: Optional[List[str]] = None,
    ) -> None:
        MarketDataService._instance = self
        self.event_bus = event_bus or EventBus()
        self.state = MarketState()
        self._lock = threading.RLock()
        self._subscribed_coins: set[str] = set()
        self._top_volumes: dict[str, float] = {}
        self._prev_day_prices: dict[str, float] = {}
        self._funding_rates: dict[str, float] = {}
        self._open_interests: dict[str, float] = {}
        self._top_coins: set[str] = set()

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._client: Optional[HyperliquidWSClient] = None
        self._started = False
        self._url = url
        self._top_n = max(1, top_n)
        self._refresh_interval_sec = max(60, refresh_interval_sec)
        self._refresh_task: Optional[asyncio.Task] = None

        self._alert_engine = AlertEngine(self.state, self.event_bus, alert_config)
        self._parser = HyperliquidMessageParser()
        # Throttle allMids event publishing: only publish when price moves > threshold.
        # Eliminates the constant flood of PriceUpdateEvents for unchanged prices.
        self._last_published_price: dict[str, float] = {}
        self._price_publish_threshold = 0.0001  # 0.01% minimum move to trigger publish
        self._watched_coins = [c.upper() for c in watched_coins] if watched_coins else None

    @property
    def is_running(self) -> bool:
        return self._started

    # ── 生命周期 ──
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()
        # 等待 loop 就绪
        while self._loop is None:
            time.sleep(0.01)
        logger.info("MarketDataService started")

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        if self._loop is not None and self._client is not None:
            future = asyncio.run_coroutine_threadsafe(self._client.stop(), self._loop)
            try:
                future.result(timeout=5)
            except Exception:
                logger.exception("Failed to stop websocket client gracefully")
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("MarketDataService stopped")

    def _run_event_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._client = HyperliquidWSClient(
            url=self._url,
            on_message=self._on_message,
            on_connection_event=self.event_bus.publish,
        )
        self._client.start()
        
        if self._watched_coins is not None:
            # Subscribe to custom watched coins directly
            for coin in self._watched_coins:
                self.subscribe(coin)
        else:
            # Subscribe to allMids exactly once
            self._client.subscribe("allMids")
            # 启动时拉取成交额前 N 名并订阅
            self._loop.create_task(self._refresh_top_coins())
            # 定时刷新排名
            self._refresh_task = self._loop.create_task(self._refresh_loop())

        # 恢复已有订阅
        with self._lock:
            coins = list(self._subscribed_coins)
        for coin in coins:
            self._client.subscribe("l2Book", coin=coin, nLevels=20)
            self._client.subscribe("trades", coin=coin)
        try:
            self._loop.run_forever()
        finally:
            if self._refresh_task is not None:
                self._refresh_task.cancel()
            self._loop.close()
            self._loop = None

    # ── 订阅管理 ──
    def subscribe(self, coin: str) -> None:
        """订阅某个币对的行情."""
        coin = coin.upper()
        with self._lock:
            if coin in self._subscribed_coins:
                return
            self._subscribed_coins.add(coin)
        if self._client is not None:
            self._client.subscribe("l2Book", coin=coin, nLevels=20)
            self._client.subscribe("trades", coin=coin)
            self._client.subscribe("candle", coin=coin, interval="1m")
        logger.info("Subscribed to %s", coin)
        # Bootstrap K-line history asynchronously
        self._bootstrap_candles(coin)

    def unsubscribe(self, coin: str) -> None:
        coin = coin.upper()
        with self._lock:
            if coin not in self._subscribed_coins:
                return
            self._subscribed_coins.discard(coin)
        if self._client is not None:
            with self._lock:
                in_top = coin in getattr(self, "_top_coins", set())
            if not in_top:
                self._client.unsubscribe("candle", coin=coin, interval="1m")
                self._client.unsubscribe("l2Book", coin=coin, nLevels=20)
                self._client.unsubscribe("trades", coin=coin)
            else:
                # Still in watchlist, keep candle and l2Book, unsubscribe trades
                self._client.unsubscribe("trades", coin=coin)

    def _bootstrap_candles(self, coin: str) -> None:
        if self._loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._async_bootstrap_candles(coin), self._loop)
            except Exception as exc:
                logger.warning("Failed to schedule K-line bootstrap for %s: %s", coin, exc)

    async def _async_bootstrap_candles(self, coin: str) -> None:
        try:
            import urllib.request
            import json
            from xmlx_vlm.ai_trader.market_service.state import _TIME_FRAME_MS, _floor_ms
            from xmlx_vlm.ai_trader.market_service.models import Bar

            info_url = self._url.replace("wss://", "https://").replace("/ws", "/info")
            sym_state = self.state.get(coin, create=True)
            
            # Bootstrap 1m, 5m, 15m, and 1h intervals
            intervals = ["1m", "5m", "15m", "1h"]
            for interval in intervals:
                interval_ms = _TIME_FRAME_MS.get(interval)
                if not interval_ms:
                    continue
                
                # Check SQLite latest timestamp
                db_latest_ts = sym_state._kline_db.get_latest_timestamp(coin, interval)
                now_ms = int(time.time() * 1000)
                
                cached_bars = []
                need_rest_fetch = True
                start_ms = 0
                end_ms = now_ms
                
                if db_latest_ts is not None:
                    # Check if gap is smaller than one candle duration
                    if now_ms - db_latest_ts < interval_ms:
                        # Load from DB and skip REST
                        cached_bars = sym_state._kline_db.load_bars(coin, interval, limit=150)
                        need_rest_fetch = False
                        logger.debug("K-lines for %s (%s) fully cached in SQLite", coin, interval)
                    else:
                        # Fetch delta gap
                        cached_bars = sym_state._kline_db.load_bars(coin, interval, limit=150)
                        # Start from latest cached candle to get the updates
                        start_ms = db_latest_ts
                        # Make sure start_ms is not too old. If it is older than 150 candles, fetch full
                        if now_ms - start_ms > 150 * interval_ms:
                            start_ms = now_ms - 150 * interval_ms
                            cached_bars = []
                else:
                    # Fetch full 150 candles
                    start_ms = now_ms - 150 * interval_ms
                
                bars = []
                if need_rest_fetch:
                    payload = {
                        "type": "candleSnapshot",
                        "req": {
                            "coin": coin,
                            "interval": interval,
                            "startTime": start_ms,
                            "endTime": end_ms
                        }
                    }
                    
                    def make_request():
                        import time as t_mod
                        for attempt in range(3):
                            try:
                                req = urllib.request.Request(
                                    info_url,
                                    data=json.dumps(payload).encode("utf-8"),
                                    headers={
                                        "Content-Type": "application/json",
                                        "Connection": "close"
                                    }
                                )
                                with urllib.request.urlopen(req, timeout=10) as response:
                                    return json.loads(response.read().decode("utf-8"))
                            except Exception as e:
                                if attempt == 2:
                                    raise e
                                t_mod.sleep(0.25 * (attempt + 1))
                            
                    raw = await self._loop.run_in_executor(None, make_request)
                    if isinstance(raw, list):
                        new_bars = []
                        for row in raw:
                            if not isinstance(row, dict):
                                continue
                            new_bars.append(Bar(
                                symbol=coin,
                                timeframe=interval,
                                open=float(row["o"]),
                                high=float(row["h"]),
                                low=float(row["l"]),
                                close=float(row["c"]),
                                volume=float(row["v"]),
                                timestamp_ms=int(row["t"]),
                                buy_volume=0.0,
                                sell_volume=0.0,
                            ))
                        
                        if new_bars:
                            # Save new bars to database
                            sym_state._kline_db.save_bars(new_bars)
                            
                            # Merge cached bars and new bars, removing duplicates by timestamp
                            seen_ts = {b.timestamp_ms for b in new_bars}
                            combined = [b for b in cached_bars if b.timestamp_ms not in seen_ts] + new_bars
                            combined.sort(key=lambda x: x.timestamp_ms)
                            bars = combined[-150:]
                        else:
                            bars = cached_bars
                    else:
                        bars = cached_bars
                else:
                    bars = cached_bars
                
                # Store in state
                with sym_state._lock:
                    if bars:
                        current_bar_ts = _floor_ms(now_ms, interval_ms)
                        latest_bar = bars[-1]
                        if latest_bar.timestamp_ms == current_bar_ts:
                            sym_state._current_bar[interval] = latest_bar
                            sym_state._bars[interval] = bars[:-1]
                        else:
                            sym_state._current_bar[interval] = None
                            sym_state._bars[interval] = bars
                    else:
                        sym_state._bars[interval] = []
                        sym_state._current_bar[interval] = None
                
                logger.info("Successfully bootstrapped %d K-lines for %s (%s)", len(bars), coin, interval)
                await asyncio.sleep(0.05)
        except Exception as exc:
            logger.warning("Failed to bootstrap K-lines for %s: %s", coin, exc)

    async def _refresh_loop(self) -> None:
        while self._started:
            try:
                await asyncio.sleep(self._refresh_interval_sec)
                await self._refresh_top_coins()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Top-coins refresh loop failed")

    async def _refresh_top_coins(self) -> None:
        """按 24h 成交额订阅前 N 名；掉出排名的币种取消订阅."""
        try:
            from .market_info import fetch_meta_and_ctxs, _to_float
            # Get info endpoint URL based on self._url
            info_url = self._url.replace("wss://", "https://").replace("/ws", "/info")
            
            meta, ctxs = await self._loop.run_in_executor(
                None, fetch_meta_and_ctxs, info_url
            )
            
            volumes: List[tuple[str, float, float, float, float]] = []
            universe = meta.get("universe", [])
            for asset, ctx in zip(universe, ctxs):
                coin = asset.get("name")
                if not coin:
                    continue
                day_vlm = _to_float(ctx.get("dayNtlVlm"))
                prev_day_px = _to_float(ctx.get("prevDayPx"))
                funding = _to_float(ctx.get("funding"))
                oi = _to_float(ctx.get("openInterest"))
                volumes.append((coin, day_vlm, prev_day_px, funding, oi))
            
            # Sort and get top N
            volumes.sort(key=lambda x: x[1], reverse=True)
            top_coins = [c[0] for c in volumes[:self._top_n]]
            
            with self._lock:
                self._top_volumes = {c[0]: c[1] for c in volumes}
                self._prev_day_prices = {c[0]: c[2] for c in volumes}
                self._funding_rates = {c[0]: c[3] for c in volumes}
                self._open_interests = {c[0]: c[4] for c in volumes}
                self._top_coins = set(top_coins)
                
                # Active coins + top watchlist coins need candle & l2Book subscription
                needed_candle_coins = self._subscribed_coins.union(self._top_coins)
                needed_active_coins = self._subscribed_coins
            
            if self._client is not None:
                # Subscribe all needed candle and l2Book coins
                for coin in needed_candle_coins:
                    self._client.subscribe("candle", coin=coin, interval="1m")
                    self._client.subscribe("l2Book", coin=coin, nLevels=20)
                    # Bootstrap if necessary
                    sym_state = self.state.get(coin, create=True)
                    if not sym_state._bars.get("1m") and not sym_state._current_bar.get("1m"):
                        self._bootstrap_candles(coin)
                
                # Subscribe trades for active coins, and unsubscribe trades for non-active watchlist coins
                for coin in needed_candle_coins:
                    if coin in needed_active_coins:
                        self._client.subscribe("trades", coin=coin)
                    else:
                        self._client.unsubscribe("trades", coin=coin)
                
                # Unsubscribe dropped coins
                for coin in list(self.state.symbols()):
                    if coin not in needed_candle_coins:
                        self._client.unsubscribe("candle", coin=coin, interval="1m")
                        self._client.unsubscribe("l2Book", coin=coin, nLevels=20)
                        self._client.unsubscribe("trades", coin=coin)
                        
        except Exception as exc:
            logger.warning("Could not refresh top volume coins and context: %s", exc)
            # Fallback to the original fetch_top_volume_coins to maintain robustness
            try:
                top_coins = await self._loop.run_in_executor(
                    None, fetch_top_volume_coins, self._top_n
                )
                with self._lock:
                    self._top_coins = set(top_coins)
                    needed_candle_coins = self._subscribed_coins.union(self._top_coins)
                    needed_active_coins = self._subscribed_coins
                if self._client is not None:
                    for coin in needed_candle_coins:
                        self._client.subscribe("candle", coin=coin, interval="1m")
                        self._client.subscribe("l2Book", coin=coin, nLevels=20)
                        sym_state = self.state.get(coin, create=True)
                        if not sym_state._bars.get("1m") and not sym_state._current_bar.get("1m"):
                            self._bootstrap_candles(coin)
                    for coin in needed_candle_coins:
                        if coin in needed_active_coins:
                            self._client.subscribe("trades", coin=coin)
                        else:
                            self._client.unsubscribe("trades", coin=coin)
                    for coin in list(self.state.symbols()):
                        if coin not in needed_candle_coins:
                            self._client.unsubscribe("candle", coin=coin, interval="1m")
                            self._client.unsubscribe("l2Book", coin=coin, nLevels=20)
                            self._client.unsubscribe("trades", coin=coin)
            except Exception as e2:
                logger.error("All refresh top coins attempts failed: %s", e2)

    def get_watched_coins(self) -> List[str]:
        if self._watched_coins is not None:
            return sorted(self._watched_coins)
        with self._lock:
            if hasattr(self, "_top_coins") and self._top_coins:
                return sorted(self._top_coins)
            return sorted(self._subscribed_coins)

    # ── 消息处理 ──
    def _on_message(self, msg: dict) -> None:
        channel = msg.get("channel")
        if channel == "allMids":
            self._handle_all_mids(msg)
        elif channel == "l2Book":
            self._handle_l2_book(msg)
        elif channel == "trades":
            self._handle_trades(msg)
        elif channel == "funding":
            self._handle_funding(msg)
        elif channel == "candle":
            self._handle_candle(msg)
        elif channel == "error":
            logger.error("Hyperliquid WS error: %s", msg)

    def _handle_all_mids(self, msg: dict) -> None:
        mids = self._parser.parse_all_mids(msg)
        now = int(time.time() * 1000)
        for coin, price in mids.items():
            # Only process coins we care about
            with self._lock:
                is_watched = (
                    coin in self._subscribed_coins or
                    coin in getattr(self, "_top_coins", set()) or
                    coin in self.state.symbols()
                )
                if not is_watched:
                    continue
            sym_state = self.state.get(coin, create=True)
            sym_state.update_tick(Tick(symbol=coin, price=price, timestamp_ms=now))
            # Throttle: only publish PriceUpdateEvent when price changed meaningfully.
            # Without this, every allMids push (1/s) fires events for 100+ coins.
            last = self._last_published_price.get(coin)
            if last is None or last == 0.0 or abs(price - last) / last > self._price_publish_threshold:
                self._last_published_price[coin] = price
                self.event_bus.publish(
                    PriceUpdateEvent(symbol=coin, timestamp_ms=now, price=price, source="mid")
                )

    def _handle_l2_book(self, msg: dict) -> None:
        book = self._parser.parse_l2_book(msg)
        if book is None:
            return
        sym_state = self.state.get(book.symbol, create=True)
        sym_state.update_book(book)
        self.event_bus.publish(BookUpdateEvent(symbol=book.symbol, timestamp_ms=book.timestamp_ms, book=book))

    def _handle_trades(self, msg: dict) -> None:
        trades = self._parser.parse_trades(msg)
        if not trades:
            return
        for trade in trades:
            sym_state = self.state.get(trade.symbol, create=True)
            sym_state.add_trade(trade)
            self.event_bus.publish(
                TradeEvent(symbol=trade.symbol, timestamp_ms=trade.timestamp_ms, trade=trade)
            )

    def _handle_funding(self, msg: dict) -> None:
        funding = self._parser.parse_funding(msg)
        if funding is None:
            return
        sym_state = self.state.get(funding.symbol, create=True)
        sym_state.add_funding(funding)
        self.event_bus.publish(
            FundingUpdateEvent(symbol=funding.symbol, timestamp_ms=funding.timestamp_ms, funding=funding)
        )

    def _handle_candle(self, msg: dict) -> None:
        bar = self._parser.parse_candle(msg)
        if bar is None:
            return
        sym_state = self.state.get(bar.symbol, create=True)
        sym_state.update_candle(bar)
        self.event_bus.publish(
            PriceUpdateEvent(symbol=bar.symbol, timestamp_ms=bar.timestamp_ms, price=bar.close, source="candle")
        )

    # ── 对外同步查询接口 ──
    def get_quote(self, symbol: str) -> Optional[Quote]:
        return self.state.get(symbol).get_quote()

    def get_summary(self, symbol: str, light: bool = False) -> Optional[MarketSummary]:
        """基于内存状态构造综合市场摘要."""
        coin = symbol.upper().replace("/USDC", "").replace("/USD", "")
        if not light:
            self.subscribe(coin)
        state = self.state.get(coin)

        quote = state.get_quote()
        book = state.get_book()
        
        mark_px = 0.0
        bid = None
        ask = None
        spread = 0.0
        
        if quote is not None:
            mark_px = quote.ask if quote.ask else quote.bid
            bid = quote.bid
            ask = quote.ask
            spread = ask - bid if ask and bid else 0.0
        else:
            if state.latest_tick is not None:
                mark_px = state.latest_tick.price
                bid = mark_px
                ask = mark_px
            else:
                return None

        # Handle light query mode
        if light:
            with self._lock:
                cached_vol = self._top_volumes.get(coin, 0.0)
                cached_prev_px = self._prev_day_prices.get(coin, 0.0)
                funding_rate = self._funding_rates.get(coin)
            
            if cached_prev_px > 0:
                change_24h_pct = (mark_px / cached_prev_px - 1.0) * 100.0
            else:
                change_24h_pct = 0.0
                
            return MarketSummary(
                symbol=symbol,
                mark_price=mark_px,
                oracle_price=None,
                basis_pct=0.0,
                bid=bid,
                ask=ask,
                spread=spread,
                high_24h=None,
                low_24h=None,
                change_24h_pct=change_24h_pct,
                volume_24h=cached_vol,
                atr14=None,
                atr_pct=None,
                adx14=None,
                rsi14=None,
                ema20=None,
                ema50=None,
                volume_profile={"poc": None, "vah": None, "val": None},
                open_interest=None,
                oi_change_1h_pct=None,
                oi_change_24h_pct=None,
                funding_rate=funding_rate,
                avg_funding_24h=None,
                funding_trend="stable",
                depth_imbalance=0.0,
                bid_qty_top=0.0,
                ask_qty_top=0.0,
                cvd_15m=None,
                cvd_1h=None,
                cvd_4h=None,
            )

        ohlcv_1h = state.get_ohlcv("1h", limit=100)
        ind_1h = state.indicators("1h") if len(ohlcv_1h) >= 26 else {}

        ohlcv_1d = state.get_ohlcv("1h", limit=24)
        high_24h = max((c.high for c in ohlcv_1d), default=None)
        low_24h = min((c.low for c in ohlcv_1d), default=None)

        # 24h 成交量用最近 24h K 线汇总（Hyperliquid 不通过 WS 直接给 dayNtlVlm）
        # 优先使用自 REST API 缓存的成交额，以保证刚启动时数据准确
        with self._lock:
            cached_vol = self._top_volumes.get(coin)
            cached_prev_px = self._prev_day_prices.get(coin)

        if cached_vol is not None and cached_vol > 0:
            volume_24h = cached_vol
        else:
            volume_24h = sum(c.volume for c in ohlcv_1d)

        # 涨跌幅：优先使用最新的 mark_price 对比缓存的 24h 前 prevDayPx 进行计算
        if cached_prev_px is not None and cached_prev_px > 0:
            change_24h_pct = (mark_px / cached_prev_px - 1.0) * 100.0
        else:
            change_24h_pct = 0.0
            if len(ohlcv_1d) >= 2:
                change_24h_pct = (ohlcv_1d[-1].close / ohlcv_1d[0].close - 1.0) * 100.0

        # 深度失衡
        depth_imbalance = 0.0
        bid_qty_top = 0.0
        ask_qty_top = 0.0
        if book is not None:
            bids = book.bids[:20]
            asks = book.asks[:20]
            bid_qty_top = sum(level.size for level in bids)
            ask_qty_top = sum(level.size for level in asks)
            total = bid_qty_top + ask_qty_top
            if total:
                depth_imbalance = (bid_qty_top - ask_qty_top) / total

        # OI & funding
        oi_rows = state.recent_oi()
        latest_oi = oi_rows[-1] if oi_rows else None
        funding_rows = state.recent_funding()
        rates = [f.rate for f in funding_rows]
        avg_funding = sum(rates) / len(rates) if rates else None
        funding_trend = "stable"
        if len(rates) >= 2:
            if rates[-1] > rates[0] * 1.2:
                funding_trend = "rising"
            elif rates[-1] < rates[0] * 0.8:
                funding_trend = "falling"

        # Read actual funding rate from WS or fallback to cache
        latest_funding_rate = rates[-1] if rates else None
        if latest_funding_rate is None:
            with self._lock:
                latest_funding_rate = self._funding_rates.get(coin)

        return MarketSummary(
            symbol=symbol,
            mark_price=mark_px,
            oracle_price=None,
            basis_pct=0.0,
            bid=bid,
            ask=ask,
            spread=spread,
            high_24h=high_24h,
            low_24h=low_24h,
            change_24h_pct=change_24h_pct,
            volume_24h=volume_24h,
            atr14=ind_1h.get("atr14"),
            atr_pct=ind_1h.get("atr_pct"),
            adx14=ind_1h.get("adx14"),
            rsi14=ind_1h.get("rsi14"),
            ema20=ind_1h.get("ema20"),
            ema50=ind_1h.get("ema50"),
            volume_profile={
                "poc": ind_1h.get("poc"),
                "vah": ind_1h.get("vah"),
                "val": ind_1h.get("val"),
            },
            open_interest=latest_oi.open_interest if latest_oi else None,
            oi_change_1h_pct=state.oi_delta_pct(60),
            oi_change_24h_pct=state.oi_delta_pct(24 * 60),
            funding_rate=latest_funding_rate,
            avg_funding_24h=avg_funding,
            funding_trend=funding_trend,
            depth_imbalance=depth_imbalance,
            bid_qty_top=bid_qty_top,
            ask_qty_top=ask_qty_top,
            cvd_15m=state.cvd_window(15),
            cvd_1h=state.cvd_window(60),
            cvd_4h=state.cvd_window(4 * 60),
        )

    def get_multi_timeframe_summary(self, symbol: str) -> Dict[str, object]:
        """返回 5m/15m/1h 三周期摘要."""
        coin = symbol.upper().replace("/USDC", "").replace("/USD", "")
        self.subscribe(coin)
        state = self.state.get(coin)
        result: Dict[str, object] = {}
        for tf in ("5m", "15m", "1h"):
            ohlcv = state.get_ohlcv(tf, limit=100)
            if len(ohlcv) < 26:
                continue
            ind = state.indicators(tf)
            result[tf] = {
                "close": ohlcv[-1].close,
                "change_pct": (ohlcv[-1].close / ohlcv[-2].close - 1.0) * 100.0 if len(ohlcv) >= 2 else 0.0,
                **ind,
            }
        return result

    def get_historical_columnar(
        self,
        symbol: str,
        timeframe: str = "1m",
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        limit: Optional[int] = 100,
        as_of_ms: Optional[int] = None,
    ) -> Dict[str, List[Any]]:
        """
        Query columnar market series with Point-in-Time (`as_of_ms`) isolation.
        """
        from .columnar_store import ColumnarMarketStore
        coin = symbol.upper().replace("/USDC", "").replace("/USD", "")
        return ColumnarMarketStore.get_instance().query_columnar(
            symbol=coin,
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=limit,
            as_of_ms=as_of_ms,
        )

    def get_snapshot_as_of(self, symbol: str, as_of_ms: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve historical point-in-time state snapshot without lookahead bias.
        """
        from .columnar_store import ColumnarMarketStore
        coin = symbol.upper().replace("/USDC", "").replace("/USD", "")
        return ColumnarMarketStore.get_instance().get_snapshot_as_of(coin, as_of_ms=as_of_ms)

    def get_connection_state(self) -> str:
        return "running" if self._started and self._thread is not None and self._thread.is_alive() else "stopped"
