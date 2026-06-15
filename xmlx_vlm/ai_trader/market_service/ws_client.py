"""Hyperliquid WebSocket 客户端.

职责单一：建立连接、订阅频道、接收消息、断线重连、把原始消息交给解析器。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable, Optional

import websockets

from .events import ConnectionStateEvent
from .models import Bar, BookLevel, BookSnapshot, FundingRate, OISnapshot, Tick, Trade

logger = logging.getLogger(__name__)

_HYPERLIQUID_WS_URL = "wss://api.hyperliquid.xyz/ws"


class HyperliquidWSClient:
    """异步 WebSocket 客户端，支持断线自动重连."""

    def __init__(
        self,
        url: str = _HYPERLIQUID_WS_URL,
        on_message: Optional[Callable[[dict], Awaitable[None] | None]] = None,
        on_connection_event: Optional[Callable[[ConnectionStateEvent], None]] = None,
        reconnect_delay_base: float = 1.0,
        reconnect_delay_max: float = 30.0,
    ) -> None:
        self._url = url
        self._on_message = on_message
        self._on_connection_event = on_connection_event
        self._reconnect_delay_base = reconnect_delay_base
        self._reconnect_delay_max = reconnect_delay_max

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._subscriptions: set[str] = set()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0
        self._reset_attempts_task: Optional[asyncio.Task] = None

    # ── 生命周期 ──
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        try:
            loop = asyncio.get_running_loop()
            self._task = loop.create_task(self._run_loop())
        except RuntimeError:
            loop = asyncio.get_event_loop()
            self._task = loop.create_task(self._run_loop())

    async def stop(self) -> None:
        self._running = False
        if getattr(self, "_reset_attempts_task", None) is not None:
            self._reset_attempts_task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                logger.exception("Error closing websocket")
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def subscribe(self, channel: str, **params: object) -> None:
        """注册订阅；连接建立后会自动发送."""
        key = self._subscription_key(channel, params)
        if key in self._subscriptions:
            return
        self._subscriptions.add(key)
        if self._ws is not None and self._ws.state == websockets.State.OPEN:
            asyncio.create_task(self._send_subscribe(channel, params))

    def unsubscribe(self, channel: str, **params: object) -> None:
        key = self._subscription_key(channel, params)
        self._subscriptions.discard(key)
        if self._ws is not None and self._ws.state == websockets.State.OPEN:
            asyncio.create_task(self._send_unsubscribe(channel, params))

    # ── 内部主循环 ──
    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._connect()
                await self._read_loop()
            except websockets.ConnectionClosed as exc:
                logger.warning("WebSocket closed: %s", exc)
            except Exception:
                logger.exception("WebSocket loop error")
            finally:
                if getattr(self, "_reset_attempts_task", None) is not None:
                    self._reset_attempts_task.cancel()
                    self._reset_attempts_task = None
                self._ws = None
                self._emit(ConnectionStateEvent(
                    state="disconnected",
                    message="connection lost",
                    timestamp_ms=now_ms(),
                ))
            if self._running:
                delay = self._backoff_delay()
                logger.info("Reconnecting in %.1fs (attempt %d)", delay, self._reconnect_attempts)
                await asyncio.sleep(delay)

    async def _connect(self) -> None:
        self._reconnect_attempts += 1
        self._emit(ConnectionStateEvent(
            state="connecting",
            message=f"attempt {self._reconnect_attempts}",
            timestamp_ms=now_ms(),
        ))
        self._ws = await websockets.connect(self._url, ping_interval=20, ping_timeout=10)
        self._emit(ConnectionStateEvent(
            state="connected",
            message="websocket open",
            timestamp_ms=now_ms(),
        ))
        # Stable connection verification: reset attempts after 10s
        self._reset_attempts_task = asyncio.create_task(self._reset_attempts_after_delay())
        # 重连后恢复所有订阅 (staggered to avoid rate limits)
        for key in self._subscriptions:
            channel, params = self._parse_subscription_key(key)
            await self._send_subscribe(channel, params)
            await asyncio.sleep(0.05)

    async def _reset_attempts_after_delay(self) -> None:
        try:
            await asyncio.sleep(10)
            self._reconnect_attempts = 0
            logger.debug("WebSocket connection stable, reset reconnect attempts")
        except asyncio.CancelledError:
            pass

    async def _read_loop(self) -> None:
        assert self._ws is not None
        async for raw in self._ws:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON from websocket: %s", raw[:200])
                continue
            try:
                if self._on_message is not None:
                    result = self._on_message(msg)
                    if asyncio.iscoroutine(result):
                        await result
            except Exception:
                logger.exception("Message handler failed")

    # ── 订阅协议 ──
    async def _send_subscribe(self, channel: str, params: dict) -> None:
        sub: dict = {"type": channel}
        sub.update(params)
        await self._send({"method": "subscribe", "subscription": sub})

    async def _send_unsubscribe(self, channel: str, params: dict) -> None:
        sub: dict = {"type": channel}
        sub.update(params)
        await self._send({"method": "unsubscribe", "subscription": sub})

    async def _send(self, payload: dict) -> None:
        if self._ws is not None and self._ws.state == websockets.State.OPEN:
            try:
                await self._ws.send(json.dumps(payload))
            except Exception:
                logger.exception("Failed to send websocket message")

    # ── 工具 ──
    def _subscription_key(self, channel: str, params: dict) -> str:
        items = sorted(params.items())
        return f"{channel}:{json.dumps(items, separators=(',', ':'))}"

    def _parse_subscription_key(self, key: str) -> tuple[str, dict]:
        channel, params_json = key.split(":", 1)
        params = dict(json.loads(params_json))
        return channel, params

    def _backoff_delay(self) -> float:
        delay = self._reconnect_delay_base * (2 ** min(self._reconnect_attempts - 1, 5))
        return min(delay, self._reconnect_delay_max)

    def _emit(self, event: ConnectionStateEvent) -> None:
        if self._on_connection_event is not None:
            try:
                self._on_connection_event(event)
            except Exception:
                logger.exception("Connection event handler failed")


def now_ms() -> int:
    return int(time.time() * 1000)


class HyperliquidMessageParser:
    """把 Hyperliquid WS 消息解析为领域模型."""

    @staticmethod
    def parse_all_mids(data: dict) -> dict[str, float]:
        """解析 {coin: midPrice} 映射."""
        return {k: _to_float(v) for k, v in data.get("data", {}).items()}

    @staticmethod
    def parse_l2_book(data: dict) -> Optional[BookSnapshot]:
        coin = data.get("data", {}).get("coin")
        levels = data.get("data", {}).get("levels")
        if not coin or not isinstance(levels, list) or len(levels) < 2:
            return None
        bids = [BookLevel(_to_float(r.get("px")), _to_float(r.get("sz"))) for r in levels[0]]
        asks = [BookLevel(_to_float(r.get("px")), _to_float(r.get("sz"))) for r in levels[1]]
        return BookSnapshot(
            symbol=coin,
            bids=bids,
            asks=asks,
            timestamp_ms=now_ms(),
        )

    @staticmethod
    def parse_trades(data: dict) -> List[Trade]:
        raw_data = data.get("data", {})
        if isinstance(raw_data, list):
            # Real Hyperliquid WS format: 'data' is a list of trade dicts
            out = []
            for t in raw_data:
                coin = t.get("coin")
                if not coin:
                    continue
                side = "buy" if str(t.get("side")).lower() == "b" else "sell"
                out.append(Trade(
                    symbol=coin,
                    side=side,
                    price=_to_float(t.get("px")),
                    size=_to_float(t.get("sz")),
                    timestamp_ms=t.get("time", now_ms()),
                ))
            return out
        elif isinstance(raw_data, dict):
            # Mock / fallback format: 'data' is a dict containing 'coin' and 'trades' list
            coin = raw_data.get("coin")
            trades = raw_data.get("trades", [])
            if not coin or not isinstance(trades, list):
                return []
            out = []
            for t in trades:
                side = "buy" if str(t.get("side")).lower() == "b" else "sell"
                out.append(Trade(
                    symbol=coin,
                    side=side,
                    price=_to_float(t.get("px")),
                    size=_to_float(t.get("sz")),
                    timestamp_ms=t.get("time", now_ms()),
                ))
            return out
        return []

    @staticmethod
    def parse_funding(data: dict) -> Optional[FundingRate]:
        coin = data.get("data", {}).get("coin")
        rate = data.get("data", {}).get("fundingRate")
        if coin is None or rate is None:
            return None
        return FundingRate(
            symbol=coin,
            rate=_to_float(rate),
            timestamp_ms=now_ms(),
        )

    @staticmethod
    def parse_candle(data: dict) -> Optional[Bar]:
        d = data.get("data")
        if not d or not isinstance(d, dict):
            return None
        coin = d.get("s")
        interval = d.get("i")
        if not coin or not interval:
            return None
        try:
            return Bar(
                symbol=coin,
                timeframe=interval,
                open=float(d.get("o", 0.0)),
                high=float(d.get("h", 0.0)),
                low=float(d.get("l", 0.0)),
                close=float(d.get("c", 0.0)),
                volume=float(d.get("v", 0.0)),
                timestamp_ms=int(d.get("t", 0)),
                buy_volume=0.0,
                sell_volume=0.0,
            )
        except (ValueError, TypeError):
            return None


def _to_float(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return 0.0
