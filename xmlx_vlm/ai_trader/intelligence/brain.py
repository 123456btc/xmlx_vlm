"""主动情报大脑：新闻扫描、信号去重、事件分发."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

import requests

from xmlx_vlm.ai_trader.intelligence.signal import Signal, SignalSeverity

logger = logging.getLogger(__name__)

SignalHandler = Callable[[Signal], Any]


@dataclass
class BrainConfig:
    """Brain 配置."""

    news_scan_interval_seconds: int = 600
    signal_debounce_seconds: int = 600
    news_history_size: int = 1024
    enabled: bool = True


class Brain:
    """主动情报层.

    负责：
    - 外部新闻/信号源扫描
    - 信号去重
    - 通过回调或事件总线分发给策略
    """

    def __init__(self, config: Optional[BrainConfig] = None):
        self.config = config or BrainConfig()
        self._handlers: List[SignalHandler] = []
        self._recent_signals: Dict[str, float] = {}
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._news_history: Deque[str] = deque(maxlen=self.config.news_history_size)

    def register_handler(self, handler: SignalHandler) -> None:
        self._handlers.append(handler)

    def unregister_handler(self, handler: SignalHandler) -> None:
        if handler in self._handlers:
            self._handlers.remove(handler)

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        if not self.config.enabled:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run())
        logger.info("Brain started")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        logger.info("Brain stopped")

    def handle_signal(self, signal: Signal) -> None:
        """处理一个信号：去重后分发."""
        if not self.config.enabled:
            return

        now = time.time()
        key = signal.debounce_key
        last = self._recent_signals.get(key, 0)
        if now - last < self.config.signal_debounce_seconds:
            return
        self._recent_signals[key] = now

        for handler in self._handlers:
            try:
                handler(signal)
            except Exception:
                logger.exception("Signal handler failed for %s", signal.debounce_key)

    def scan_news_once(self) -> List[Signal]:
        """单次扫描 CryptoCompare 新闻，返回信号列表."""
        signals: List[Signal] = []
        try:
            resp = requests.get(
                "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&sortOrder=latest",
                timeout=15,
            )
            if resp.status_code != 200:
                return signals
            data = resp.json()
            for item in data.get("Data", []):
                url = item.get("URL", "")
                if url in self._news_history:
                    continue
                self._news_history.append(url)

                published = item.get("PublishedOn", 0)
                if time.time() - published > 600:
                    continue

                sentiment = self._classify_sentiment(item.get("Title", "") + " " + item.get("Body", ""))
                if sentiment == "neutral":
                    continue

                severity = SignalSeverity.WARNING if sentiment == "bearish" else SignalSeverity.INFO
                signals.append(
                    Signal(
                        type="news",
                        symbol="CRYPTO",
                        severity=severity,
                        title=item.get("Title", ""),
                        detail=f"{item.get('Body', '')[:200]}...\nSource: {item.get('Source', '')}\nURL: {url}",
                        source="cryptocompare",
                        metadata={"sentiment": sentiment, "url": url},
                    )
                )
        except Exception as exc:
            logger.debug("News scan failed: %s", exc)
        return signals

    def _classify_sentiment(self, text: str) -> str:
        text = text.lower()
        bullish = ["surge", "rally", "bullish", "breakout", "ath", "pump", "adoption"]
        bearish = ["crash", "dump", "bearish", "sell-off", "plunge", "hack", "ban", "fraud"]
        bc = sum(1 for w in bullish if w in text)
        brc = sum(1 for w in bearish if w in text)
        if bc > brc:
            return "bullish"
        if brc > bc:
            return "bearish"
        return "neutral"

    async def _run(self) -> None:
        interval = max(10, self.config.news_scan_interval_seconds)
        while not self._stop_event.is_set():
            try:
                signals = self.scan_news_once()
                for signal in signals:
                    self.handle_signal(signal)
            except Exception:
                logger.exception("Brain scan cycle failed")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
