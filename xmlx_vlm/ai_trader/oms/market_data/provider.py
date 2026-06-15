"""市场数据 Provider 抽象."""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, Optional

from xmlx_vlm.ai_trader.oms.market_data.models import OrderBook, Quote, VolumeProfile


class MarketDataProvider(ABC):
    """市场数据提供方抽象.

    由具体交易所/行情源实现，供 SmartOrderRouter、ExecutionAlgorithm、
    PaperExecutionAdapter 共享使用。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """获取最新报价."""
        ...

    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 10) -> Optional[OrderBook]:
        """获取订单簿."""
        ...

    @abstractmethod
    async def get_recent_volume(
        self, symbol: str, window_seconds: int = 300
    ) -> Optional[Decimal]:
        """获取最近窗口成交量."""
        ...

    @abstractmethod
    async def get_volume_profile(
        self,
        symbol: str,
        duration_seconds: int = 86400,
        buckets: int = 24,
    ) -> Optional[VolumeProfile]:
        """获取成交量分布，默认 24 小时按小时分桶."""
        ...

    async def get_volatility(
        self,
        symbol: str,
        window_days: int = 30,
    ) -> Optional[Decimal]:
        """获取日波动率（可选实现）."""
        return None


class StaticMarketDataProvider(MarketDataProvider):
    """静态 market data provider，用于测试."""

    def __init__(
        self,
        quotes: Optional[Dict[str, Quote]] = None,
        books: Optional[Dict[str, OrderBook]] = None,
        volumes: Optional[Dict[str, Decimal]] = None,
        profiles: Optional[Dict[str, VolumeProfile]] = None,
    ):
        self._quotes = quotes or {}
        self._books = books or {}
        self._volumes = volumes or {}
        self._profiles = profiles or {}

    @property
    def name(self) -> str:
        return "static"

    async def get_quote(self, symbol: str) -> Optional[Quote]:
        return self._quotes.get(symbol.upper())

    async def get_order_book(self, symbol: str, depth: int = 10) -> Optional[OrderBook]:
        return self._books.get(symbol.upper())

    async def get_recent_volume(
        self, symbol: str, window_seconds: int = 300
    ) -> Optional[Decimal]:
        return self._volumes.get(symbol.upper())

    async def get_volume_profile(
        self,
        symbol: str,
        duration_seconds: int = 86400,
        buckets: int = 24,
    ) -> Optional[VolumeProfile]:
        return self._profiles.get(symbol.upper())
