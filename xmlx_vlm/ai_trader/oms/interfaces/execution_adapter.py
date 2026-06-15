"""执行适配器抽象接口."""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from xmlx_vlm.ai_trader.oms.market_data.models import OrderBook, Quote


class OrderAck:
    """订单提交确认."""

    def __init__(
        self,
        success: bool,
        order_id: Optional[str] = None,
        message: Optional[str] = None,
        raw: Optional[Dict] = None,
    ):
        self.success = success
        self.order_id = order_id or ""
        self.message = message or ""
        self.raw = raw or {}


class CancelAck:
    """撤单确认."""

    def __init__(
        self,
        success: bool,
        order_id: Optional[str] = None,
        message: Optional[str] = None,
        raw: Optional[Dict] = None,
    ):
        self.success = success
        self.order_id = order_id or ""
        self.message = message or ""
        self.raw = raw or {}


class ExecutionAdapter(ABC):
    """交易所执行适配器抽象基类.

    所有具体交易所（Hyperliquid、纸盘等）都必须实现此接口，
    保证 OMS 核心与交易所细节解耦。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """适配器名称."""
        ...

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """是否连接真实交易所."""
        ...

    @abstractmethod
    async def submit(self, order: "Order") -> OrderAck:
        """提交订单到交易所."""
        ...

    @abstractmethod
    async def cancel(self, order_id: str, client_order_id: Optional[str] = None) -> CancelAck:
        """撤销订单."""
        ...

    @abstractmethod
    async def query_order(self, order_id: str) -> Optional["Order"]:
        """查询订单最新状态."""
        ...

    @abstractmethod
    async def sync_positions(self) -> Dict[str, "Position"]:
        """同步当前持仓."""
        ...

    @abstractmethod
    async def sync_account(self) -> "AccountSnapshot":
        """同步账户快照."""
        ...

    async def get_quote(self, symbol: str) -> Optional["Quote"]:
        """获取最新报价，供 router 使用。默认返回 None。"""
        return None

    async def get_order_book(self, symbol: str, depth: int = 10) -> Optional["OrderBook"]:
        """获取订单簿，供 router/paper 使用。默认返回 None。"""
        return None

    async def get_recent_volume(
        self, symbol: str, window_seconds: int = 300
    ) -> Optional[Decimal]:
        """获取近期成交量，供 VWAP/POV 使用。默认返回 None。"""
        return None

    async def get_volume_profile(
        self,
        symbol: str,
        duration_seconds: int = 86400,
        buckets: int = 24,
    ) -> Optional[Any]:
        """获取成交量分布，供 VWAP 使用。默认返回 None。"""
        return None

    @property
    def is_simulation(self) -> bool:
        """是否为本地仿真机构盘（与实盘地位相同，只是不连接真实交易所）."""
        return not self.is_live

    def close(self):
        """释放资源，子类可重写."""
        pass
