"""订单同步 worker 抽象基类."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
    from xmlx_vlm.ai_trader.oms.interfaces.execution_adapter import ExecutionAdapter

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """一次同步的结果统计."""

    orders_checked: int = 0
    orders_updated: int = 0
    fills_applied: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "orders_checked": self.orders_checked,
            "orders_updated": self.orders_updated,
            "fills_applied": self.fills_applied,
            "errors": self.errors,
        }


class OrderSyncWorker(ABC):
    """后台订单同步 worker 基类.

    职责：
    - 定期从交易所查询未完成订单状态
    - 更新 OMS 本地订单状态
    - 应用新增成交到 portfolio
    - 发布 OrderEvent / FillEvent
    """

    def __init__(
        self,
        adapter: "ExecutionAdapter",
        oms: "OMSEngine",
        interval_seconds: int = 5,
    ):
        self.adapter = adapter
        self.oms = oms
        self.interval_seconds = max(1, interval_seconds)
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.is_running:
            logger.warning("OrderSyncWorker already running")
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("OrderSyncWorker started (interval=%ds)", self.interval_seconds)

    async def stop(self) -> None:
        if not self.is_running:
            return
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("OrderSyncWorker stopped")

    async def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                result = await self.sync_once()
                if result.orders_checked or result.fills_applied or result.errors:
                    logger.debug("Order sync result: %s", result.to_dict())
            except Exception:
                logger.exception("Order sync cycle failed")
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.interval_seconds
                )
            except asyncio.TimeoutError:
                pass

    @abstractmethod
    async def sync_once(self) -> SyncResult:
        """执行一次同步，子类实现."""
        ...
