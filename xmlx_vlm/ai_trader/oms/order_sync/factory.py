"""订单同步 worker 工厂."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from xmlx_vlm.ai_trader.oms.order_sync.hyperliquid_worker import HyperliquidOrderSyncWorker

if TYPE_CHECKING:
    from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
    from xmlx_vlm.ai_trader.oms.interfaces.execution_adapter import ExecutionAdapter
    from xmlx_vlm.ai_trader.oms.order_sync.base import OrderSyncWorker


def create_order_sync_worker(
    adapter: "ExecutionAdapter",
    oms: "OMSEngine",
    interval_seconds: int = 5,
) -> Optional["OrderSyncWorker"]:
    """根据适配器类型创建对应的订单同步 worker.

    - paper 适配器无需同步
    - hyperliquid 返回 HyperliquidOrderSyncWorker
    """
    if not adapter.is_live:
        return None

    name = adapter.name.lower()
    if name == "hyperliquid":
        return HyperliquidOrderSyncWorker(
            adapter=adapter,
            oms=oms,
            interval_seconds=interval_seconds,
        )

    # 其他交易所暂未实现
    return None
