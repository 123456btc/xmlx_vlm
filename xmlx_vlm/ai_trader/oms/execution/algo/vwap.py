"""VWAP 执行算法."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any, Optional

from xmlx_vlm.ai_trader.oms.constants import OrderState
from xmlx_vlm.ai_trader.oms.execution.algo.base import ExecutionAlgorithm, ParentOrder
from xmlx_vlm.ai_trader.oms.market_data.provider import MarketDataProvider
from xmlx_vlm.ai_trader.oms.routing.context import RoutingContext
from xmlx_vlm.ai_trader.oms.utils.decimal import quantize_qty, to_decimal, ZERO
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms

logger = logging.getLogger(__name__)


class VWAPAlgorithm(ExecutionAlgorithm):
    """成交量加权平均价格算法.

    参数：
    - duration_seconds: 总执行时长（默认 86400）
    - buckets: 拆分的桶数（默认 24）
    - urgency: 每 bucket 的 urgency
    """

    @property
    def name(self) -> str:
        return "vwap"

    async def start(
        self,
        parent: ParentOrder,
        router: Any,
        on_child_update: Optional[Any] = None,
    ) -> None:
        self._parent = parent
        self._router = router
        parent.state = OrderState.SUBMITTED

        duration = int(parent.params.get("duration_seconds", 86400))
        buckets = int(parent.params.get("buckets", 24))
        urgency = parent.params.get("urgency", "normal")
        tick_seconds = parent.params.get("tick_seconds")

        if buckets <= 0:
            buckets = 1
        bucket_seconds = max(0, duration // buckets)
        if tick_seconds is not None:
            bucket_seconds = max(0, int(tick_seconds))

        # 获取成交量分布权重
        weights = await self._get_weights(parent, duration, buckets)
        total_weight = sum(weights, ZERO)
        if total_weight <= ZERO:
            weights = [Decimal("1")] * buckets
            total_weight = Decimal(str(buckets))

        for i, weight in enumerate(weights):
            if self.is_cancelled or parent.is_done():
                break

            if i == len(weights) - 1:
                target_qty = parent.remaining_qty
            else:
                target_qty = quantize_qty(parent.total_qty * weight / total_weight)

            if target_qty <= ZERO:
                await self._sleep(bucket_seconds)
                continue

            child = self._create_child(
                qty=target_qty,
                price=None,
                order_type="market",
                urgency=urgency,
            )
            parent.child_orders.append(child)

            try:
                await router.submit(child, RoutingContext(urgency=urgency))
            except Exception as exc:
                logger.warning("VWAP child submit failed: %s", exc)

            if child.filled_qty > ZERO:
                parent.apply_child_fill(child)

            if i < len(weights) - 1:
                await self._sleep(bucket_seconds)

        parent.state = (
            OrderState.FILLED
            if parent.remaining_qty <= ZERO
            else (OrderState.PARTIAL_FILLED if parent.filled_qty > ZERO else OrderState.CANCELLED)
        )
        parent.updated_at_ms = utc_now_ms()

    async def _get_weights(self, parent: ParentOrder, duration: int, buckets: int) -> list:
        provider: Optional[MarketDataProvider] = getattr(
            self._router, "_market_data_provider", None
        )
        if provider is None:
            # 无 provider 时退化为等权
            return [Decimal("1")] * buckets
        try:
            profile = await provider.get_volume_profile(
                parent.symbol, duration_seconds=duration, buckets=buckets
            )
            if profile is not None:
                return profile.weights()
        except Exception as exc:
            logger.warning("VWAP failed to get volume profile: %s", exc)
        return [Decimal("1")] * buckets

    async def _sleep(self, seconds: int) -> None:
        if seconds > 0:
            await asyncio.sleep(seconds)
