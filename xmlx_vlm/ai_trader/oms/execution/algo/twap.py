"""TWAP 执行算法."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any, Optional

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.execution.algo.base import ExecutionAlgorithm, ParentOrder
from xmlx_vlm.ai_trader.oms.routing.context import RoutingContext
from xmlx_vlm.ai_trader.oms.utils.decimal import quantize_qty, to_decimal, ZERO
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms

logger = logging.getLogger(__name__)


class TWAPAlgorithm(ExecutionAlgorithm):
    """时间加权平均价格算法.

    参数：
    - duration_seconds: 总执行时长（默认 300）
    - buckets: 拆分的桶数（默认 10）
    - leftover_mode: "carry"（未完成推到下一桶）或 "immediate"（立即补单）
    - urgency: 每 bucket 的 urgency
    """

    @property
    def name(self) -> str:
        return "twap"

    async def start(
        self,
        parent: ParentOrder,
        router: Any,
        on_child_update: Optional[Any] = None,
    ) -> None:
        self._parent = parent
        self._router = router
        parent.state = OrderState.SUBMITTED

        duration = int(parent.params.get("duration_seconds", 300))
        buckets = int(parent.params.get("buckets", 10))
        leftover_mode = parent.params.get("leftover_mode", "carry")
        urgency = parent.params.get("urgency", "normal")
        tick_seconds = parent.params.get("tick_seconds")

        if buckets <= 0:
            buckets = 1
        bucket_seconds = max(0, duration // buckets)
        if tick_seconds is not None:
            bucket_seconds = max(0, int(tick_seconds))
        base_qty = parent.total_qty / Decimal(buckets)
        leftover = ZERO

        for i in range(buckets):
            if self.is_cancelled or parent.is_done():
                break

            target_qty = quantize_qty(base_qty + leftover)
            if i == buckets - 1:
                # 最后一份：全部剩余
                target_qty = parent.remaining_qty

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
                await router.submit(
                    child,
                    RoutingContext(urgency=urgency),
                )
            except Exception as exc:
                logger.warning("TWAP child submit failed: %s", exc)

            if child.filled_qty > ZERO:
                parent.apply_child_fill(child)

            if child.remaining_qty > ZERO:
                if leftover_mode == "carry":
                    leftover = child.remaining_qty
                else:
                    # immediate：重试一次
                    retry = self._create_child(
                        qty=child.remaining_qty,
                        price=None,
                        order_type="market",
                        urgency="aggressive",
                    )
                    parent.child_orders.append(retry)
                    try:
                        await router.submit(retry, RoutingContext(urgency="aggressive"))
                    except Exception as exc:
                        logger.warning("TWAP immediate retry failed: %s", exc)
                    if retry.filled_qty > ZERO:
                        parent.apply_child_fill(retry)
                    leftover = retry.remaining_qty
            else:
                leftover = ZERO

            if i < buckets - 1:
                await self._sleep(bucket_seconds)

        if parent.remaining_qty <= ZERO:
            parent.state = OrderState.FILLED
        elif self.is_cancelled:
            parent.state = OrderState.CANCELLED
        else:
            parent.state = OrderState.FILLED if parent.filled_qty >= parent.total_qty else OrderState.PARTIAL_FILLED
        parent.updated_at_ms = utc_now_ms()

    async def _sleep(self, seconds: int) -> None:
        if seconds > 0:
            await asyncio.sleep(seconds)
