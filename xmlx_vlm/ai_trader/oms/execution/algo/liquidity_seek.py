"""Liquidity Seek 执行算法."""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any, Optional

from xmlx_vlm.ai_trader.oms.constants import OrderState
from xmlx_vlm.ai_trader.oms.execution.algo.base import ExecutionAlgorithm, ParentOrder
from xmlx_vlm.ai_trader.oms.routing.context import RoutingContext
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms

logger = logging.getLogger(__name__)


class LiquiditySeekAlgorithm(ExecutionAlgorithm):
    """流动性搜寻算法.

    参数：
    - slice_qty: 每次尝试的量（默认 total/10）
    - patience_seconds: 每档耐心时间（默认 5）
    - aggression_levels: ["passive", "normal", "aggressive"]
    """

    @property
    def name(self) -> str:
        return "liquidity_seek"

    async def start(
        self,
        parent: ParentOrder,
        router: Any,
        on_child_update: Optional[Any] = None,
    ) -> None:
        self._parent = parent
        self._router = router
        parent.state = OrderState.SUBMITTED

        slice_qty = to_decimal(parent.params.get("slice_qty", parent.total_qty / Decimal("10")))
        patience = int(parent.params.get("patience_seconds", 5))
        levels = parent.params.get("aggression_levels", ["passive", "normal", "aggressive"])
        tick_seconds = parent.params.get("tick_seconds")
        if tick_seconds is not None:
            patience = max(0, int(tick_seconds))

        while not self.is_cancelled and parent.remaining_qty > ZERO:
            qty = min(slice_qty, parent.remaining_qty)
            if qty <= ZERO:
                break

            filled_in_slice = ZERO
            for level in levels:
                if self.is_cancelled or parent.remaining_qty <= ZERO:
                    break

                child = self._create_child(
                    qty=qty,
                    price=None,
                    order_type="market",
                    urgency=level,
                )
                parent.child_orders.append(child)

                try:
                    await router.submit(child, RoutingContext(urgency=level))
                except Exception as exc:
                    logger.warning("LiquiditySeek child submit failed: %s", exc)

                if child.filled_qty > ZERO:
                    parent.apply_child_fill(child)
                    filled_in_slice += child.filled_qty
                    qty -= child.filled_qty

                if child.remaining_qty <= ZERO:
                    break

                if level != levels[-1] and not self.is_cancelled and patience > 0:
                    await asyncio.sleep(patience)

            if filled_in_slice <= ZERO and not self.is_cancelled and patience > 0:
                # 完全没成交，等待下一轮
                await asyncio.sleep(patience)

        parent.state = (
            OrderState.FILLED
            if parent.remaining_qty <= ZERO
            else (OrderState.CANCELLED if self.is_cancelled else OrderState.PARTIAL_FILLED)
        )
        parent.updated_at_ms = utc_now_ms()
