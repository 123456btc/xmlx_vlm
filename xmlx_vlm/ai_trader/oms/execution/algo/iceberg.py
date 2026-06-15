"""Iceberg 执行算法."""

from __future__ import annotations

import asyncio
import logging
import random
from decimal import Decimal
from typing import Any, Optional

from xmlx_vlm.ai_trader.oms.constants import OrderState
from xmlx_vlm.ai_trader.oms.execution.algo.base import ExecutionAlgorithm, ParentOrder
from xmlx_vlm.ai_trader.oms.routing.context import RoutingContext
from xmlx_vlm.ai_trader.oms.utils.decimal import quantize_price, quantize_qty, to_decimal, ZERO
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms

logger = logging.getLogger(__name__)


class IcebergAlgorithm(ExecutionAlgorithm):
    """冰山订单执行算法.

    参数：
    - display_qty: 每次披露的挂单量
    - variance_pct: 价格和数量的随机化幅度（默认 0）
    - refresh_interval_seconds: 补单间隔（默认 5）
    - max_slippage_pct: 相对 mark 的最大价格偏移
    """

    @property
    def name(self) -> str:
        return "iceberg"

    async def start(
        self,
        parent: ParentOrder,
        router: Any,
        on_child_update: Optional[Any] = None,
    ) -> None:
        self._parent = parent
        self._router = router
        parent.state = OrderState.SUBMITTED

        display_qty = to_decimal(parent.params.get("display_qty", parent.total_qty / Decimal("10")))
        variance_pct = to_decimal(parent.params.get("variance_pct", "0"))
        refresh_interval = int(parent.params.get("refresh_interval_seconds", 5))
        max_slippage_pct = to_decimal(parent.params.get("max_slippage_pct", "0.5"))
        time_in_force = parent.params.get("time_in_force", "GTC")
        max_retries = int(parent.params.get("max_retries", 1000))
        tick_seconds = parent.params.get("tick_seconds")
        if tick_seconds is not None:
            refresh_interval = max(0, int(tick_seconds))

        retries = 0
        while not self.is_cancelled and parent.remaining_qty > ZERO:
            qty = min(display_qty, parent.remaining_qty)
            if variance_pct > ZERO:
                qty = self._jitter_qty(qty, variance_pct)
                qty = min(qty, parent.remaining_qty)

            if qty <= ZERO:
                break

            # 获取参考价构造 passive 限价单
            quote = None
            try:
                quote = await router.adapter.get_quote(parent.symbol)
            except Exception as exc:
                logger.warning("Iceberg failed to get quote: %s", exc)

            price = self._build_passive_price(parent, quote, variance_pct, max_slippage_pct)
            child = self._create_child(
                qty=qty,
                price=price,
                order_type="limit",
                time_in_force=time_in_force,
                urgency="passive",
            )
            parent.child_orders.append(child)

            try:
                await router.submit(child, RoutingContext(urgency="passive"))
            except Exception as exc:
                logger.warning("Iceberg child submit failed: %s", exc)

            if child.filled_qty > ZERO:
                parent.apply_child_fill(child)

            if parent.remaining_qty <= ZERO:
                break

            # GTC 未成交时等待 refresh_interval；IOC/FOK 未成交则直接下一轮
            if child.filled_qty <= ZERO and not self.is_cancelled:
                if time_in_force.upper() == "GTC":
                    retries += 1
                    if retries >= max_retries:
                        logger.warning("Iceberg max retries reached")
                        break
                    if refresh_interval > 0:
                        await asyncio.sleep(refresh_interval)
                # IOC/FOK 未成交不等待，继续拆下一单
            elif child.filled_qty < qty and child.remaining_qty > ZERO:
                # 部分成交：立即补单
                pass

        parent.state = (
            OrderState.FILLED
            if parent.remaining_qty <= ZERO
            else (OrderState.CANCELLED if self.is_cancelled else OrderState.PARTIAL_FILLED)
        )
        parent.updated_at_ms = utc_now_ms()

    def _jitter_qty(self, qty: Decimal, variance_pct: Decimal) -> Decimal:
        if variance_pct <= ZERO:
            return qty
        factor = Decimal(str(random.uniform(
            float(1 - variance_pct / Decimal("100")),
            float(1 + variance_pct / Decimal("100")),
        )))
        return quantize_qty(qty * factor)

    def _build_passive_price(
        self,
        parent: ParentOrder,
        quote: Any,
        variance_pct: Decimal,
        max_slippage_pct: Decimal,
    ) -> Optional[Decimal]:
        if quote is None:
            return None
        mid = quote.mid()
        if mid is None:
            return None

        # 买挂 bid 附近，卖挂 ask 附近
        if parent.side.value == "buy":
            base = quote.bid or mid
            offset = -max_slippage_pct / Decimal("2")
        else:
            base = quote.ask or mid
            offset = max_slippage_pct / Decimal("2")

        if variance_pct > ZERO:
            offset += Decimal(str(random.uniform(
                float(-variance_pct / Decimal("2")),
                float(variance_pct / Decimal("2")),
            )))

        price = base * (Decimal("1") + offset / Decimal("100"))
        return quantize_price(price)
