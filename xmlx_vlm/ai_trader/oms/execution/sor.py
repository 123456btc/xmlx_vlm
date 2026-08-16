# SPDX-License-Identifier: Apache-2.0
"""
Smart Order Router (SOR): Passive Maker-First Execution with Aggressive Fallback.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Optional

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState, OrderType
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.interfaces.execution_adapter import ExecutionAdapter as ExchangeAdapter, OrderAck
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO

logger = logging.getLogger(__name__)


class SmartOrderRouter:
    """智能订单路由器 (Smart Order Router - SOR).
    
    实现 Maker-First 挂单策略：
    1. 优先尝试以盘口被动价 (Maker) 挂单，争取 0 手续费或 Maker 负费率返佣；
    2. 等待最多 timeout_seconds 秒；
    3. 若未能迅速完全撮合且盘口开始走远，自动取消剩余挂单并转为 IOC 追单保证最终成交。
    """

    def __init__(
        self,
        default_timeout_sec: float = 2.0,
        slippage_threshold: float = 0.002,  # 0.2% 价格漂移触发追单
    ):
        self.default_timeout_sec = default_timeout_sec
        self.slippage_threshold = to_decimal(slippage_threshold)

    async def route_order(
        self,
        adapter: ExchangeAdapter,
        order: Order,
        mark_price: Optional[Decimal | float] = None,
        maker_first: bool = False,
        timeout_sec: Optional[float] = None,
    ) -> OrderAck:
        """根据策略路由订单执行."""
        if not maker_first or order.order_type != OrderType.MARKET:
            # 常规订单直接提交给底层适配器
            return await adapter.submit(order)

        timeout = timeout_sec if timeout_sec is not None else self.default_timeout_sec
        price_dec = to_decimal(mark_price) if mark_price else to_decimal(order.price)

        if price_dec <= ZERO:
            # 无有效参考价，直接以市价单提交
            return await adapter.submit(order)

        logger.info(
            "SOR: Initiating Maker-First execution for %s %s %s @ ~%s",
            order.symbol, order.side.value, order.qty, price_dec
        )

        # 1. 尝试以 Maker 限价单发出
        maker_order = Order(
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            order_type=OrderType.LIMIT,
            qty=order.qty,
            price=price_dec,
            reduce_only=order.reduce_only,
        )

        try:
            ack = await adapter.submit(maker_order)
            if not ack.success or maker_order.state == OrderState.REJECTED:
                # 挂单被拒，立即回退到常规市价单
                logger.warning("SOR: Maker order rejected, falling back to direct IOC market order")
                return await adapter.submit(order)

            if maker_order.state == OrderState.FILLED:
                # 瞬间全部成交，达成最优 Maker 执行
                logger.info("SOR: Maker order filled immediately for %s!", order.symbol)
                order.order_id = maker_order.order_id
                order.fills = maker_order.fills
                order.filled_qty = maker_order.filled_qty
                order.avg_fill_price = maker_order.avg_fill_price
                order.remaining_qty = maker_order.remaining_qty
                order.transition_to(OrderState.FILLED)
                return ack

            # 2. 等待撮合
            await asyncio.sleep(timeout)

            # 3. 检查最新状态
            if maker_order.state in (OrderState.FILLED, OrderState.CANCELLED):
                return ack

            # 4. 超时未全成，撤单并以市价 IOC 追单剩余未成部分
            remaining = maker_order.remaining_qty if maker_order.remaining_qty > ZERO else order.qty
            logger.info(
                "SOR: Maker order timeout after %.1fs for %s. Canceling and falling back to IOC for remaining %s",
                timeout, order.symbol, remaining
            )

            try:
                await adapter.cancel(maker_order.client_order_id)
            except Exception as cancel_err:
                logger.warning("SOR: Cancel resting maker order error (may already be filled): %s", cancel_err)

            # 发出剩余部分的市价单
            fallback_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=OrderType.MARKET,
                qty=remaining,
                price=price_dec,
                reduce_only=order.reduce_only,
            )
            fallback_ack = await adapter.submit(fallback_order)
            order.order_id = fallback_order.order_id
            order.fills.extend(fallback_order.fills)
            order.filled_qty += fallback_order.filled_qty
            order.avg_fill_price = fallback_order.avg_fill_price
            order.remaining_qty = fallback_order.remaining_qty
            if fallback_order.state in (OrderState.FILLED, OrderState.PARTIAL_FILLED):
                order.transition_to(fallback_order.state)
            return fallback_ack

        except Exception as exc:
            logger.error("SOR: Smart routing encountered error: %s, executing IOC fallback", exc)
            return await adapter.submit(order)
