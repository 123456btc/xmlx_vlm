"""智能订单路由."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, Optional

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState, OrderType, TimeInForce
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.exceptions import AdapterError
from xmlx_vlm.ai_trader.oms.impact.market_impact import MarketImpactModel
from xmlx_vlm.ai_trader.oms.interfaces.execution_adapter import (
    ExecutionAdapter,
    OrderAck,
)
from xmlx_vlm.ai_trader.oms.market_data.models import Quote
from xmlx_vlm.ai_trader.oms.routing.context import RoutingContext
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO, quantize_price

logger = logging.getLogger(__name__)


class SmartOrderRouter:
    """智能订单路由.

    职责：
    1. 获取行情报价。
    2. 估算 expected_slippage。
    3. 根据 urgency 选择下单方式（GTC / IOC / FOK）和价格偏移。
    4. 调用底层 ExecutionAdapter.submit()。

    当前仅支持单交易所路径；未来可扩展多交易所择优。
    """

    def __init__(
        self,
        adapter: ExecutionAdapter,
        impact_model: MarketImpactModel,
        default_max_slippage_pct: Decimal = Decimal("0.5"),
    ):
        self._adapter = adapter
        self._impact_model = impact_model
        self._default_max_slippage_pct = to_decimal(default_max_slippage_pct)

    @property
    def adapter(self) -> ExecutionAdapter:
        return self._adapter

    async def submit(
        self,
        order: Order,
        context: Optional[RoutingContext] = None,
    ) -> OrderAck:
        """路由提交 child order."""
        if order.state == OrderState.DRAFT:
            order.transition_to(OrderState.PRE_TRADE_OK)
        order.transition_to(OrderState.SENT)

        ctx = context
        if ctx is None or ctx.mark_price is None:
            built = await self._build_context(order)
            if ctx is not None:
                # 保留调用方指定的 urgency/max_slippage，其余用 built 补全
                ctx.mark_price = ctx.mark_price or built.mark_price
                ctx.bid = ctx.bid or built.bid
                ctx.ask = ctx.ask or built.ask
                ctx.spread_pct = ctx.spread_pct or built.spread_pct
                ctx.book_depth = ctx.book_depth or built.book_depth
                ctx.recent_volume = ctx.recent_volume or built.recent_volume
                ctx.volatility = ctx.volatility or built.volatility
            else:
                ctx = built
        self._apply_routing_decision(order, ctx)

        # 滑点检查：仅对市价/IOC/FOK 做
        if order.order_type == OrderType.MARKET or order.time_in_force in (
            TimeInForce.IOC,
            TimeInForce.FOK,
        ):
            estimate = self._impact_model.estimate(
                order_qty=order.qty,
                side=order.side,
                price=ctx.mark_price or order.price or ZERO,
                adv=ctx.recent_volume,
                spread_pct=ctx.spread_pct,
                volatility=ctx.volatility,
                urgency=ctx.urgency,
            )
            max_slippage = ctx.max_slippage_pct or self._default_max_slippage_pct
            if estimate.expected_slippage_pct > max_slippage:
                reason = (
                    f"expected slippage {estimate.expected_slippage_pct:.4f}% "
                    f"exceeds max {max_slippage:.4f}%"
                )
                order.transition_to(OrderState.REJECTED, reason=reason)
                return OrderAck(
                    success=False,
                    order_id=order.client_order_id,
                    message=reason,
                    raw={"impact": estimate.to_dict() if hasattr(estimate, "to_dict") else {}},
                )

        try:
            order.transition_to(OrderState.SUBMITTED)
            ack = await self._adapter.submit(order)
            return ack
        except AdapterError as exc:
            if not order.is_done():
                order.transition_to(OrderState.REJECTED, reason=str(exc))
            raise

    async def _build_context(self, order: Order) -> RoutingContext:
        """从 adapter 获取行情构造 RoutingContext."""
        quote = await self._adapter.get_quote(order.symbol)
        if quote is None:
            return RoutingContext(urgency="normal")

        book = await self._adapter.get_order_book(order.symbol, depth=10)
        depth = None
        if book is not None:
            side = "ask" if order.side == OrderSide.BUY else "bid"
            depth = book.depth_at(side)

        recent_volume = await self._adapter.get_recent_volume(order.symbol, window_seconds=300)
        volatility = None
        if hasattr(self._adapter, "get_volatility"):
            try:
                volatility = await self._adapter.get_volatility(order.symbol, window_days=30)
            except Exception:
                pass

        return RoutingContext(
            mark_price=quote.mark or quote.mid() or quote.last,
            bid=quote.bid,
            ask=quote.ask,
            spread_pct=quote.spread_pct(),
            book_depth=depth,
            recent_volume=recent_volume,
            volatility=volatility,
            urgency="normal",
        )

    def _apply_routing_decision(self, order: Order, ctx: RoutingContext) -> None:
        """根据 urgency 设置订单类型、TIF、价格."""
        urgency = ctx.urgency.lower()
        mark = ctx.mark_price or order.price or ZERO

        if urgency == "passive":
            tif = TimeInForce.GTC
        elif urgency == "aggressive":
            tif = TimeInForce.FOK
        else:  # normal
            tif = TimeInForce.IOC

        # 市价单统一转为 IOC/FOK limit 近似（HL 无原生 market）
        if order.order_type == OrderType.MARKET:
            order.order_type = OrderType.LIMIT
            order.time_in_force = tif
            order.price = self._crossing_price(order, ctx, mark, urgency)
        elif order.order_type == OrderType.LIMIT and order.price is not None:
            order.time_in_force = tif
            if urgency != "passive":
                order.price = self._crossing_price(order, ctx, order.price, urgency)
        else:
            order.time_in_force = tif

    def _crossing_price(
        self, order: Order, ctx: RoutingContext, base_price: Decimal, urgency: str
    ) -> Optional[Decimal]:
        """确保 IOC/FOK 限价单能吃到对手盘."""
        if order.side == OrderSide.BUY:
            best = ctx.ask or base_price
        else:
            best = ctx.bid or base_price

        if urgency == "passive":
            return quantize_price(base_price)

        # aggressive / normal：按 base 偏移，但至少要 crossing 对手盘
        if urgency == "aggressive":
            offset_pct = Decimal("0.20")
        else:
            offset_pct = Decimal("0.10")

        if order.side == OrderSide.BUY:
            price = max(base_price * (Decimal("1") + offset_pct / Decimal("100")), best)
        else:
            price = min(base_price * (Decimal("1") - offset_pct / Decimal("100")), best)
        return quantize_price(price)
