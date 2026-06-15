"""Hyperliquid 后台订单同步 worker."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Set

from xmlx_vlm.ai_trader.oms.constants import OrderState
from xmlx_vlm.ai_trader.oms.core.order import Fill, Order
from xmlx_vlm.ai_trader.oms.events.types import FillEvent, OrderEvent
from xmlx_vlm.ai_trader.oms.interfaces.execution_adapter import ExecutionAdapter
from xmlx_vlm.ai_trader.oms.order_sync.base import OrderSyncWorker, SyncResult
from xmlx_vlm.ai_trader.oms.utils.decimal import ZERO
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms

logger = logging.getLogger(__name__)


class HyperliquidOrderSyncWorker(OrderSyncWorker):
    """Hyperliquid 专用订单同步.

    策略：
    1. 对 OMS 中所有未终结订单调用 orderStatus
    2. 根据响应更新状态、已成交数量
    3. 用 userFills 兜底最近成交，防止漏记
    """

    def __init__(self, adapter: ExecutionAdapter, oms: Any, interval_seconds: int = 5):
        super().__init__(adapter, oms, interval_seconds)
        self._seen_fill_ids: Set[str] = set()

    async def sync_once(self) -> SyncResult:
        result = SyncResult()

        # 1. 同步未终结订单
        open_orders = self._get_open_orders()
        for order in open_orders:
            result.orders_checked += 1
            try:
                updated = await self.adapter.query_order(
                    order.order_id or order.client_order_id
                )
                if updated is None:
                    continue
                changed = self._order_changed(order, updated)
                fills_applied = self._apply_order_update(order, updated)
                result.fills_applied += fills_applied
                if changed:
                    result.orders_updated += 1
            except Exception as exc:
                msg = f"sync order {order.client_order_id} failed: {exc}"
                logger.warning(msg)
                result.errors.append(msg)

        # 2. userFills 兜底
        try:
            fills = await self.adapter.query_recent_fills(limit=100)
            for fill in fills:
                if fill.fill_id in self._seen_fill_ids:
                    continue
                self._seen_fill_ids.add(fill.fill_id)
                if self._apply_fill(fill):
                    result.fills_applied += 1
        except Exception as exc:
            msg = f"sync recent fills failed: {exc}"
            logger.warning(msg)
            result.errors.append(msg)

        return result

    def _get_open_orders(self) -> List[Order]:
        """返回 OMS 中未终结的订单."""
        return [o for o in self.oms.list_orders() if not o.is_done()]

    def _order_changed(self, local: Order, remote: Order) -> bool:
        return (
            local.state != remote.state
            or local.filled_qty != remote.filled_qty
            or local.order_id != remote.order_id
        )

    def _apply_order_update(self, local: Order, remote: Order) -> int:
        """把远程订单状态合并到本地订单，返回新增 fill 数量."""
        new_fills = 0
        old_filled = local.filled_qty
        new_filled = remote.filled_qty

        if new_filled > old_filled:
            fill_qty = new_filled - old_filled
            fill_px = remote.avg_fill_price or local.price or ZERO
            fill = Fill(
                fill_id=f"sync:{local.order_id or local.client_order_id}:{new_filled}",
                order_id=local.order_id or local.client_order_id,
                symbol=local.symbol,
                side=local.side,
                qty=fill_qty,
                price=fill_px,
                timestamp_ms=utc_now_ms(),
                raw=remote.to_dict() if hasattr(remote, "to_dict") else {},
            )
            local.apply_fill(fill)
            self.oms._process_fill(local, fill)
            self._publish_fill_event(local, fill)
            new_fills += 1

        # 状态迁移
        if local.state != remote.state:
            old_state = local.state
            local.state = remote.state
            local.order_id = remote.order_id or local.order_id
            self.oms._publish_order_event(local, self._state_to_event_type(remote.state))
            logger.info(
                "Order %s state updated: %s -> %s",
                local.client_order_id,
                old_state.value,
                remote.state.value,
            )
        else:
            # 即使没有状态变化，也可能更新 order_id
            local.order_id = remote.order_id or local.order_id

        return new_fills

    def _apply_fill(self, fill: Fill) -> bool:
        """从 userFills 反查本地订单并应用 fill."""
        order = self.oms.get_order(fill.order_id) or self.oms.get_order(fill.fill_id)
        if order is None:
            # 尝试通过 symbol + side 匹配最近订单（简化兜底）
            order = self._find_candidate_order(fill)
        if order is None or order.is_done():
            return False

        # 避免重复应用
        existing_ids = {f.fill_id for f in order.fills}
        if fill.fill_id in existing_ids:
            return False

        order.apply_fill(fill)
        self.oms._process_fill(order, fill)
        self._publish_fill_event(order, fill)
        return True

    def _find_candidate_order(self, fill: Fill) -> Any:
        """根据 symbol/side 找最可能匹配的未完结订单."""
        candidates = [
            o for o in self.oms.list_orders()
            if o.symbol == fill.symbol and not o.is_done()
            and o.side == fill.side
        ]
        if not candidates:
            return None
        # 优先选已部分成交的
        partial = [o for o in candidates if o.state == OrderState.PARTIAL_FILLED]
        return partial[0] if partial else candidates[0]

    def _publish_fill_event(self, order: Order, fill: Fill) -> None:
        try:
            from xmlx_vlm.ai_trader.oms.constants import EventType

            event_type = (
                EventType.ORDER_FILLED
                if order.state == OrderState.FILLED
                else EventType.ORDER_PARTIAL_FILLED
            )
            self.oms.event_bus.publish(
                FillEvent(
                    event_type=event_type,
                    fill_id=fill.fill_id,
                    order_id=order.order_id or order.client_order_id,
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    qty=fill.qty,
                    price=fill.price,
                    fee=fill.fee,
                    raw=fill.raw,
                )
            )
        except Exception:
            logger.exception("Failed to publish fill event")

    def _state_to_event_type(self, state: OrderState):
        from xmlx_vlm.ai_trader.oms.constants import EventType

        mapping = {
            OrderState.FILLED: EventType.ORDER_FILLED,
            OrderState.PARTIAL_FILLED: EventType.ORDER_PARTIAL_FILLED,
            OrderState.CANCELLED: EventType.ORDER_CANCELLED,
            OrderState.REJECTED: EventType.ORDER_REJECTED,
            OrderState.EXPIRED: EventType.ORDER_EXPIRED,
        }
        return mapping.get(state, EventType.ORDER_ACKED)
