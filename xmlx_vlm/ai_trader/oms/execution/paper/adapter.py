"""纸盘执行适配器."""

from __future__ import annotations

import logging
import uuid
from decimal import Decimal
from typing import Dict, Optional

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState, PositionSide, TimeInForce
from xmlx_vlm.ai_trader.oms.core.account import AccountSnapshot
from xmlx_vlm.ai_trader.oms.core.order import Fill, Order
from xmlx_vlm.ai_trader.oms.core.position import Position
from xmlx_vlm.ai_trader.oms.core.trade import Trade
from xmlx_vlm.ai_trader.oms.execution.paper.matcher import PaperMatcher
from xmlx_vlm.ai_trader.oms.interfaces.execution_adapter import (
    CancelAck,
    ExecutionAdapter,
    OrderAck,
)
from xmlx_vlm.ai_trader.oms.market_data.provider import MarketDataProvider, StaticMarketDataProvider
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms

logger = logging.getLogger(__name__)


class PaperExecutionAdapter(ExecutionAdapter):
    """本地仿真机构盘撮合适配器.

    地位与 hyperliquid 实盘相同：同样走 OMS、风控、审计、执行算法、
    SmartOrderRouter、市场冲击模型。区别仅在于不连接真实交易所，
    成交由本地 order book 模拟，零真实资金风险。
    """

    def __init__(
        self,
        market_data_tool=None,
        market_data_provider: Optional[MarketDataProvider] = None,
        fill_slippage_pct: Decimal = Decimal("0.0"),
        partial_fill_threshold: Optional[Decimal] = None,
        default_price: Decimal = Decimal("50000"),
        latency_ms: int = 0,
    ):
        self._market = market_data_tool
        self._market_data_provider = market_data_provider
        self._fill_slippage_pct = to_decimal(fill_slippage_pct)
        self._partial_fill_threshold = (
            to_decimal(partial_fill_threshold) if partial_fill_threshold else None
        )
        self._default_price = to_decimal(default_price)
        self._latency_ms = max(0, int(latency_ms))
        self._matcher = PaperMatcher(
            market_data_provider=market_data_provider,
            fill_slippage_pct=self._fill_slippage_pct,
            default_price=self._default_price,
        )
        self._orders: Dict[str, Order] = {}
        self._positions: Dict[str, Position] = {}
        self._account: AccountSnapshot = AccountSnapshot(
            equity=Decimal("100000"),
            available_margin=Decimal("100000"),
            used_margin=ZERO,
            cash=Decimal("100000"),
        )

    @property
    def name(self) -> str:
        return "paper"

    @property
    def is_live(self) -> bool:
        return False

    async def submit(self, order: Order) -> OrderAck:
        order.exchange = self.name

        if self._latency_ms > 0:
            import asyncio

            await asyncio.sleep(self._latency_ms / 1000.0)

        price, filled_qty = await self._matcher.match(order)
        if price is None or price <= ZERO:
            order.transition_to(OrderState.REJECTED, reason="no market price")
            self._orders[order.client_order_id] = order
            return OrderAck(
                success=False,
                order_id=order.client_order_id,
                message="no market price available",
            )

        order.order_id = order.client_order_id
        if order.state == OrderState.DRAFT:
            order.transition_to(OrderState.SUBMITTED)
        order.transition_to(OrderState.ACKNOWLEDGED)

        if filled_qty > ZERO:
            fill = Fill(
                fill_id=uuid.uuid4().hex[:16],
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                qty=filled_qty,
                price=price,
                timestamp_ms=utc_now_ms(),
            )
            order.apply_fill(fill)
            self._update_positions(order, fill)

        self._orders[order.client_order_id] = order
        return OrderAck(
            success=True,
            order_id=order.order_id,
            message="paper fill",
            raw={"fill_price": str(price), "filled_qty": str(filled_qty)},
        )

    async def cancel(self, order_id: str, client_order_id: Optional[str] = None) -> CancelAck:
        target = self._orders.get(client_order_id or order_id)
        if target is None or target.is_done():
            return CancelAck(success=False, order_id=order_id, message="order not found or done")
        if target.state not in (OrderState.CANCEL_REQUESTED,):
            target.transition_to(OrderState.CANCEL_REQUESTED)
        target.transition_to(OrderState.CANCELLED)
        return CancelAck(success=True, order_id=order_id)

    async def query_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    async def sync_positions(self) -> Dict[str, Position]:
        return dict(self._positions)

    async def sync_account(self) -> AccountSnapshot:
        return self._account

    async def get_quote(self, symbol: str) -> Optional[Any]:
        if self._market_data_provider is not None:
            return await self._market_data_provider.get_quote(symbol)
        return self._matcher._synthetic_quote(symbol)

    async def get_order_book(self, symbol: str, depth: int = 10) -> Optional[Any]:
        if self._market_data_provider is not None:
            return await self._market_data_provider.get_order_book(symbol, depth)
        return self._matcher._synthetic_book(symbol)

    def _update_positions(self, order: Order, fill: Fill) -> None:
        pos = self._positions.setdefault(
            order.symbol, Position(symbol=order.symbol, side=PositionSide("flat"))
        )
        pos.apply_fill(order.side.value, fill.qty, fill.price)
        if pos.is_flat():
            self._positions.pop(order.symbol, None)

        # 简化的账户更新
        notional = fill.qty * fill.price
        if order.side == OrderSide.BUY:
            self._account.cash -= notional
            self._account.used_margin += notional
        else:
            self._account.cash += notional
            self._account.used_margin = max(ZERO, self._account.used_margin - notional)
        self._account.available_margin = self._account.equity - self._account.used_margin
        self._account.timestamp_ms = utc_now_ms()
