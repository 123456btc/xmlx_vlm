"""纸盘撮合引擎."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Optional, Tuple

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderType, TimeInForce
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.market_data.models import OrderBook, OrderBookLevel, Quote
from xmlx_vlm.ai_trader.oms.market_data.provider import MarketDataProvider
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO, quantize_price
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms

logger = logging.getLogger(__name__)


class PaperMatcher:
    """基于 order book 的纸盘撮合."""

    def __init__(
        self,
        market_data_provider: Optional[MarketDataProvider] = None,
        fill_slippage_pct: Decimal = Decimal("0.0"),
        default_price: Decimal = Decimal("50000"),
        synthetic_spread_pct: Decimal = Decimal("0.02"),
        synthetic_depth_qty: Decimal = Decimal("100"),
    ):
        self._provider = market_data_provider
        self._fill_slippage_pct = to_decimal(fill_slippage_pct)
        self._default_price = to_decimal(default_price)
        self._synthetic_spread_pct = to_decimal(synthetic_spread_pct)
        self._synthetic_depth_qty = to_decimal(synthetic_depth_qty)

    async def match(self, order: Order) -> Tuple[Optional[Decimal], Decimal]:
        """撮合订单，返回 (成交价, 成交量)."""
        book = await self._get_book(order.symbol)
        quote = await self._get_quote(order.symbol)

        if order.order_type == OrderType.MARKET:
            return self._match_market(order, book, quote)

        if order.order_type == OrderType.LIMIT:
            return self._match_limit(order, book, quote)

        # 其他类型按市价处理
        return self._match_market(order, book, quote)

    async def _get_book(self, symbol: str) -> Optional[OrderBook]:
        if self._provider is not None:
            try:
                return await self._provider.get_order_book(symbol)
            except Exception as exc:
                logger.warning("paper matcher failed to get book: %s", exc)
        return self._synthetic_book(symbol)

    async def _get_quote(self, symbol: str) -> Optional[Quote]:
        if self._provider is not None:
            try:
                return await self._provider.get_quote(symbol)
            except Exception as exc:
                logger.warning("paper matcher failed to get quote: %s", exc)
        return self._synthetic_quote(symbol)

    def _match_market(
        self, order: Order, book: Optional[OrderBook], quote: Optional[Quote]
    ) -> Tuple[Optional[Decimal], Decimal]:
        """市价单撮合：按对手方最优价 + slippage 成交."""
        ref = self._reference_price(order, book, quote)
        if ref is None:
            return None, ZERO

        # 确定吃单价
        if order.side == OrderSide.BUY:
            exec_price = book.best_ask() if book and book.asks else ref
        else:
            exec_price = book.best_bid() if book and book.bids else ref

        exec_price = self._apply_slippage(exec_price, order.side)
        filled_qty = self._compute_fill_qty(order, book)
        return quantize_price(exec_price), filled_qty

    def _match_limit(
        self, order: Order, book: Optional[OrderBook], quote: Optional[Quote]
    ) -> Tuple[Optional[Decimal], Decimal]:
        """限价单撮合：判断价格是否触及."""
        if order.price is None:
            return self._match_market(order, book, quote)

        ref = self._reference_price(order, book, quote)
        if ref is None:
            return None, ZERO

        # 买方限价 >= ask 或 卖方限价 <= bid 时成交
        can_fill = False
        if order.side == OrderSide.BUY:
            best_ask = book.best_ask() if book and book.asks else ref
            if order.price >= best_ask:
                can_fill = True
                exec_price = min(order.price, best_ask)
        else:
            best_bid = book.best_bid() if book and book.bids else ref
            if order.price <= best_bid:
                can_fill = True
                exec_price = max(order.price, best_bid)

        if not can_fill:
            # GTC 挂单未成交
            if order.time_in_force == TimeInForce.GTC:
                return quantize_price(order.price), ZERO
            # IOC/FOK 立即取消
            return quantize_price(order.price), ZERO

        exec_price = self._apply_slippage(exec_price, order.side)
        filled_qty = self._compute_fill_qty(order, book)
        if order.time_in_force == TimeInForce.FOK and filled_qty < order.qty:
            return quantize_price(exec_price), ZERO
        return quantize_price(exec_price), filled_qty

    def _reference_price(
        self, order: Order, book: Optional[OrderBook], quote: Optional[Quote]
    ) -> Optional[Decimal]:
        if book is not None:
            mid = book.best_bid() and book.best_ask()
            if mid:
                return (book.best_bid() + book.best_ask()) / Decimal("2")
        if quote is not None:
            return quote.mid() or quote.mark or quote.last
        return order.price or self._default_price

    def _apply_slippage(self, price: Decimal, side: OrderSide) -> Decimal:
        if self._fill_slippage_pct <= ZERO:
            return price
        factor = Decimal("1") + self._fill_slippage_pct / Decimal("100")
        if side == OrderSide.BUY:
            return price * factor
        return price / factor

    def _compute_fill_qty(self, order: Order, book: Optional[OrderBook]) -> Decimal:
        if book is None:
            return order.qty
        side = "ask" if order.side == OrderSide.BUY else "bid"
        depth = book.depth_at(side)
        if depth <= ZERO:
            # 无深度：按 IOC 退化为 0，GTC 挂单
            if order.time_in_force == TimeInForce.GTC:
                return ZERO
            return ZERO
        filled = min(order.qty, depth)
        if order.time_in_force == TimeInForce.FOK and filled < order.qty:
            return ZERO
        return filled

    def _synthetic_quote(self, symbol: str) -> Quote:
        """构造合成 quote."""
        mid = self._default_price
        half_spread = mid * self._synthetic_spread_pct / Decimal("200")
        return Quote(
            symbol=symbol,
            bid=mid - half_spread,
            ask=mid + half_spread,
            mark=mid,
            timestamp_ms=utc_now_ms(),
        )

    def _synthetic_book(self, symbol: str) -> OrderBook:
        """构造合成 order book."""
        quote = self._synthetic_quote(symbol)
        return OrderBook(
            symbol=symbol,
            bids=[OrderBookLevel(price=quote.bid, qty=self._synthetic_depth_qty)],
            asks=[OrderBookLevel(price=quote.ask, qty=self._synthetic_depth_qty)],
            timestamp_ms=utc_now_ms(),
        )
