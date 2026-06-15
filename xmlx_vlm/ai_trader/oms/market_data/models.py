"""市场数据模型."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO


@dataclass
class Quote:
    """行情报价."""

    symbol: str
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    last: Optional[Decimal] = None
    mark: Optional[Decimal] = None
    timestamp_ms: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.bid is not None:
            self.bid = to_decimal(self.bid)
        if self.ask is not None:
            self.ask = to_decimal(self.ask)
        if self.last is not None:
            self.last = to_decimal(self.last)
        if self.mark is not None:
            self.mark = to_decimal(self.mark)

    def mid(self) -> Optional[Decimal]:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / Decimal("2")
        return self.mark or self.last

    def spread_pct(self) -> Optional[Decimal]:
        if self.bid is None or self.ask is None or self.bid <= ZERO:
            return None
        return (self.ask - self.bid) / self.bid * Decimal("100")

    def best_buy_price(self) -> Optional[Decimal]:
        """买方最优成交价：默认 ask（买需吃单）."""
        return self.ask or self.mark or self.last

    def best_sell_price(self) -> Optional[Decimal]:
        """卖方最优成交价：默认 bid（卖需吃单）."""
        return self.bid or self.mark or self.last


@dataclass
class OrderBookLevel:
    """订单簿档位."""

    price: Decimal
    qty: Decimal
    side: str = ""  # bid / ask

    def __post_init__(self):
        self.price = to_decimal(self.price)
        self.qty = to_decimal(self.qty)


@dataclass
class OrderBook:
    """订单簿快照."""

    symbol: str
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)
    timestamp_ms: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.bids = [OrderBookLevel(price=b.price, qty=b.qty, side="bid") for b in self.bids]
        self.asks = [OrderBookLevel(price=a.price, qty=a.qty, side="ask") for a in self.asks]
        # 买高卖低排序
        self.bids.sort(key=lambda x: x.price, reverse=True)
        self.asks.sort(key=lambda x: x.price)

    def best_bid(self) -> Optional[Decimal]:
        return self.bids[0].price if self.bids else None

    def best_ask(self) -> Optional[Decimal]:
        return self.asks[0].price if self.asks else None

    def depth_at(self, side: str, max_price: Optional[Decimal] = None) -> Decimal:
        """指定方向累计深度，可限制价格区间."""
        levels = self.bids if side.lower() == "bid" else self.asks
        total = ZERO
        for level in levels:
            if max_price is not None:
                if side.lower() == "bid" and level.price < max_price:
                    break
                if side.lower() == "ask" and level.price > max_price:
                    break
            total += level.qty
        return total

    def impact_price(self, side: str, qty: Decimal) -> Optional[Decimal]:
        """估算吃掉 qty 后的加权成交均价."""
        qty = to_decimal(qty)
        if qty <= ZERO:
            return None
        levels = self.bids if side.lower() == "bid" else self.asks
        if not levels:
            return None
        remaining = qty
        total_cost = ZERO
        filled = ZERO
        for level in levels:
            take = min(level.qty, remaining)
            if take <= ZERO:
                continue
            total_cost += level.price * take
            filled += take
            remaining -= take
            if remaining <= ZERO:
                break
        if filled <= ZERO:
            return None
        return total_cost / filled


@dataclass
class VolumeProfile:
    """成交量分布（用于 VWAP）."""

    symbol: str
    total_volume: Decimal = ZERO
    buckets: List[Decimal] = field(default_factory=list)
    bucket_labels: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.total_volume = to_decimal(self.total_volume)
        self.buckets = [to_decimal(b) for b in self.buckets]

    def weights(self) -> List[Decimal]:
        if self.total_volume <= ZERO:
            return [Decimal("1")] * max(len(self.buckets), 1)
        return [b / self.total_volume for b in self.buckets]
