"""内部 Order/Trade 与 Hyperliquid API 格式映射."""

from __future__ import annotations

import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState, OrderType, TimeInForce, PositionSide
from xmlx_vlm.ai_trader.oms.core.order import Fill, Order
from xmlx_vlm.ai_trader.oms.core.position import Position
from xmlx_vlm.ai_trader.oms.core.trade import Trade
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO


HL_TIF_MAP = {
    TimeInForce.GTC: "Gtc",
    TimeInForce.IOC: "Ioc",
    TimeInForce.FOK: "Fok",
}


def coin_from_symbol(symbol: str) -> str:
    """从 BTC/USDC 提取 BTC."""
    symbol = symbol.strip().upper()
    if "/" in symbol:
        return symbol.split("/")[0]
    for quote in ("USDC", "USDT", "USD", "BTC", "ETH"):
        if symbol.endswith(quote):
            return symbol[: -len(quote)]
    return symbol


def order_to_hl_action(order: Order) -> Dict[str, Any]:
    """把内部 Order 转为 Hyperliquid 下单 action."""
    coin = coin_from_symbol(order.symbol)
    tif = HL_TIF_MAP.get(order.time_in_force, "Gtc")
    action: Dict[str, Any] = {
        "type": "order",
        "orders": [
            {
                "coin": coin,
                "isBuy": order.side == OrderSide.BUY,
                "sz": float(order.qty),
                "limitPx": float(order.price) if order.price else float(ZERO),
                "orderType": {
                    "limit": {"tif": tif}
                },
                "reduceOnly": False,
                "cloid": order.client_order_id,
            }
        ],
        "grouping": "na",
    }
    return action


def hl_response_to_order(hl_response: Any, order: Order) -> Order:
    """根据 HL 响应更新 Order 状态."""
    if not isinstance(hl_response, dict):
        return order

    status = hl_response.get("status", "")
    if status == "ok":
        # 响应结构：{'status': 'ok', 'response': {'type': 'order', 'data': {...}}}
        data = hl_response.get("response", {}).get("data", {})
        statuses = data.get("statuses", [])
        if statuses:
            first = statuses[0]
            if "resting" in first:
                order.order_id = first["resting"].get("oid", order.order_id)
                order.transition_to(OrderState.ACKNOWLEDGED)
            elif "filled" in first:
                order.order_id = first["filled"].get("oid", order.order_id)
                order.transition_to(OrderState.FILLED)
            elif "error" in first:
                order.transition_to(OrderState.REJECTED, reason=first["error"])
    else:
        order.transition_to(
            OrderState.REJECTED,
            reason=hl_response.get("response", "unknown error"),
        )
    order.raw_response = hl_response
    return order


def hl_fill_to_trade(fill: Dict[str, Any], order: Order) -> Trade:
    """Hyperliquid fill 转为内部 Trade."""
    return Trade(
        trade_id=str(fill.get("tid", "")),
        order_id=order.order_id or order.client_order_id,
        client_order_id=order.client_order_id,
        symbol=order.symbol,
        side=order.side,
        qty=to_decimal(fill.get("sz", "0")),
        price=to_decimal(fill.get("px", "0")),
        fee=to_decimal(fill.get("fee", "0")),
        timestamp_ms=int(fill.get("time", time.time() * 1000)),
        exchange="hyperliquid",
        raw=fill,
    )


def hl_positions_to_positions(hl_positions: List[Dict[str, Any]]) -> Dict[str, Position]:
    """HL 持仓格式转为内部 Position."""
    result: Dict[str, Position] = {}
    for p in hl_positions:
        coin = p.get("coin", "")
        symbol = f"{coin}/USDC"
        size = to_decimal(p.get("szi", "0"))
        entry_px = to_decimal(p.get("entryPx", "0"))
        unrealized_pnl = to_decimal(p.get("unrealizedPnl", "0"))
        pos_side = PositionSide.LONG if size > ZERO else PositionSide.SHORT
        
        mark_px = to_decimal(p.get("markPx")) if p.get("markPx") is not None else ZERO
        if mark_px == ZERO and entry_px != ZERO and size != ZERO:
            mark_px = entry_px + (unrealized_pnl / abs(size)) if size > ZERO else entry_px - (unrealized_pnl / abs(size))
        if mark_px == ZERO:
            mark_px = entry_px
        liq_px = to_decimal(p.get("liquidationPx")) if p.get("liquidationPx") is not None else ZERO
        lev_info = p.get("leverage", {})
        lev_type = lev_info.get("type", "cross")
        lev_val = int(lev_info.get("value", 1))
            
        result[symbol] = Position(
            symbol=symbol,
            side=pos_side,
            qty=abs(size),
            avg_entry_price=entry_px,
            mark_price=mark_px,
            unrealized_pnl=unrealized_pnl,
            leverage=lev_val,
            margin_type=lev_type,
            liq_price=liq_px,
        )
    return result
