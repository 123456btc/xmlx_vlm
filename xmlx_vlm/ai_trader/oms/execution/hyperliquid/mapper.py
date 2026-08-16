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
    """从 BTC/USDC 提取 BTC (使用统一的 extract_base_coin SSOT)."""
    from xmlx_vlm.ai_trader.oms.utils.symbol import extract_base_coin
    return extract_base_coin(symbol)


def format_hl_price(px: float) -> float:
    """Hyperliquid 价格格式化：最多 5 位有效数字，最多 6 位小数."""
    if px <= 0:
        return 0.0
    # 5 位有效数字
    from decimal import Decimal
    d = Decimal(str(px))
    # 格式化为最多 5 位有效数字
    formatted = float(f"{px:.5g}")
    return round(formatted, 6)


def format_hl_size(sz: float, sz_decimals: int = 4) -> float:
    """Hyperliquid 下单数量格式化：按照资产规定的 szDecimals 截断/舍入."""
    if sz <= 0:
        return 0.0
    return round(sz, sz_decimals)


def format_hl_cloid(client_order_id: Optional[str]) -> Optional[str]:
    """格式化为 Hyperliquid 标准 128-bit (16 bytes) 34 字符 hex Cloid (0x...)."""
    if not client_order_id:
        return None
    h = str(client_order_id).replace("-", "").strip()
    if not h.startswith("0x"):
        h = "0x" + h
    # 若不足 34 位则右补 0，若超过则截取 34 位
    if len(h) < 34:
        h = h.ljust(34, "0")
    elif len(h) > 34:
        h = h[:34]
    return h


def order_to_hl_action(order: Order, sz_decimals: int = 4) -> Dict[str, Any]:
    """把内部 Order 转为 Hyperliquid 下单 action."""
    coin = coin_from_symbol(order.symbol)
    
    # 区分市价单与限价单
    is_market = (order.order_type == OrderType.MARKET)
    if is_market:
        tif = "Ioc"
    else:
        tif = HL_TIF_MAP.get(order.time_in_force, "Gtc")

    limit_px = float(order.price) if order.price else 0.0
    formatted_px = format_hl_price(limit_px)
    formatted_sz = format_hl_size(float(order.qty), sz_decimals=sz_decimals)
    cloid = format_hl_cloid(order.client_order_id)

    order_entry: Dict[str, Any] = {
        "coin": coin,
        "isBuy": order.side == OrderSide.BUY,
        "sz": formatted_sz,
        "limitPx": formatted_px,
        "orderType": {
            "limit": {"tif": tif}
        },
        "reduceOnly": bool(order.reduce_only),
    }
    if cloid:
        order_entry["cloid"] = cloid

    action: Dict[str, Any] = {
        "type": "order",
        "orders": [order_entry],
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
                order.order_id = str(first["resting"].get("oid", order.order_id or ""))
                if order.state != OrderState.ACKNOWLEDGED:
                    order.transition_to(OrderState.ACKNOWLEDGED)
            elif "filled" in first:
                fill_data = first["filled"]
                order.order_id = str(fill_data.get("oid", order.order_id or ""))
                filled_sz = to_decimal(fill_data.get("totalSz", order.qty))
                avg_px = to_decimal(fill_data.get("avgPx", order.price or ZERO))
                if filled_sz > ZERO:
                    order.filled_qty = filled_sz
                    order.remaining_qty = max(ZERO, order.qty - filled_sz)
                if avg_px > ZERO:
                    order.avg_fill_price = avg_px
                if order.state != OrderState.FILLED:
                    order.transition_to(OrderState.FILLED)
            elif "error" in first:
                err_msg = str(first["error"])
                order.transition_to(OrderState.REJECTED, reason=err_msg)
    else:
        err_msg = str(hl_response.get("response", "unknown error"))
        order.transition_to(OrderState.REJECTED, reason=err_msg)
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
