# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for Hyperliquid order execution, reduce-only protection, and precision fixes.
"""

from decimal import Decimal
from unittest.mock import patch, AsyncMock
import pytest

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderType, PositionSide
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.core.position import Position
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.oms.config.settings import get_settings
from xmlx_vlm.ai_trader.oms.execution.hyperliquid.mapper import (
    format_hl_cloid,
    format_hl_price,
    format_hl_size,
    order_to_hl_action,
)
from xmlx_vlm.ai_trader.tools.trading import TradingTool


# ─── 1. Cloid & Formatting Tests ──────────────────────────────────────────────

def test_order_reduce_only_and_cloid_format():
    order = Order(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        qty=Decimal("0.123456"),
        price=Decimal("65432.123"),
        reduce_only=True,
    )
    assert order.reduce_only is True

    cloid = format_hl_cloid(order.client_order_id)
    assert cloid is not None
    assert cloid.startswith("0x")
    assert len(cloid) == 34


def test_precision_formatting():
    assert format_hl_price(65432.1) == 65432.0
    assert format_hl_price(0.00123456) == 0.001235
    assert format_hl_price(12.34567) == 12.346

    assert format_hl_size(0.123456, sz_decimals=4) == 0.1235
    assert format_hl_size(0.123456, sz_decimals=2) == 0.12
    assert format_hl_size(10.5, sz_decimals=0) in (10.0, 11.0)


# ─── 2. Hyperliquid Action Conversion Tests ───────────────────────────────────

def test_order_to_hl_action_market_ioc_and_reduce_only():
    market_order = Order(
        symbol="ETH/USDC",
        side=OrderSide.SELL,
        qty=Decimal("1.5"),
        order_type=OrderType.MARKET,
        price=Decimal("3450.5"),
        reduce_only=True,
    )
    action = order_to_hl_action(market_order, sz_decimals=4)
    assert action["type"] == "order"
    entry = action["orders"][0]
    assert entry["coin"] == "ETH"
    assert entry["isBuy"] is False
    assert entry["reduceOnly"] is True
    assert entry["orderType"]["limit"]["tif"] == "Ioc"
    assert entry["sz"] == 1.5

    limit_order = Order(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        qty=Decimal("0.5"),
        order_type=OrderType.LIMIT,
        price=Decimal("60000"),
        reduce_only=False,
    )
    action_limit = order_to_hl_action(limit_order, sz_decimals=5)
    entry_limit = action_limit["orders"][0]
    assert entry_limit["coin"] == "BTC"
    assert entry_limit["isBuy"] is True
    assert entry_limit["reduceOnly"] is False
    assert entry_limit["orderType"]["limit"]["tif"] == "Gtc"


# ─── 3. OMSEngine Close Position Tests ────────────────────────────────────────

@pytest.mark.asyncio
async def test_oms_engine_close_position_enforces_reduce_only():
    settings = get_settings()
    engine = OMSEngine(settings=settings)

    pos = Position(
        symbol="BTC/USDC",
        side=PositionSide.LONG,
        qty=Decimal("0.25"),
        avg_entry_price=Decimal("60000"),
    )
    engine.portfolio.sync_positions({"BTC/USDC": pos})
    assert not engine.portfolio.get_position("BTC/USDC").is_flat()

    with patch.object(engine, "sync", new_callable=AsyncMock), \
         patch.object(engine, "submit_order", new_callable=AsyncMock, return_value={"status": "ok"}) as mock_submit:
        closed_order = await engine.close_position("BTC")
        assert closed_order is not None
        assert closed_order.symbol == "BTC/USDC"
        assert closed_order.side == OrderSide.SELL
        assert closed_order.qty == Decimal("0.25")
        assert closed_order.reduce_only is True
        mock_submit.assert_called_once()


# ─── 4. TradingTool Tool Entry Tests ──────────────────────────────────────────

def test_trading_tool_reduce_only_passthrough():
    settings = get_settings()
    engine = OMSEngine(settings=settings)
    tool = TradingTool(oms=engine)

    # 0.01 * 60000 = 600 notional (within 1000 max limit)
    with patch.object(tool, "_current_price", return_value=60000.0):
        res = tool.place_order(
            symbol="BTC/USDC",
            side="sell",
            qty=0.01,
            mode="paper",
            order_type="market",
            reduce_only=True,
        )
        assert "已提交" in res or "状态" in res
