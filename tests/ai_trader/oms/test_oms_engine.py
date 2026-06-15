"""测试 OMS 引擎."""

import asyncio
from decimal import Decimal

import pytest

from xmlx_vlm.ai_trader.oms.config.settings import OMSSettings
from xmlx_vlm.ai_trader.oms.constants import OrderSide
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine


def test_oms_engine_default_is_paper():
    settings = OMSSettings(exchange="paper", live_enabled=False)
    engine = OMSEngine(settings=settings)
    assert not engine.is_live
    engine.close()


def test_oms_engine_create_order():
    settings = OMSSettings(exchange="paper", live_enabled=False)
    engine = OMSEngine(settings=settings)
    order = engine.create_order(
        symbol="BTC/USDC",
        side="buy",
        qty=Decimal("0.01"),
        order_type="market",
    )
    assert order.side == OrderSide.BUY
    engine.close()


@pytest.mark.anyio
async def test_oms_engine_submit_paper_order():
    settings = OMSSettings(exchange="paper", live_enabled=False)
    engine = OMSEngine(settings=settings)
    order = engine.create_order(
        symbol="BTC/USDC",
        side="buy",
        qty=Decimal("0.01"),
        order_type="market",
        price=Decimal("50000"),
    )
    result = await engine.submit_order(order)
    assert result["status"] == "submitted"
    engine.close()


def test_oms_engine_emergency_stop():
    settings = OMSSettings(exchange="paper", live_enabled=False)
    engine = OMSEngine(settings=settings)
    result = asyncio.run(engine.emergency_stop(flatten=False))
    assert result["status"] == "killed"
    assert engine.kill_switch.is_locked
    engine.close()
