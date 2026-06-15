"""测试 OMSEngine.submit_algo."""

from decimal import Decimal

import pytest

from xmlx_vlm.ai_trader.oms.config.settings import OMSSettings
from xmlx_vlm.ai_trader.oms.constants import OrderSide
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.oms.execution.algo.base import ParentOrder


@pytest.fixture
def engine(tmp_path):
    settings = OMSSettings(
        exchange="paper",
        live_enabled=False,
        risk_profile="custom",
        max_orders_per_second=100,
        max_orders_per_minute=1000,
        max_single_order_notional=Decimal("1000000"),
        audit_db_path=tmp_path / "audit.db",
    )
    return OMSEngine(settings=settings)


@pytest.mark.anyio
async def test_submit_twap(engine):
    parent = ParentOrder(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        total_qty=Decimal("1.0"),
        algo_type="twap",
        params={"duration_seconds": 0, "buckets": 5, "tick_seconds": 0},
    )
    result = await engine.submit_algo(parent, mark_price=Decimal("50000"))
    assert result["status"] == "started"
    assert "algo_id" in result

    # 等待算法完成
    import asyncio
    for _ in range(20):
        status = engine.get_algo_status(result["algo_id"])
        if status and status["is_done"]:
            break
        await asyncio.sleep(0.1)

    status = engine.get_algo_status(result["algo_id"])
    assert status["is_done"]
    assert Decimal(status["parent_order"]["filled_qty"]) == Decimal("1.0")
    engine.close()


@pytest.mark.anyio
async def test_cancel_algo(engine):
    parent = ParentOrder(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        total_qty=Decimal("10.0"),
        algo_type="twap",
        params={"duration_seconds": 0, "buckets": 100, "tick_seconds": 0},
    )
    result = await engine.submit_algo(parent, mark_price=Decimal("50000"))
    algo_id = result["algo_id"]

    import asyncio
    await asyncio.sleep(0.05)
    cancel_result = await engine.cancel_algo(algo_id)
    assert cancel_result["status"] == "cancelled"

    status = engine.get_algo_status(algo_id)
    assert status["parent_order"]["state"] == "cancelled"
    engine.close()
