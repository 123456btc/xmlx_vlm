"""测试 TWAP 执行算法."""

from decimal import Decimal

import pytest

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.execution.algo.base import ParentOrder
from xmlx_vlm.ai_trader.oms.execution.algo.twap import TWAPAlgorithm
from xmlx_vlm.ai_trader.oms.execution.paper.adapter import PaperExecutionAdapter
from xmlx_vlm.ai_trader.oms.impact.market_impact import AlmgrenChrissImpactModel
from xmlx_vlm.ai_trader.oms.routing.router import SmartOrderRouter


@pytest.fixture
def router():
    adapter = PaperExecutionAdapter(default_price=Decimal("50000"))
    model = AlmgrenChrissImpactModel()
    return SmartOrderRouter(adapter, model, default_max_slippage_pct=Decimal("5.0"))


@pytest.mark.anyio
async def test_twap_splits_total_qty(router):
    parent = ParentOrder(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        total_qty=Decimal("1.0"),
        algo_type="twap",
        params={"duration_seconds": 0, "buckets": 5, "leftover_mode": "carry", "tick_seconds": 0},
    )
    algo = TWAPAlgorithm()
    await algo.start(parent, router)

    assert parent.state == OrderState.FILLED
    assert parent.filled_qty == Decimal("1.0")
    assert len(parent.child_orders) <= 5


@pytest.mark.anyio
async def test_twap_respects_cancel(router):
    parent = ParentOrder(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        total_qty=Decimal("10.0"),
        algo_type="twap",
        params={"duration_seconds": 0, "buckets": 10, "tick_seconds": 0},
    )
    algo = TWAPAlgorithm()
    algo.cancel()
    await algo.start(parent, router)

    assert parent.state == OrderState.CANCELLED
