"""测试 VWAP 执行算法."""

from decimal import Decimal

import pytest

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState
from xmlx_vlm.ai_trader.oms.execution.algo.base import ParentOrder
from xmlx_vlm.ai_trader.oms.execution.algo.vwap import VWAPAlgorithm
from xmlx_vlm.ai_trader.oms.execution.paper.adapter import PaperExecutionAdapter
from xmlx_vlm.ai_trader.oms.impact.market_impact import AlmgrenChrissImpactModel
from xmlx_vlm.ai_trader.oms.market_data.models import VolumeProfile
from xmlx_vlm.ai_trader.oms.market_data.provider import StaticMarketDataProvider
from xmlx_vlm.ai_trader.oms.routing.router import SmartOrderRouter


@pytest.mark.anyio
async def test_vwap_with_volume_profile():
    profile = VolumeProfile(
        symbol="BTC/USDC",
        total_volume=Decimal("100"),
        buckets=[Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40")],
    )
    provider = StaticMarketDataProvider(profiles={"BTC/USDC": profile})
    adapter = PaperExecutionAdapter(market_data_provider=provider, default_price=Decimal("50000"))
    model = AlmgrenChrissImpactModel()
    router = SmartOrderRouter(adapter, model, default_max_slippage_pct=Decimal("5.0"))

    parent = ParentOrder(
        symbol="BTC/USDC",
        side=OrderSide.SELL,
        total_qty=Decimal("1.0"),
        algo_type="vwap",
        params={"duration_seconds": 0, "buckets": 4, "tick_seconds": 0},
    )
    algo = VWAPAlgorithm()
    await algo.start(parent, router)

    assert parent.state == OrderState.FILLED
    assert parent.filled_qty == Decimal("1.0")
    assert len(parent.child_orders) == 4
