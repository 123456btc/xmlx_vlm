"""测试 Iceberg 执行算法."""

from decimal import Decimal

import pytest

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState
from xmlx_vlm.ai_trader.oms.execution.algo.base import ParentOrder
from xmlx_vlm.ai_trader.oms.execution.algo.iceberg import IcebergAlgorithm
from xmlx_vlm.ai_trader.oms.execution.paper.adapter import PaperExecutionAdapter
from xmlx_vlm.ai_trader.oms.impact.market_impact import AlmgrenChrissImpactModel
from xmlx_vlm.ai_trader.oms.routing.router import SmartOrderRouter


@pytest.mark.anyio
async def test_iceberg_respects_display_qty():
    """验证 Iceberg 按 display_qty 拆分，并用 GTC 挂单."""
    from xmlx_vlm.ai_trader.oms.market_data.models import OrderBook, OrderBookLevel, Quote
    from xmlx_vlm.ai_trader.oms.market_data.provider import StaticMarketDataProvider

    provider = StaticMarketDataProvider(
        quotes={
            "BTC/USDC": Quote(
                symbol="BTC/USDC", bid=Decimal("49995"), ask=Decimal("50005"), mark=Decimal("50000")
            )
        },
        books={
            "BTC/USDC": OrderBook(
                symbol="BTC/USDC",
                bids=[OrderBookLevel(Decimal("49995"), Decimal("10"))],
                asks=[OrderBookLevel(Decimal("50005"), Decimal("10"))],
            )
        },
    )
    adapter = PaperExecutionAdapter(market_data_provider=provider, default_price=Decimal("50000"))
    model = AlmgrenChrissImpactModel()
    router = SmartOrderRouter(adapter, model, default_max_slippage_pct=Decimal("5.0"))

    parent = ParentOrder(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        total_qty=Decimal("1.0"),
        algo_type="iceberg",
        params={
            "display_qty": Decimal("0.2"),
            "time_in_force": "GTC",
            "max_retries": 5,
            "tick_seconds": 0,
        },
    )
    algo = IcebergAlgorithm()
    await algo.start(parent, router)

    # 纸盘 GTC 不自动成交，但应拆成 5 份并达到 max_retries 后停止
    assert len(parent.child_orders) == 5
    assert parent.filled_qty == Decimal("0")


@pytest.mark.anyio
async def test_iceberg_max_retries_stops():
    adapter = PaperExecutionAdapter(default_price=Decimal("50000"))
    model = AlmgrenChrissImpactModel()
    router = SmartOrderRouter(adapter, model, default_max_slippage_pct=Decimal("5.0"))

    parent = ParentOrder(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        total_qty=Decimal("1.0"),
        algo_type="iceberg",
        params={
            "display_qty": Decimal("0.2"),
            "time_in_force": "GTC",
            "max_retries": 3,
            "tick_seconds": 0,
        },
    )
    algo = IcebergAlgorithm()
    await algo.start(parent, router)

    assert not parent.is_done() or parent.state == OrderState.PARTIAL_FILLED
    assert len(parent.child_orders) <= 3
