"""测试纸盘适配器."""

import asyncio
from decimal import Decimal

import pytest

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.execution.paper.adapter import PaperExecutionAdapter


@pytest.fixture
def adapter():
    return PaperExecutionAdapter(market_data_tool=None, fill_slippage_pct=Decimal("0"))


def test_paper_adapter_is_not_live(adapter):
    assert not adapter.is_live


@pytest.mark.anyio
async def test_paper_submit_market(adapter):
    order = Order(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        qty=Decimal("0.1"),
        price=Decimal("50000"),
    )
    ack = await adapter.submit(order)
    assert ack.success
    assert order.state == OrderState.FILLED


@pytest.mark.anyio
async def test_paper_sync_account(adapter):
    account = await adapter.sync_account()
    assert account.equity == Decimal("100000")
