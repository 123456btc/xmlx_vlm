"""测试 HyperliquidOrderSyncWorker."""

from decimal import Decimal

import pytest

from xmlx_vlm.ai_trader.oms.config.settings import OMSSettings
from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.order_sync.hyperliquid_worker import HyperliquidOrderSyncWorker


class FakeHyperliquidAdapter:
    def __init__(self):
        self.name = "hyperliquid"
        self.is_live = True
        self._order_status = None
        self._fills = []

    def set_order_status(self, order):
        self._order_status = order

    def set_recent_fills(self, fills):
        self._fills = fills

    async def query_order(self, order_id: str):
        return self._order_status

    async def query_recent_fills(self, limit: int = 100):
        return self._fills


@pytest.fixture
def oms_with_order(tmp_path):
    settings = OMSSettings(
        exchange="paper",
        live_enabled=False,
        risk_profile="custom",
        max_orders_per_second=100,
        max_orders_per_minute=1000,
        audit_db_path=tmp_path / "audit.db",
    )
    oms = OMSEngine(settings=settings)
    order = Order(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        qty=Decimal("0.1"),
        order_type="limit",
        price=Decimal("60000"),
        client_order_id="cloid1",
    )
    order.order_id = "oid1"
    order.transition_to(OrderState.SUBMITTED)
    oms._orders["cloid1"] = order
    yield oms
    oms.close()


@pytest.mark.anyio
async def test_sync_updates_filled_order(oms_with_order):
    oms = oms_with_order
    adapter = FakeHyperliquidAdapter()
    updated = Order(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        qty=Decimal("0.1"),
        order_type="limit",
        price=Decimal("60000"),
        client_order_id="cloid1",
    )
    updated.order_id = "oid1"
    updated.filled_qty = Decimal("0.1")
    updated.remaining_qty = Decimal("0")
    updated.state = OrderState.FILLED
    adapter.set_order_status(updated)

    worker = HyperliquidOrderSyncWorker(adapter, oms, interval_seconds=1)
    result = await worker.sync_once()

    assert result.orders_checked == 1
    assert result.fills_applied == 1
    local = oms.get_order("cloid1")
    assert local.state == OrderState.FILLED
    assert local.filled_qty == Decimal("0.1")


@pytest.mark.anyio
async def test_sync_detects_cancelled_order(oms_with_order):
    oms = oms_with_order
    adapter = FakeHyperliquidAdapter()
    updated = Order(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        qty=Decimal("0.1"),
        order_type="limit",
        price=Decimal("60000"),
        client_order_id="cloid1",
    )
    updated.order_id = "oid1"
    updated.state = OrderState.CANCELLED
    adapter.set_order_status(updated)

    worker = HyperliquidOrderSyncWorker(adapter, oms, interval_seconds=1)
    result = await worker.sync_once()

    assert result.orders_updated == 1
    assert oms.get_order("cloid1").state == OrderState.CANCELLED
