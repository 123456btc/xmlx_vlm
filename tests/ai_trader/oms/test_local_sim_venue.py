"""测试本地仿真机构盘与实盘地位相同."""

from decimal import Decimal

import pytest

from xmlx_vlm.ai_trader.oms.config.settings import OMSSettings
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.oms.execution.factory import ExecutionAdapterFactory
from xmlx_vlm.ai_trader.oms.execution.paper.adapter import PaperExecutionAdapter


def test_factory_creates_local_sim():
    adapter = ExecutionAdapterFactory.create(exchange="local_sim")
    assert isinstance(adapter, PaperExecutionAdapter)
    assert adapter.name == "paper"
    assert not adapter.is_live
    assert adapter.is_simulation


def test_factory_creates_paper():
    adapter = ExecutionAdapterFactory.create(exchange="paper")
    assert isinstance(adapter, PaperExecutionAdapter)
    assert adapter.is_simulation


def test_oms_engine_venue_type_for_paper():
    settings = OMSSettings(exchange="paper", live_enabled=False)
    engine = OMSEngine(settings=settings)
    assert engine.venue_type == "local_simulation"
    assert not engine.is_live
    engine.close()


@pytest.mark.anyio
async def test_paper_does_not_require_live_enabled():
    settings = OMSSettings(exchange="paper", live_enabled=False)
    engine = OMSEngine(settings=settings)
    order = engine.create_order("BTC/USDC", "buy", Decimal("0.01"), order_type="market")
    result = await engine.submit_order(order, mark_price=Decimal("50000"))
    assert result["status"] == "submitted"
    engine.close()
