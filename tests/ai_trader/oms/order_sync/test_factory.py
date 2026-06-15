"""测试 order_sync 工厂."""

from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.oms.config.settings import OMSSettings
from xmlx_vlm.ai_trader.oms.order_sync.factory import create_order_sync_worker


def test_factory_returns_none_for_paper():
    settings = OMSSettings(exchange="paper", live_enabled=False)
    oms = OMSEngine(settings=settings)
    worker = create_order_sync_worker(oms.adapter, oms)
    assert worker is None
    oms.close()


def test_factory_returns_hyperliquid_worker():
    settings = OMSSettings(
        exchange="hyperliquid",
        live_enabled=True,
        wallet_address="0x1234567890123456789012345678901234567890",
        signer_endpoint="http://localhost:1",
    )
    oms = OMSEngine(settings=settings)
    worker = create_order_sync_worker(oms.adapter, oms)
    assert worker is not None
    assert worker.__class__.__name__ == "HyperliquidOrderSyncWorker"
    oms.close()
