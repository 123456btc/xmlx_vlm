"""测试 TraderManager."""

import pytest

from xmlx_vlm.ai_trader.runtime import StrategyConfig, StrategyInstance, TraderManager


def test_register_and_list():
    manager = TraderManager()
    cfg = StrategyConfig(id="s1", name="Test", exchange="paper")
    manager.register(cfg)
    assert manager.list_ids() == ["s1"]
    assert manager.get("s1") is not None


def test_unregister():
    manager = TraderManager()
    cfg = StrategyConfig(id="s1", exchange="paper")
    manager.register(cfg)
    assert manager.unregister("s1") is True
    assert manager.unregister("s1") is False


@pytest.mark.anyio
async def test_start_stop_all():
    manager = TraderManager()
    cfg = StrategyConfig(id="s1", exchange="paper", scan_interval_seconds=10)
    manager.register(cfg)
    await manager.start_all()
    assert manager.get("s1").is_running
    await manager.stop_all()
    assert not manager.get("s1").is_running


def test_get_all_status():
    manager = TraderManager()
    manager.register(StrategyConfig(id="s1", name="A", exchange="paper"))
    manager.register(StrategyConfig(id="s2", name="B", exchange="paper"))
    status = manager.get_all_status()
    assert len(status) == 2
    assert {s["id"] for s in status} == {"s1", "s2"}
