"""测试 Brain."""

import time

import pytest

from xmlx_vlm.ai_trader.intelligence.brain import Brain, BrainConfig
from xmlx_vlm.ai_trader.intelligence.signal import Signal, SignalSeverity


def test_signal_debounce():
    brain = Brain(BrainConfig(signal_debounce_seconds=10))
    received = []
    brain.register_handler(lambda s: received.append(s))

    sig = Signal(type="news", symbol="BTC", severity=SignalSeverity.INFO, title="t", detail="d")
    brain.handle_signal(sig)
    brain.handle_signal(sig)
    assert len(received) == 1


def test_signal_debounce_expires():
    brain = Brain(BrainConfig(signal_debounce_seconds=0))
    received = []
    brain.register_handler(lambda s: received.append(s))

    sig = Signal(type="news", symbol="BTC", severity=SignalSeverity.INFO, title="t", detail="d")
    brain.handle_signal(sig)
    time.sleep(0.01)
    brain.handle_signal(sig)
    assert len(received) == 2


def test_classify_sentiment():
    brain = Brain()
    assert brain._classify_sentiment("Bitcoin surge to new ATH") == "bullish"
    assert brain._classify_sentiment("Market crash and sell-off") == "bearish"
    assert brain._classify_sentiment("Normal trading day") == "neutral"


@pytest.mark.anyio
async def test_brain_start_stop():
    brain = Brain(BrainConfig(enabled=True, news_scan_interval_seconds=1))
    await brain.start()
    assert brain._task is not None
    await brain.stop()
    assert brain._task is None or brain._task.done()
