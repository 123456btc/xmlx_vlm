"""测试 Hyperliquid adapter 行情接口."""

from decimal import Decimal

import pytest

from xmlx_vlm.ai_trader.oms.execution.hyperliquid.adapter import HyperliquidExecutionAdapter


class FakeHLClient:
    def __init__(self):
        self.meta = {
            "universe": [
                {"name": "BTC", "szDecimals": 5},
                {"name": "ETH", "szDecimals": 4},
            ]
        }
        self.ctxs = [
            {
                "funding": "0.00001",
                "openInterest": "1000",
                "prevDayPx": "64000",
                "dayNtlVlm": "1000000000",
                "premium": "0",
                "oraclePx": "64300",
                "markPx": "64280",
                "midPx": "64275",
                "impactPxs": ["64270", "64280"],
                "dayBaseVlm": "15000",
            },
            {
                "markPx": "3500",
                "midPx": "3500",
                "impactPxs": ["3499", "3501"],
                "dayBaseVlm": "50000",
            },
        ]

    def get_meta_and_asset_ctxs(self):
        return [self.meta, self.ctxs]

    def get_candles(self, coin, interval, start_time_ms, end_time_ms):
        base = Decimal("64000")
        candles = []
        for i in range(24):
            candles.append({
                "t": start_time_ms + i * 3600 * 1000,
                "o": str(base),
                "c": str(base + Decimal(str(i * 10))),
                "h": str(base + Decimal(str(i * 10 + 50))),
                "l": str(base),
                "v": str(Decimal("100") + i),
            })
        return candles


@pytest.fixture
def adapter():
    class DummySigner:
        wallet_address = "0x123"

    a = HyperliquidExecutionAdapter.__new__(HyperliquidExecutionAdapter)
    a._client = FakeHLClient()
    a._signer = DummySigner()
    a._wallet_address = "0x123"
    a._account_mode = "disabled"
    return a


@pytest.mark.anyio
async def test_get_quote(adapter):
    quote = await adapter.get_quote("BTC/USDC")
    assert quote is not None
    assert quote.bid == Decimal("64270")
    assert quote.ask == Decimal("64280")
    assert quote.mark == Decimal("64280")
    assert quote.symbol == "BTC/USDC"


@pytest.mark.anyio
async def test_get_order_book(adapter):
    book = await adapter.get_order_book("BTC/USDC")
    assert book is not None
    assert book.best_bid() == Decimal("64270")
    assert book.best_ask() == Decimal("64280")


@pytest.mark.anyio
async def test_get_recent_volume(adapter):
    vol = await adapter.get_recent_volume("BTC/USDC", window_seconds=3600)
    assert vol is not None
    # dayBaseVlm 15000 / 24
    assert vol == Decimal("15000") / Decimal("24")


@pytest.mark.anyio
async def test_get_volume_profile(adapter):
    profile = await adapter.get_volume_profile("BTC/USDC", duration_seconds=86400, buckets=24)
    assert profile is not None
    assert profile.total_volume > Decimal("0")
    assert len(profile.buckets) == 24


@pytest.mark.anyio
async def test_get_volatility(adapter):
    vol = await adapter.get_volatility("BTC/USDC", window_days=30)
    assert vol is not None
    assert vol > Decimal("0")
