"""MarketDataService 单元测试.

不依赖真实网络，只测内存状态机、事件总线、消息解析与指标计算。
"""

from __future__ import annotations

import time

import pytest

from xmlx_vlm.ai_trader.market_service.alerts import AlertConfig, AlertEngine
from xmlx_vlm.ai_trader.market_service.events import (
    BookUpdateEvent,
    EventBus,
    FundingUpdateEvent,
    IndicatorAlertEvent,
    PriceUpdateEvent,
    TradeEvent,
)
from xmlx_vlm.ai_trader.market_service.indicators import (
    adx,
    atr,
    cvd,
    ema,
    rsi,
    volume_profile,
)
from xmlx_vlm.ai_trader.market_service.market_info import fetch_top_volume_coins
from xmlx_vlm.ai_trader.market_service.models import (
    BookLevel,
    BookSnapshot,
    FundingRate,
    OHLCV,
    OISnapshot,
    Tick,
    Trade,
)
from xmlx_vlm.ai_trader.market_service.service import MarketDataService
from xmlx_vlm.ai_trader.market_service.state import MarketState, SymbolState
from xmlx_vlm.ai_trader.market_service.ws_client import HyperliquidMessageParser


class TestIndicators:
    def test_ema_basic(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = ema(values, 3)
        assert len(out) == len(values)
        assert out[-1] > out[0]

    def test_rsi_insufficient_data(self):
        assert rsi([1.0, 2.0], 14) == [50.0, 50.0]

    def test_atr_insufficient_data(self):
        ohlcv = [
            OHLCV(0, 100, 110, 90, 105, 1000),
            OHLCV(1, 105, 115, 95, 110, 1200),
        ]
        assert atr(ohlcv, 14) == []

    def test_adx_returns_lists(self):
        ohlcv = [
            OHLCV(i, 100 + i, 110 + i, 90 + i, 105 + i, 1000)
            for i in range(60)
        ]
        adx_values, plus_di, minus_di = adx(ohlcv, 14)
        assert len(adx_values) == len(ohlcv)
        assert len(plus_di) == len(ohlcv)
        assert len(minus_di) == len(ohlcv)

    def test_volume_profile(self):
        ohlcv = [
            OHLCV(i, 100, 110, 90, 100 + i, 1000 + i * 100)
            for i in range(50)
        ]
        vp = volume_profile(ohlcv, bins=12)
        assert "poc" in vp and "vah" in vp and "val" in vp
        assert vp["val"] <= vp["poc"] <= vp["vah"]

    def test_cvd(self):
        trades = [
            ("buy", 100.0, 1.0, 0),
            ("sell", 101.0, 1.0, 0),
            ("buy", 102.0, 2.0, 0),
        ]
        assert cvd(trades) == 100.0 - 101.0 + 204.0


class TestEventBus:
    def test_publish_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe(PriceUpdateEvent, lambda e: received.append(e))
        bus.subscribe(object, lambda e: received.append("wildcard"))

        event = PriceUpdateEvent(symbol="BTC", timestamp_ms=1, price=100.0)
        bus.publish(event)

        assert len(received) == 2
        assert received[0].price == 100.0
        assert received[1] == "wildcard"

    def test_handler_exception_does_not_break(self):
        bus = EventBus()
        received = []
        bus.subscribe(PriceUpdateEvent, lambda e: 1 / 0)
        bus.subscribe(PriceUpdateEvent, lambda e: received.append(e))
        bus.publish(PriceUpdateEvent(symbol="BTC", timestamp_ms=1, price=100.0))
        assert len(received) == 1


class TestSymbolState:
    def test_tick_updates_quote_and_bars(self):
        state = SymbolState("BTC")
        now = int(time.time() * 1000)
        state.update_tick(Tick(symbol="BTC", price=50000.0, timestamp_ms=now))
        quote = state.get_quote()
        assert quote is not None
        assert quote.bid == 50000.0
        ohlcv = state.get_ohlcv("1m", limit=10)
        assert len(ohlcv) == 1
        assert ohlcv[0].close == 50000.0

    def test_book_updates_quote(self):
        state = SymbolState("ETH")
        book = BookSnapshot(
            symbol="ETH",
            bids=[BookLevel(3000.0, 1.0), BookLevel(2999.0, 2.0)],
            asks=[BookLevel(3001.0, 1.5)],
            timestamp_ms=int(time.time() * 1000),
        )
        state.update_book(book)
        quote = state.get_quote()
        assert quote.bid == 3000.0
        assert quote.ask == 3001.0

    def test_trade_cvd_and_bars(self):
        state = SymbolState("BTC")
        now = int(time.time() * 1000)
        state.add_trade(Trade(symbol="BTC", side="buy", price=100.0, size=1.0, timestamp_ms=now))
        state.add_trade(Trade(symbol="BTC", side="sell", price=101.0, size=1.0, timestamp_ms=now + 1000))
        assert state.cvd_window(1) == 100.0 - 101.0

    def test_oi_delta(self):
        state = SymbolState("BTC")
        now = int(time.time() * 1000)
        state.add_oi(OISnapshot(symbol="BTC", open_interest=1000.0, mark_price=100.0, timestamp_ms=now - 61 * 60_000))
        state.add_oi(OISnapshot(symbol="BTC", open_interest=1100.0, mark_price=100.0, timestamp_ms=now))
        assert state.oi_delta_pct(60) == pytest.approx(10.0)


class TestMessageParser:
    def test_parse_all_mids(self):
        msg = {"channel": "allMids", "data": {"BTC": "65000.5", "ETH": "3500.2"}}
        mids = HyperliquidMessageParser.parse_all_mids(msg)
        assert mids["BTC"] == 65000.5
        assert mids["ETH"] == 3500.2

    def test_parse_l2_book(self):
        msg = {
            "channel": "l2Book",
            "data": {
                "coin": "BTC",
                "levels": [
                    [{"px": "64000", "sz": "1.5"}, {"px": "63900", "sz": "2.0"}],
                    [{"px": "64100", "sz": "1.0"}],
                ],
            },
        }
        book = HyperliquidMessageParser.parse_l2_book(msg)
        assert book is not None
        assert book.symbol == "BTC"
        assert book.bids[0].price == 64000.0
        assert book.asks[0].price == 64100.0

    def test_parse_trades(self):
        msg = {
            "channel": "trades",
            "data": {
                "coin": "BTC",
                "trades": [
                    {"px": "65000", "sz": "0.5", "side": "B", "time": 1},
                    {"px": "65001", "sz": "0.2", "side": "A", "time": 2},
                ],
            },
        }
        trades = HyperliquidMessageParser.parse_trades(msg)
        assert len(trades) == 2
        assert trades[0].side == "buy"
        assert trades[1].side == "sell"

    def test_parse_funding(self):
        msg = {"channel": "funding", "data": {"coin": "BTC", "fundingRate": "0.0001"}}
        funding = HyperliquidMessageParser.parse_funding(msg)
        assert funding is not None
        assert funding.symbol == "BTC"
        assert funding.rate == 0.0001


class TestMarketDataService:
    def test_service_state_updated_by_message(self):
        service = MarketDataService()
        # 不启动真实 WS，直接喂消息；先订阅币种
        service.subscribe("BTC")
        service._on_message({"channel": "allMids", "data": {"BTC": "70000.0"}})
        state = service.state.get("BTC")
        assert state.latest_tick is not None
        assert state.latest_tick.price == 70000.0

    def test_service_book_event(self):
        service = MarketDataService()
        received = []
        service.event_bus.subscribe(BookUpdateEvent, lambda e: received.append(e))
        service._on_message(
            {
                "channel": "l2Book",
                "data": {
                    "coin": "ETH",
                    "levels": [
                        [{"px": "3000", "sz": "1"}],
                        [{"px": "3001", "sz": "1"}],
                    ],
                },
            }
        )
        assert len(received) == 1
        assert received[0].book.symbol == "ETH"

    def test_trade_event(self):
        service = MarketDataService()
        received = []
        service.event_bus.subscribe(TradeEvent, lambda e: received.append(e))
        service._on_message(
            {
                "channel": "trades",
                "data": {
                    "coin": "BTC",
                    "trades": [{"px": "70000", "sz": "1", "side": "B", "time": 1}],
                },
            }
        )
        assert len(received) == 1


class TestMarketInfo:
    def test_fetch_top_volume_coins(self, monkeypatch):
        def fake_post(url, json, timeout):
            class Resp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return [
                        {
                            "universe": [
                                {"name": "BTC", "szDecimals": 4},
                                {"name": "ETH", "szDecimals": 4},
                                {"name": "SOL", "szDecimals": 2},
                            ]
                        },
                        [
                            {"dayNtlVlm": "5000000000"},
                            {"dayNtlVlm": "2000000000"},
                            {"dayNtlVlm": "1000000000"},
                        ],
                    ]

            return Resp()

        monkeypatch.setattr("requests.post", fake_post)
        top = fetch_top_volume_coins(n=2)
        assert top == ["BTC", "ETH"]

    def test_service_watched_coins(self):
        service = MarketDataService(top_n=30)
        service.subscribe("BTC")
        service.subscribe("ETH")
        assert service.get_watched_coins() == ["BTC", "ETH"]


class TestAlertEngine:
    def _make_state_with_bars(self, symbol: str, bars):
        from xmlx_vlm.ai_trader.market_service.state import SymbolState

        state = SymbolState(symbol)
        for b in bars:
            state.add_trade(
                Trade(
                    symbol=symbol,
                    side=b.get("side", "buy"),
                    price=b["close"],
                    size=b["volume"] / b["close"],
                    timestamp_ms=b["timestamp_ms"],
                )
            )
        return state

    def test_funding_flip_alert(self):
        bus = EventBus()
        received = []
        bus.subscribe(IndicatorAlertEvent, lambda e: received.append(e))
        state = MarketState()
        engine = AlertEngine(state, bus, AlertConfig(funding_flip_threshold=0.0))

        bus.publish(FundingUpdateEvent(symbol="BTC", timestamp_ms=1, funding=FundingRate(symbol="BTC", rate=-0.0001, timestamp_ms=1)))
        bus.publish(FundingUpdateEvent(symbol="BTC", timestamp_ms=2, funding=FundingRate(symbol="BTC", rate=0.0002, timestamp_ms=2)))

        assert any(a.alert_type == "funding_flip" for a in received)

    def test_book_imbalance_spike(self):
        bus = EventBus()
        received = []
        bus.subscribe(IndicatorAlertEvent, lambda e: received.append(e))
        state = MarketState()
        engine = AlertEngine(state, bus, AlertConfig(book_imbalance_threshold=0.5))

        book = BookSnapshot(
            symbol="BTC",
            bids=[BookLevel(100.0, 100.0)],
            asks=[BookLevel(101.0, 1.0)],
            timestamp_ms=1,
        )
        bus.publish(BookUpdateEvent(symbol="BTC", timestamp_ms=1, book=book))
        assert any(a.alert_type == "book_imbalance_spike" for a in received)

    def test_large_order_cluster(self):
        bus = EventBus()
        received = []
        bus.subscribe(IndicatorAlertEvent, lambda e: received.append(e))
        state = MarketState()
        cfg = AlertConfig(
            large_trade_notional=10_000,
            large_trade_window_sec=60,
            large_trade_cluster_count=2,
            large_trade_cluster_notional=25_000,
        )
        engine = AlertEngine(state, bus, cfg)
        now = int(time.time() * 1000)
        for _ in range(3):
            bus.publish(
                TradeEvent(
                    symbol="BTC",
                    timestamp_ms=now,
                    trade=Trade(symbol="BTC", side="buy", price=50000.0, size=1.0, timestamp_ms=now),
                )
            )
        assert any(a.alert_type == "large_order_cluster" for a in received)
