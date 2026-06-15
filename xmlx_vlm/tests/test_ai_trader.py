"""AI Trader 工具单元测试."""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from xmlx_vlm.ai_trader.cli import parse_tool_calls
from xmlx_vlm.ai_trader.tools.chart import ChartTool
from xmlx_vlm.ai_trader.tools.market import (
    MarketDataTool,
    OHLCV,
    _adx,
    _atr,
    _format_notional,
    _structure,
    _volume_profile,
)
from xmlx_vlm.ai_trader.tools.registry import ToolRegistry
from xmlx_vlm.ai_trader.tools.trading import TradingTool
from xmlx_vlm.ai_trader.oms.execution.hyperliquid.adapter import (
    HyperliquidExecutionAdapter,
)
from xmlx_vlm.ai_trader.oms.execution.hyperliquid.client import HyperliquidClient
from xmlx_vlm.ai_trader.oms.execution.hyperliquid.signer import (
    ExternalSigner,
    create_signer,
)


class TestParseToolCalls:
    def test_parse_single_tool_call(self):
        text = (
            '我先看看行情。<tool_call>{"name": "market_data", '
            '"arguments": {"action": "get_ticker", "symbol": "BTC/USDT", "exchange": "okx"}}</tool_call>'
        )
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "market_data"
        assert calls[0]["arguments"]["symbol"] == "BTC/USDT"

    def test_parse_xml_tool_call(self):
        text = """我先看看行情。
<tool_call>
<function=market_data>
<parameter=action>get_ticker</parameter>
<parameter=symbol>BTC/USDT</parameter>
<parameter=exchange>okx</parameter>
</function>
</tool_call>"""
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "market_data"
        assert calls[0]["arguments"]["action"] == "get_ticker"

    def test_parse_no_tool_call(self):
        assert parse_tool_calls("今天天气不错") == []

    def test_parse_call_brace_tool_call(self):
        text = "我来查一下价格。\n<|tool_call>call:market_data{action:get_ticker,symbol:BTC/USDT,exchange:okx}<tool_call|>"
        calls = parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "market_data"
        assert calls[0]["arguments"]["action"] == "get_ticker"
        assert calls[0]["arguments"]["symbol"] == "BTC/USDT"
        assert calls[0]["arguments"]["exchange"] == "okx"


class TestToolRegistry:
    def test_registry_lists_all_tools(self):
        registry = ToolRegistry()
        names = [t["function"]["name"] for t in registry.list_tools()]
        assert "market_data" in names
        assert "trading" in names
        # ChartTool is intentionally excluded: the model uses structured
        # kline+L2 feature data, not visual chart images.
        assert "render_chart" not in names

    @pytest.mark.skip(reason="需要网络连接，CI 中可选")
    def test_market_tool_ticker(self):
        registry = ToolRegistry()
        result = registry.execute(
            "market_data",
            {"action": "get_ticker", "symbol": "BTC/USDC", "exchange": "hyperliquid"},
        )
        assert "BTC/USDC" in result
        assert "mark=" in result

    @pytest.mark.skip(reason="需要网络连接，CI 中可选")
    def test_market_tool_orderbook(self):
        registry = ToolRegistry()
        result = registry.execute(
            "market_data",
            {"action": "get_orderbook", "symbol": "ETH/USDC", "depth": 10},
        )
        assert "best_bid" in result
        assert "depth_imbalance" in result

    @pytest.mark.skip(reason="需要网络连接，CI 中可选")
    def test_market_tool_market_summary(self):
        registry = ToolRegistry()
        result = registry.execute(
            "market_data",
            {"action": "get_market_summary", "symbol": "ETH/USDC"},
        )
        assert "orderbook" in result
        assert "derivatives" in result
        assert "atr14" in result
        assert "adx14" in result
        assert "poc" in result
        assert "cvd_1h" in result
        assert "basis_pct" in result

    @pytest.mark.skip(reason="需要网络连接，CI 中可选")
    def test_market_tool_multi_timeframe(self):
        registry = ToolRegistry()
        result = registry.execute(
            "market_data",
            {"action": "get_multi_timeframe_summary", "symbol": "ETH/USDC"},
        )
        assert "5m" in result
        assert "15m" in result
        assert "1h" in result
        assert "aggregated_signal" in result
        assert "trend_strength" in result
        assert "avg_adx" in result

    def test_market_tool_open_interest_fields(self, monkeypatch):
        tool = MarketDataTool()

        def fake_hl_post(payload):
            if payload.get("type") == "metaAndAssetCtxs":
                return (
                    {"universe": [{"name": "ETH", "szDecimals": 4}]},
                    [
                        {
                            "markPx": "3500.5",
                            "midPx": "3500.3",
                            "oraclePx": "3500.0",
                            "prevDayPx": "3400.0",
                            "openInterest": "12000.5",
                            "funding": "0.0001",
                            "dayNtlVlm": "500000000",
                            "impactPxs": ["3499.8", "3500.8"],
                        }
                    ],
                )
            raise ValueError(f"unexpected payload {payload}")

        monkeypatch.setattr(tool, "_hl_post", fake_hl_post)
        result = tool.get_open_interest("ETH/USDC")
        assert "oi_change_1h_pct" in result
        assert "oi_change_24h_pct" in result
        assert "open_interest_notional_fmt" in result
        data = json.loads(result)
        assert data["open_interest_notional_fmt"].endswith(("M", "K", "B"))

    @pytest.mark.skip(reason="需要网络连接，CI 中可选")
    def test_trading_tool_paper_order(self):
        tool = TradingTool()
        result = tool.run(action="place_order", symbol="BTC/USDT", side="buy", qty=0.01)
        assert "PAPER" in result
        assert "buy" in result

        positions = tool.run(action="get_positions")
        assert "BTC/USDT" in positions

        close = tool.run(action="close_position", symbol="BTC/USDT")
        assert "已平仓" in close


@pytest.mark.skip(reason="需要网络连接，CI 中可选")
class TestHyperliquidAPI:
    def test_get_ticker_hyperliquid(self):
        tool = MarketDataTool()
        text = tool.get_ticker("BTC/USDC", "hyperliquid")
        assert "BTC/USDC" in text

    def test_get_ohlcv_hyperliquid(self):
        tool = MarketDataTool()
        data = tool.get_ohlcv("BTC/USDC", "hyperliquid", "1h", 10)
        assert len(data) > 0


@pytest.mark.skip(reason="需要网络连接，CI 中可选")
class TestChartTool:
    def test_render_chart(self):
        tool = ChartTool()
        result = tool.render("BTC/USDC", "hyperliquid", "1h", 30)
        assert result.ohlcv_count > 0
        assert result.image_path.endswith(".png")


class TestMarketDataHelpers:
    def test_format_notional(self):
        assert _format_notional(1_240_000_000) == "1.24B"
        assert _format_notional(12_400_000) == "12.40M"
        assert _format_notional(850_000) == "850.00K"
        assert _format_notional(1_200) == "1.20K"
        assert _format_notional(500) == "500.00"

    def test_atr(self):
        ohlcv = [
            OHLCV(0, 100, 110, 90, 105, 1000),
            OHLCV(1, 105, 115, 95, 110, 1200),
            OHLCV(2, 110, 120, 100, 115, 1500),
        ]
        # 数据不足 period+1 时返回空
        assert _atr(ohlcv, 14) == []

    def test_adx_returns_lists(self):
        ohlcv = [
            OHLCV(i, 100 + i, 110 + i, 90 + i, 105 + i, 1000)
            for i in range(60)
        ]
        adx, plus_di, minus_di = _adx(ohlcv, 14)
        assert len(adx) == len(ohlcv)
        assert len(plus_di) == len(ohlcv)
        assert len(minus_di) == len(ohlcv)

    def test_volume_profile(self):
        ohlcv = [
            OHLCV(i, 100, 110, 90, 100 + i, 1000 + i * 100)
            for i in range(50)
        ]
        vp = _volume_profile(ohlcv, bins=12)
        assert "poc" in vp
        assert "vah" in vp
        assert "val" in vp
        assert vp["val"] <= vp["poc"] <= vp["vah"]

    def test_structure(self):
        up = [
            OHLCV(i, i, i + 2, i, i + 1, 1)
            for i in range(25)
        ]
        assert _structure(up, lookback=20) == "higher_highs_higher_lows"
        down = [
            OHLCV(i, 25 - i, 27 - i, 23 - i, 24 - i, 1)
            for i in range(25)
        ]
        assert _structure(down, lookback=20) == "lower_highs_lower_lows"


class TestExternalSigner:
    """外部签名器 mock 测试：验证 OMS 不触碰私钥即可完成签名流程."""

    def test_external_signer_posts_action_and_returns_signature(self):
        signer = ExternalSigner(
            wallet_address="0x1234567890abcdef1234567890abcdef12345678",
            signer_endpoint="http://localhost:8000/sign",
        )
        action = {"type": "order", "orders": [{"coin": "BTC", "sz": "0.1"}]}
        timestamp_ms = 1_700_000_000_000

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "signature": "0xdeadbeef",
            "vaultAddress": "0xvault",
        }

        with patch("xmlx_vlm.ai_trader.oms.execution.hyperliquid.signer.requests.post", return_value=mock_resp) as mock_post:
            payload = signer.sign(action, timestamp_ms)

        mock_post.assert_called_once_with(
            "http://localhost:8000/sign",
            json={
                "wallet_address": "0x1234567890abcdef1234567890abcdef12345678",
                "action": action,
                "nonce": timestamp_ms,
            },
            timeout=10,
        )
        assert payload["action"] == action
        assert payload["nonce"] == timestamp_ms
        assert payload["signature"] == "0xdeadbeef"
        assert payload["vaultAddress"] == "0xvault"

    def test_create_signer_prefers_external_endpoint_over_private_key(self, monkeypatch):
        monkeypatch.setenv("HL_API_WALLET_ADDRESS", "0xwallet")
        monkeypatch.setenv("HL_SIGNER_ENDPOINT", "http://signer/sign")
        monkeypatch.setenv("HL_API_PRIVATE_KEY", "0xsecret")

        signer = create_signer()
        assert isinstance(signer, ExternalSigner)
        assert signer.signer_endpoint == "http://signer/sign"
        assert not hasattr(signer, "private_key")


class TestHyperliquidAccountAbstraction:
    """验证 Hyperliquid 适配器自动识别 standard / unified / portfolio margin."""

    @pytest.fixture
    def base_ch_state(self):
        return {
            "marginSummary": {
                "accountValue": "12000.0",
                "totalNtlPos": "2000.0",
                "totalRawUsd": "10000.0",
                "totalMarginUsed": "500.0",
            },
            "withdrawable": "11500.0",
            "assetPositions": [],
        }

    @pytest.fixture
    def unified_spot_state(self):
        return {
            "balances": [
                {"coin": "USDC", "token": 0, "total": "15000.0", "hold": "2000.0"},
                {"coin": "HYPE", "token": 1, "total": "100.0", "hold": "0.0"},
            ]
        }

    def test_unified_account_uses_spot_usdc_for_equity(
        self, monkeypatch, base_ch_state, unified_spot_state
    ):
        """unified 模式下 sync_account 的权益与可用保证金应来自 spot USDC."""
        wallet = "0x1234567890abcdef1234567890abcdef12345678"
        monkeypatch.setenv("HL_API_WALLET_ADDRESS", wallet)
        monkeypatch.setattr(
            HyperliquidClient, "get_user_abstraction", lambda self, addr: "unifiedAccount"
        )

        adapter = HyperliquidExecutionAdapter(signer_endpoint="http://signer/sign")

        def fake_info(payload):
            if payload.get("type") == "clearinghouseState":
                return base_ch_state
            if payload.get("type") == "spotClearinghouseState":
                return unified_spot_state
            raise ValueError(f"unexpected payload {payload}")

        monkeypatch.setattr(adapter._client, "info", fake_info)

        account = asyncio.run(adapter.sync_account())
        assert account.mode == "unifiedAccount"
        assert account.equity == Decimal("15000.0")
        assert account.available_margin == Decimal("13000.0")
        assert account.cash == Decimal("13000.0")
        assert account.used_margin == Decimal("500.0")

    def test_standard_account_uses_clearinghouse_state(self, monkeypatch, base_ch_state):
        """standard / disabled 模式下 sync_account 应保持原有 clearinghouseState 行为."""
        wallet = "0x1234567890abcdef1234567890abcdef12345678"
        monkeypatch.setenv("HL_API_WALLET_ADDRESS", wallet)
        monkeypatch.setattr(
            HyperliquidClient, "get_user_abstraction", lambda self, addr: "disabled"
        )

        adapter = HyperliquidExecutionAdapter(signer_endpoint="http://signer/sign")

        def fake_info(payload):
            if payload.get("type") == "clearinghouseState":
                return base_ch_state
            raise ValueError(f"unexpected payload {payload}")

        monkeypatch.setattr(adapter._client, "info", fake_info)

        account = asyncio.run(adapter.sync_account())
        assert account.mode == "disabled"
        assert account.equity == Decimal("12000.0")
        assert account.available_margin == Decimal("11500.0")
        assert account.cash == Decimal("11500.0")
        assert account.used_margin == Decimal("500.0")

    def test_adapter_exposes_account_mode(self, monkeypatch):
        wallet = "0x1234567890abcdef1234567890abcdef12345678"
        monkeypatch.setenv("HL_API_WALLET_ADDRESS", wallet)
        monkeypatch.setattr(
            HyperliquidClient,
            "get_user_abstraction",
            lambda self, addr: "portfolioMargin",
        )
        adapter = HyperliquidExecutionAdapter(signer_endpoint="http://signer/sign")
        assert adapter.account_mode == "portfolioMargin"


class TestWebSearchTools:
    def test_web_search_tavily(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "test_tavily_key")
        
        class FakeResponse:
            def raise_for_status(self):
                pass
            def json(self):
                return {
                    "results": [
                        {"title": "Bitcoin Price Soars", "url": "https://crypto.news/btc", "content": "Bitcoin broke $100k today."}
                    ]
                }
        
        def fake_post(url, json, timeout):
            assert "api.tavily.com/search" in url
            assert json["api_key"] == "test_tavily_key"
            assert json["query"] == "bitcoin"
            return FakeResponse()

        monkeypatch.setattr("requests.post", fake_post)

        registry = ToolRegistry()
        result_str = registry.execute("web_search", {"query": "bitcoin"})
        result = json.loads(result_str)
        assert result["success"] is True
        assert len(result["data"]["web"]) == 1
        assert result["data"]["web"][0]["title"] == "Bitcoin Price Soars"
        assert result["data"]["web"][0]["url"] == "https://crypto.news/btc"

    def test_web_search_ddg(self, monkeypatch):
        # Unset TAVILY and BRAVE API keys to force DDG fallback
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)

        class FakeDDGS:
            def __init__(self):
                pass
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
            def text(self, query, max_results):
                assert query == "ethereum"
                return [
                    {"title": "Ethereum upgrade", "href": "https://eth.news/up", "body": "Ethereum completes upgrade."}
                ]

        monkeypatch.setattr("duckduckgo_search.DDGS", FakeDDGS)

        registry = ToolRegistry()
        result_str = registry.execute("web_search", {"query": "ethereum"})
        result = json.loads(result_str)
        assert result["success"] is True
        assert len(result["data"]["web"]) == 1
        assert result["data"]["web"][0]["title"] == "Ethereum upgrade"
        assert result["data"]["web"][0]["url"] == "https://eth.news/up"

    def test_web_extract_scraper_fallback(self, monkeypatch):
        # Force scraper fallback by unsetting keys
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)

        class FakeResponse:
            def raise_for_status(self):
                pass
            @property
            def text(self):
                return "<html><head><title>Test Page</title></head><body><h1>Hello World</h1><p>This is a paragraph.</p></body></html>"

        def fake_get(url, headers, timeout):
            assert url == "https://test.com/page"
            return FakeResponse()

        monkeypatch.setattr("requests.get", fake_get)

        registry = ToolRegistry()
        result_str = registry.execute("web_extract", {"url": "https://test.com/page"})
        result = json.loads(result_str)
        assert result["success"] is True
        assert "Test Page" in result["data"]["title"]
        assert "Hello World" in result["data"]["content"]
        assert "This is a paragraph." in result["data"]["content"]

