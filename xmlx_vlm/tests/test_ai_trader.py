"""AI Trader 工具单元测试."""

from __future__ import annotations

import json

import pytest

from xmlx_vlm.ai_trader.cli import parse_tool_calls
from xmlx_vlm.ai_trader.tools.chart import ChartTool
from xmlx_vlm.ai_trader.tools.market import MarketDataTool
from xmlx_vlm.ai_trader.tools.registry import ToolRegistry
from xmlx_vlm.ai_trader.tools.trading import TradingTool


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
        assert "render_chart" in names
        assert "trading" in names

    def test_market_tool_ticker(self):
        registry = ToolRegistry()
        result = registry.execute(
            "market_data",
            {"action": "get_ticker", "symbol": "BTC/USDC", "exchange": "hyperliquid"},
        )
        assert "BTC/USDC" in result
        assert "last=" in result

    def test_market_tool_orderbook(self):
        registry = ToolRegistry()
        result = registry.execute(
            "market_data",
            {"action": "get_orderbook", "symbol": "ETH/USDC", "depth": 10},
        )
        assert "best_bid" in result
        assert "depth_imbalance" in result

    def test_market_tool_market_summary(self):
        registry = ToolRegistry()
        result = registry.execute(
            "market_data",
            {"action": "get_market_summary", "symbol": "ETH/USDC"},
        )
        assert "orderbook" in result
        assert "derivatives" in result

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
