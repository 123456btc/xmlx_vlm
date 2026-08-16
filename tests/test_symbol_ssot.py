"""Unit tests for Symbol SSOT (Single Source of Truth) architecture."""

import pytest
from decimal import Decimal
from unittest.mock import patch, AsyncMock

from xmlx_vlm.ai_trader.oms.utils.symbol import (
    normalize_symbol,
    extract_base_coin,
    symbol_matches,
    parse_symbol_parts,
)
from xmlx_vlm.ai_trader.oms.core.order import Order, Fill
from xmlx_vlm.ai_trader.oms.core.position import Position
from xmlx_vlm.ai_trader.oms.core.trade import Trade
from xmlx_vlm.ai_trader.oms.core.portfolio import Portfolio
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.oms.config.settings import get_settings
from xmlx_vlm.ai_trader.oms.constants import OrderSide, PositionSide
from xmlx_vlm.ai_trader.oms.execution.hyperliquid.mapper import coin_from_symbol
from xmlx_vlm.ai_trader.tools.market import MarketDataTool


class TestSymbolSSOT:
    """Test suite for Symbol SSOT utilities and integration."""

    def test_normalize_symbol_standard(self):
        assert normalize_symbol("BTC") == "BTC/USDC"
        assert normalize_symbol("btc") == "BTC/USDC"
        assert normalize_symbol("BTC/USDC") == "BTC/USDC"
        assert normalize_symbol("btc/usdc") == "BTC/USDC"
        assert normalize_symbol("ETH-USDT") == "ETH/USDT"
        assert normalize_symbol("SOL_USD") == "SOL/USD"
        assert normalize_symbol("BTCUSDT") == "BTC/USDT"
        assert normalize_symbol("ETHUSDC") == "ETH/USDC"

    def test_normalize_symbol_k_prefix(self):
        # All variations of k-prefix tokens must normalize with lowercase 'k'
        assert normalize_symbol("kSHIB") == "kSHIB/USDC"
        assert normalize_symbol("KSHIB") == "kSHIB/USDC"
        assert normalize_symbol("kshib") == "kSHIB/USDC"
        assert normalize_symbol("kSHIB/USDC") == "kSHIB/USDC"
        assert normalize_symbol("KSHIB/USDC") == "kSHIB/USDC"
        assert normalize_symbol("kBONK") == "kBONK/USDC"
        assert normalize_symbol("KBONK/USDC") == "kBONK/USDC"
        assert normalize_symbol("kLUNC/USDC") == "kLUNC/USDC"
        assert normalize_symbol("KLUNC/USDC") == "kLUNC/USDC"
        assert normalize_symbol("kFLOKI/USDC") == "kFLOKI/USDC"
        assert normalize_symbol("KFLOKI/USDC") == "kFLOKI/USDC"
        assert normalize_symbol("kPEPE") == "kPEPE/USDC"
        assert normalize_symbol("kNEIRO") == "kNEIRO/USDC"

    def test_extract_base_coin(self):
        assert extract_base_coin("BTC/USDC") == "BTC"
        assert extract_base_coin("btc/usdc") == "BTC"
        assert extract_base_coin("BTC") == "BTC"
        assert extract_base_coin("kSHIB/USDC") == "kSHIB"
        assert extract_base_coin("KSHIB/USDC") == "kSHIB"
        assert extract_base_coin("kshib") == "kSHIB"
        assert extract_base_coin("kBONK/USDT") == "kBONK"
        assert extract_base_coin("KBONK") == "kBONK"
        assert extract_base_coin("kLUNC/USD") == "kLUNC"
        assert extract_base_coin("kFLOKIUSDC") == "kFLOKI"

    def test_symbol_matches(self):
        assert symbol_matches("BTC", "BTC/USDC") is True
        assert symbol_matches("btc", "BTC/USDC") is True
        assert symbol_matches("BTC/USDC", "BTC") is True
        assert symbol_matches("kSHIB", "kSHIB/USDC") is True
        assert symbol_matches("kSHIB", "KSHIB/USDC") is True
        assert symbol_matches("KSHIB", "kSHIB") is True
        assert symbol_matches("BTC/USDC", "ETH/USDC") is False
        assert symbol_matches("BTC/USDC", "BTC/USDT") is False

    def test_core_entities_auto_normalize_symbol(self):
        # Order auto-normalizes symbol
        o = Order(symbol="kSHIB", side=OrderSide.BUY, qty=Decimal("100"))
        assert o.symbol == "kSHIB/USDC"

        o2 = Order(symbol="KSHIB/USDC", side=OrderSide.BUY, qty=Decimal("100"))
        assert o2.symbol == "kSHIB/USDC"

        # Position auto-normalizes symbol
        p = Position(symbol="kshib", side=PositionSide.LONG, qty=Decimal("50"))
        assert p.symbol == "kSHIB/USDC"

        # Trade auto-normalizes symbol
        t = Trade(
            trade_id="t1",
            order_id="o1",
            client_order_id="c1",
            symbol="KBONK",
            side=OrderSide.BUY,
            qty=Decimal("10"),
            price=Decimal("1"),
        )
        assert t.symbol == "kBONK/USDC"

        # Fill auto-normalizes symbol
        f = Fill(
            fill_id="f1",
            order_id="o1",
            symbol="btc",
            side=OrderSide.BUY,
            qty=Decimal("1"),
            price=Decimal("50000"),
        )
        assert f.symbol == "BTC/USDC"

    def test_portfolio_get_position_matching(self):
        portfolio = Portfolio()
        # Insert canonical positions
        pos_btc = Position(
            symbol="BTC/USDC",
            side=PositionSide.LONG,
            qty=Decimal("1.5"),
            avg_entry_price=Decimal("60000"),
            mark_price=Decimal("60000"),
        )
        pos_shib = Position(
            symbol="kSHIB/USDC",
            side=PositionSide.LONG,
            qty=Decimal("1000000"),
            avg_entry_price=Decimal("0.02"),
            mark_price=Decimal("0.02"),
        )

        portfolio.sync_positions({
            "BTC/USDC": pos_btc,
            "kSHIB/USDC": pos_shib,
        })

        # Exact match
        assert portfolio.get_position("BTC/USDC") is pos_btc
        assert portfolio.get_position("kSHIB/USDC") is pos_shib

        # Bare coin match
        assert portfolio.get_position("BTC") is pos_btc
        assert portfolio.get_position("btc") is pos_btc
        assert portfolio.get_position("kSHIB") is pos_shib
        assert portfolio.get_position("KSHIB") is pos_shib
        assert portfolio.get_position("KSHIB/USDC") is pos_shib

        # Non-existent
        assert portfolio.get_position("ETH") is None
        assert portfolio.get_position("kBONK") is None

        # position_notional
        assert portfolio.position_notional("BTC") == Decimal("1.5") * Decimal("60000")
        assert portfolio.position_notional("kSHIB") == Decimal("1000000") * Decimal("0.02")

    def test_mapper_coin_from_symbol(self):
        assert coin_from_symbol("kSHIB/USDC") == "kSHIB"
        assert coin_from_symbol("KSHIB/USDC") == "kSHIB"
        assert coin_from_symbol("kBONK/USDC") == "kBONK"
        assert coin_from_symbol("KBONK") == "kBONK"
        assert coin_from_symbol("BTC/USDC") == "BTC"
        assert coin_from_symbol("ETH/USDC") == "ETH"

    def test_market_tool_coin_extraction(self):
        tool = MarketDataTool()
        assert tool._coin("kSHIB/USDC") == "kSHIB"
        assert tool._coin("KSHIB/USDC") == "kSHIB"
        assert tool._coin("kBONK") == "kBONK"
        assert tool._coin("BTC/USDC") == "BTC"
        assert tool._coin("ETHUSDT") == "ETH"

    @pytest.mark.asyncio
    async def test_oms_engine_close_position_with_various_symbols(self):
        settings = get_settings()
        engine = OMSEngine(settings=settings)
        # Add position to portfolio
        pos = Position(
            symbol="kSHIB/USDC",
            side=PositionSide.LONG,
            qty=Decimal("10000"),
            avg_entry_price=Decimal("0.02"),
            mark_price=Decimal("0.02"),
        )
        engine.portfolio.sync_positions({"kSHIB/USDC": pos})

        submitted = []

        async def fake_submit(order, mark_price=None, oracle_price=None):
            submitted.append(order)
            return {"status": "ok", "order": order.to_dict()}

        with patch.object(engine, "sync", new_callable=AsyncMock), \
             patch.object(engine, "submit_order", side_effect=fake_submit):

            # Close using bare coin "kSHIB"
            order = await engine.close_position("kSHIB")
            assert order is not None
            assert order.symbol == "kSHIB/USDC"
            assert order.side == OrderSide.SELL
            assert order.reduce_only is True
            assert order.qty == Decimal("10000")

            # Close with uppercase "KSHIB"
            engine.portfolio.sync_positions({"kSHIB/USDC": pos})
            order2 = await engine.close_position("KSHIB")
            assert order2 is not None
            assert order2.symbol == "kSHIB/USDC"
            assert order2.side == OrderSide.SELL
            assert order2.reduce_only is True
