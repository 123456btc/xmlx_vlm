"""Unit tests for quantitative strategy expert enhancements:
- MarketRegimeDetector (Regime Switching)
- PortfolioCorrelationRiskVerifier (Net Delta & Covariance)
- FundingRateCarryVerifier (Carry Cost)
- StructuralAnchorResolver (Price Anchoring)
- TradingTool Post-Only Maker Execution
"""

from decimal import Decimal
import time
import pytest

from xmlx_vlm.ai_trader.agent.decision import ActionType, TradeProposal
from xmlx_vlm.ai_trader.agent.verifier import (
    DeterministicProposalVerifier,
    FundingRateCarryVerifier,
    PortfolioCorrelationRiskVerifier,
    StructuralAnchorResolver,
)
from xmlx_vlm.ai_trader.market_service.models import OHLCV
from xmlx_vlm.ai_trader.market_service.regime import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeAnalysis,
)
from xmlx_vlm.ai_trader.sdk.client import TraderSDK
from xmlx_vlm.ai_trader.tools.trading import TradingTool


class TestMarketRegimeDetector:
    """测试市场状态识别与策略路由."""

    def _generate_candles(self, n: int = 50, trend: str = "bullish") -> list[OHLCV]:
        now_ms = int(time.time() * 1000)
        candles = []
        base = 60000.0
        for i in range(n):
            if trend == "bullish":
                base += 150.0 + (i % 3) * 10
            elif trend == "range":
                base += (100.0 if i % 2 == 0 else -100.0)
            elif trend == "panic":
                base += (1500.0 if i % 2 == 0 else -1500.0)

            candles.append(
                OHLCV(
                    timestamp_ms=now_ms + i * 3600 * 1000,
                    open=base - 50.0,
                    high=base + 100.0,
                    low=base - 100.0,
                    close=base + 20.0,
                    volume=1000.0 + i * 10.0,
                )
            )
        return candles

    def test_detect_bullish_trend(self):
        detector = MarketRegimeDetector()
        candles = self._generate_candles(n=50, trend="bullish")
        analysis = detector.detect_regime("BTC", candles)
        assert analysis.symbol == "BTC"
        assert analysis.regime in (MarketRegime.TREND_BULLISH, MarketRegime.RANGE_BOUND)
        assert analysis.suggested_strategy in ("trend_following", "grid_mean_reversion")

    def test_detect_panic_shock(self):
        detector = MarketRegimeDetector(panic_atr_multiplier=1.5)
        # 构造后半段剧烈暴涨暴跌的 K 线
        candles = self._generate_candles(n=40, trend="range")
        last_price = candles[-1].close
        now_ms = int(time.time() * 1000) + 40 * 3600 * 1000
        for i in range(10):
            candles.append(
                OHLCV(
                    timestamp_ms=now_ms + i * 3600 * 1000,
                    open=last_price,
                    high=last_price + 8000.0,
                    low=last_price - 8000.0,
                    close=last_price + (2000.0 if i % 2 == 0 else -2000.0),
                    volume=50000.0,
                )
            )
        analysis = detector.detect_regime("BTC", candles)
        assert analysis.regime == MarketRegime.HIGH_VOL_PANIC
        assert analysis.suggested_strategy == "defensive_cash"

    def test_sdk_get_market_regime(self):
        sdk = TraderSDK()
        res = sdk.market.get_market_regime("BTC")
        assert "regime" in res
        assert "suggested_strategy" in res


class TestPortfolioCorrelationRiskVerifier:
    """测试组合级相关性与总 Net Delta 敞口拦截."""

    def test_net_delta_within_limit(self):
        verifier = PortfolioCorrelationRiskVerifier(max_net_delta_multiplier=1.5)
        equity = Decimal("10000.0")  # 最大允许敞口 = $15,000
        existing = [
            {"symbol": "ETH", "side": "long", "size_usd": 4000.0, "beta": 1.1},
            {"symbol": "SOL", "side": "long", "size_usd": 3000.0, "beta": 1.2},
        ]  # 当前 Net Delta = 4000*1.1 + 3000*1.2 = 8000
        proposal = TradeProposal(
            action=ActionType.OPEN_LONG,
            symbol="AVAX",
            size_usd=Decimal("3000.0"),  # +3000 -> 11000 < 15000
            entry_price=Decimal("30.0"),
            stop_loss=Decimal("28.0"),
            take_profit=Decimal("35.0"),
        )
        res = verifier.verify(proposal, equity=equity, existing_positions=existing)
        assert res.passed is True
        assert res.metrics["projected_net_delta_usd"] == 11000.0

    def test_net_delta_breach_rejected(self):
        verifier = PortfolioCorrelationRiskVerifier(max_net_delta_multiplier=1.5)
        equity = Decimal("10000.0")  # 最大允许敞口 = $15,000
        existing = [
            {"symbol": "BTC", "side": "long", "size_usd": 10000.0, "beta": 1.0},
            {"symbol": "ETH", "side": "long", "size_usd": 4000.0, "beta": 1.0},
        ]  # 当前 Net Delta = 14,000
        proposal = TradeProposal(
            action=ActionType.OPEN_LONG,
            symbol="SOL",
            size_usd=Decimal("3000.0"),  # +3000 -> 17,000 > 15,000
            entry_price=Decimal("150.0"),
            stop_loss=Decimal("145.0"),
            take_profit=Decimal("165.0"),
        )
        res = verifier.verify(proposal, equity=equity, existing_positions=existing)
        assert res.passed is False
        assert any("总净敞口" in r for r in res.rejection_reasons)


class TestFundingRateCarryVerifier:
    """测试资金费率磨损折算."""

    def test_high_funding_rate_warning(self):
        verifier = FundingRateCarryVerifier(max_acceptable_daily_funding_loss_pct=0.002)
        proposal = TradeProposal(
            action=ActionType.OPEN_LONG,
            symbol="BTC",
            size_usd=Decimal("2000.0"),
            entry_price=Decimal("60000.0"),
            stop_loss=Decimal("59000.0"),
            take_profit=Decimal("63000.0"),
        )
        # 8h 费率 0.08% -> 24h = 0.24% > 0.2%
        res = verifier.verify(proposal, funding_rate=0.0008, estimated_holding_hours=24.0)
        assert res.passed is True
        assert len(res.warnings) > 0
        assert any("资金费率较高" in w for w in res.warnings)


class TestStructuralAnchorResolver:
    """测试结构化技术位解析."""

    def test_resolve_swing_levels(self):
        sl = StructuralAnchorResolver.resolve_anchor(
            anchor_name="swing_low_1h",
            mark_price=64000.0,
            atr=500.0,
            swing_low=63200.0,
            buffer_atr_mult=0.5,
        )
        # 63200 - 500*0.5 = 62950.0
        assert sl == 62950.0

    def test_resolve_volume_profile_vah_val(self):
        vp = {"vah": 65500.0, "val": 62500.0, "poc": 64100.0}
        tp = StructuralAnchorResolver.resolve_anchor(
            anchor_name="vah",
            mark_price=64000.0,
            atr=400.0,
            volume_profile=vp,
            buffer_atr_mult=0.5,
        )
        # 65500 + 400*0.5 = 65700.0
        assert tp == 65700.0


class TestTradingToolPostOnly:
    """测试 TradingTool Post-Only Maker 挂单."""

    def test_place_order_post_only(self):
        tool = TradingTool()
        res = tool.place_order(
            symbol="BTC",
            side="buy",
            qty=0.01,
            price=60000.0,
            mode="paper",
            post_only=True,
        )
        assert "PAPER" in res or "已提交" in res or "state" in res
