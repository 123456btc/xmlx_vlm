"""Unit tests for the 6 battle-tested Crypto Quantitative Factors."""

import pytest
from xmlx_vlm.ai_trader.market_service.models import OHLCV, Bar, Tick, Trade, FundingRate, OISnapshot
from xmlx_vlm.ai_trader.market_service.state import SymbolState
from xmlx_vlm.ai_trader.market_service.indicators import (
    pinbar_liquidity_sweep,
    cvd_price_divergence,
    oi_price_regime,
    bollinger_bands,
    bollinger_squeeze,
    funding_rate_zscore,
    candle_efficiency,
)


class TestCryptoFactors:
    """Test suite for the 6 core crypto quantitative factors."""

    def test_pinbar_liquidity_sweep(self):
        # 1. Bullish sweep: Long lower wick + high volume (2.5x MA)
        candles = [
            OHLCV(timestamp_ms=i * 60000, open=100.0, high=102.0, low=99.0, close=101.0, volume=1000.0)
            for i in range(20)
        ]
        # Append extreme lower wick pin-bar with high volume (3000.0 volume, low=90.0, open=99.0, close=100.0, high=101.0)
        # range = 11.0, lower wick = (99 - 90) / 11 = 9/11 = 0.818 (> 0.60)
        candles.append(
            OHLCV(timestamp_ms=21 * 60000, open=99.0, high=101.0, low=90.0, close=100.0, volume=3000.0)
        )
        res = pinbar_liquidity_sweep(candles)
        assert res["is_sweep"] is True
        assert res["sweep_type"] == "bullish_sweep"
        assert res["wick_ratio"] > 0.80
        assert res["volume_ratio"] >= 2.5

        # 2. Bearish sweep: Long upper wick + high volume
        candles_bear = [
            OHLCV(timestamp_ms=i * 60000, open=100.0, high=102.0, low=99.0, close=101.0, volume=1000.0)
            for i in range(20)
        ]
        # high=115.0, open=101.0, close=100.0, low=99.0 -> range=16.0, upper wick=(115-101)/16=14/16=0.875
        candles_bear.append(
            OHLCV(timestamp_ms=21 * 60000, open=101.0, high=115.0, low=99.0, close=100.0, volume=2500.0)
        )
        res_bear = pinbar_liquidity_sweep(candles_bear)
        assert res_bear["is_sweep"] is True
        assert res_bear["sweep_type"] == "bearish_sweep"

        # 3. Normal candle (no sweep)
        candles_normal = [
            OHLCV(timestamp_ms=i * 60000, open=100.0, high=102.0, low=98.0, close=101.0, volume=1000.0)
            for i in range(21)
        ]
        res_norm = pinbar_liquidity_sweep(candles_normal)
        assert res_norm["is_sweep"] is False
        assert res_norm["sweep_type"] == "none"

    def test_cvd_price_divergence(self):
        # 1. Bearish divergence: Price rising, but CVD falling (passive absorption / distribution)
        prices = [100.0 + i * 1.0 for i in range(20)]
        cvds = [1000.0 - i * 50.0 for i in range(20)]
        res = cvd_price_divergence(prices, cvds, lookback=15)
        assert res["is_divergence"] is True
        assert res["divergence_type"] == "bearish_divergence"
        assert res["correlation"] < -0.9

        # 2. Bullish divergence: Price falling, but CVD rising (passive accumulation / dip buying)
        prices_down = [100.0 - i * 1.0 for i in range(20)]
        cvds_up = [500.0 + i * 50.0 for i in range(20)]
        res_bull = cvd_price_divergence(prices_down, cvds_up, lookback=15)
        assert res_bull["is_divergence"] is True
        assert res_bull["divergence_type"] == "bullish_divergence"

        # 3. Confirmed trend: Price and CVD rising together
        cvds_in_sync = [500.0 + i * 50.0 for i in range(20)]
        res_sync = cvd_price_divergence(prices, cvds_in_sync, lookback=15)
        assert res_sync["divergence_type"] == "confirmed_trend"
        assert res_sync["is_divergence"] is False

    def test_oi_price_regime(self):
        # 1. Long buildup: Price up (+5%), OI up (+10%)
        p_up = [100.0] * 10 + [105.0]
        oi_up = [1000.0] * 10 + [1100.0]
        res1 = oi_price_regime(p_up, oi_up, lookback=10)
        assert res1["regime"] == "long_buildup"
        assert "增仓拉升" in res1["regime_desc"]

        # 2. Short squeeze: Price up (+5%), OI down (-10%)
        oi_down = [1000.0] * 10 + [900.0]
        res2 = oi_price_regime(p_up, oi_down, lookback=10)
        assert res2["regime"] == "short_squeeze"
        assert "轧空反弹" in res2["regime_desc"]

        # 3. Short buildup: Price down (-5%), OI up (+10%)
        p_down = [100.0] * 10 + [95.0]
        res3 = oi_price_regime(p_down, oi_up, lookback=10)
        assert res3["regime"] == "short_buildup"
        assert "增仓下砸" in res3["regime_desc"]

        # 4. Long liquidation: Price down (-5%), OI down (-10%)
        res4 = oi_price_regime(p_down, oi_down, lookback=10)
        assert res4["regime"] == "long_liquidation"
        assert "多头踩踏爆仓" in res4["regime_desc"]

    def test_bollinger_bands_and_squeeze(self):
        # 20 steady values around 100
        values = [100.0 + (i % 2) * 0.5 for i in range(30)]
        bb = bollinger_bands(values, period=20, std_dev_mult=2.0)
        assert bb["middle"] > 0
        assert bb["upper"] > bb["middle"]
        assert bb["lower"] < bb["middle"]
        assert bb["bandwidth"] > 0
        assert 0.0 <= bb["percent_b"] <= 1.0

        # Squeeze test: past bandwidths were large (0.10 ~ 0.20), current is 0.01 (historical minimum)
        bw_history = [0.15] * 90 + [0.01] * 10
        sq = bollinger_squeeze(bw_history, lookback=100)
        assert sq["is_squeezed"] is True
        assert sq["squeeze_score"] <= 0.15

    def test_funding_rate_zscore(self):
        # 70 rates around 0.0001 (0.01%), last rate is 0.001 (0.1%, extreme jump)
        rates = [0.0001] * 70 + [0.0010]
        res = funding_rate_zscore(rates, lookback=72)
        assert res["is_crowded"] is True
        assert res["crowding_status"] == "long_overcrowded"
        assert res["zscore"] > 2.5

        # Extreme negative funding rate
        rates_neg = [0.0001] * 70 + [-0.0010]
        res_neg = funding_rate_zscore(rates_neg, lookback=72)
        assert res_neg["is_crowded"] is True
        assert res_neg["crowding_status"] == "short_overcrowded"

    def test_candle_efficiency(self):
        # 1. High efficiency Marubozu (Open=100, Close=110, High=110.2, Low=99.8) -> body=10, range=10.4 -> eff=0.96
        candle_maru = [OHLCV(timestamp_ms=0, open=100.0, high=110.2, low=99.8, close=110.0, volume=500.0)]
        eff_high = candle_efficiency(candle_maru)
        assert eff_high["is_high_efficiency"] is True
        assert eff_high["is_fakeout_risk"] is False
        assert eff_high["efficiency"] > 0.85

        # 2. Fakeout Doji / Long wicks (Open=105, Close=105.5, High=120, Low=90) -> body=0.5, range=30 -> eff=0.016
        candle_doji = [OHLCV(timestamp_ms=0, open=105.0, high=120.0, low=90.0, close=105.5, volume=500.0)]
        eff_low = candle_efficiency(candle_doji)
        assert eff_low["is_fakeout_risk"] is True
        assert eff_low["is_high_efficiency"] is False
        assert eff_low["efficiency"] < 0.20

    def test_symbol_state_integration_factors(self):
        state = SymbolState("BTC")
        # Add 40 bars to satisfy min requirements for indicators
        for i in range(40):
            bar = Bar(
                symbol="BTC",
                timeframe="1h",
                open=100.0 + i,
                high=102.0 + i,
                low=99.0 + i,
                close=101.0 + i,
                volume=1000.0 + i * 10,
                timestamp_ms=i * 3600_000,
            )
            state._bars["1h"].append(bar)

        # Add some funding and OI snapshots
        for i in range(20):
            state.add_funding(FundingRate(symbol="BTC", rate=0.0001, timestamp_ms=i * 3600_000))
            state.add_oi(OISnapshot(symbol="BTC", open_interest=5000.0 + i * 100, mark_price=100.0 + i, timestamp_ms=i * 3600_000))

        ind = state.indicators("1h")
        assert "bb_bandwidth" in ind
        assert "squeeze_score" in ind
        assert "is_squeezed" in ind
        assert "candle_efficiency" in ind
        assert "pinbar_type" in ind
        assert "oi_regime" in ind
        assert "funding_zscore" in ind
