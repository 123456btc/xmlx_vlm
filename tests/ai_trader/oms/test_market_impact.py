"""测试市场冲击模型."""

from decimal import Decimal

import pytest

from xmlx_vlm.ai_trader.oms.constants import OrderSide
from xmlx_vlm.ai_trader.oms.impact.market_impact import AlmgrenChrissImpactModel
from xmlx_vlm.ai_trader.oms.utils.decimal import ZERO


@pytest.fixture
def model():
    return AlmgrenChrissImpactModel()


def test_impact_increases_with_qty(model):
    small = model.estimate(
        Decimal("1"), OrderSide.BUY, Decimal("50000"),
        adv=Decimal("1000"), spread_pct=Decimal("0.01"), volatility=Decimal("0.03")
    )
    large = model.estimate(
        Decimal("10"), OrderSide.BUY, Decimal("50000"),
        adv=Decimal("1000"), spread_pct=Decimal("0.01"), volatility=Decimal("0.03")
    )
    assert large.expected_slippage_pct > small.expected_slippage_pct


def test_impact_uses_spread(model):
    with_spread = model.estimate(
        Decimal("1"), OrderSide.BUY, Decimal("50000"),
        adv=Decimal("1000"), spread_pct=Decimal("0.05"), volatility=Decimal("0.03")
    )
    no_spread = model.estimate(
        Decimal("1"), OrderSide.BUY, Decimal("50000"),
        adv=Decimal("1000"), spread_pct=Decimal("0"), volatility=Decimal("0.03")
    )
    assert with_spread.expected_slippage_pct > no_spread.expected_slippage_pct


def test_impact_urgency(model):
    passive = model.estimate(
        Decimal("5"), OrderSide.BUY, Decimal("50000"),
        adv=Decimal("1000"), spread_pct=Decimal("0.01"), volatility=Decimal("0.03"),
        urgency="passive"
    )
    aggressive = model.estimate(
        Decimal("5"), OrderSide.BUY, Decimal("50000"),
        adv=Decimal("1000"), spread_pct=Decimal("0.01"), volatility=Decimal("0.03"),
        urgency="aggressive"
    )
    assert aggressive.expected_slippage_pct > passive.expected_slippage_pct


def test_impact_estimate_fields(model):
    est = model.estimate(
        Decimal("1"), OrderSide.SELL, Decimal("50000"),
        adv=Decimal("1000"), spread_pct=Decimal("0.01"), volatility=Decimal("0.03")
    )
    assert est.expected_slippage_pct > ZERO
    assert est.expected_slippage_abs > ZERO
    assert est.confidence == "high"
