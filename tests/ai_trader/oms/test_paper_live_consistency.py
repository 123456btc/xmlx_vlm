"""测试 Paper-to-Live 一致性校验."""

from decimal import Decimal

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState
from xmlx_vlm.ai_trader.oms.consistency.validator import PaperLiveConsistencyValidator
from xmlx_vlm.ai_trader.oms.core.order import Order


def make_order(filled_qty: Decimal, avg_px: Decimal, state: OrderState) -> Order:
    order = Order(
        symbol="BTC/USDC",
        side=OrderSide.BUY,
        qty=Decimal("1.0"),
        order_type="market",
    )
    order.filled_qty = filled_qty
    order.avg_fill_price = avg_px
    order.remaining_qty = order.qty - filled_qty
    order.state = state
    return order


def test_consistent_orders():
    paper = make_order(Decimal("1.0"), Decimal("50000"), OrderState.FILLED)
    live = make_order(Decimal("1.0"), Decimal("50002"), OrderState.FILLED)
    validator = PaperLiveConsistencyValidator(tolerance_pct=Decimal("0.1"))
    report = validator.compare(paper, live)
    assert report.consistent


def test_inconsistent_price():
    paper = make_order(Decimal("1.0"), Decimal("50000"), OrderState.FILLED)
    live = make_order(Decimal("1.0"), Decimal("51000"), OrderState.FILLED)
    validator = PaperLiveConsistencyValidator(tolerance_pct=Decimal("0.1"))
    report = validator.compare(paper, live)
    assert not report.consistent
    assert report.avg_price_diff_pct > Decimal("0.1")


def test_inconsistent_state():
    paper = make_order(Decimal("1.0"), Decimal("50000"), OrderState.FILLED)
    live = make_order(Decimal("1.0"), Decimal("50000"), OrderState.CANCELLED)
    validator = PaperLiveConsistencyValidator(tolerance_pct=Decimal("1.0"))
    report = validator.compare(paper, live)
    assert not report.consistent
    assert any("state" in m.lower() for m in report.messages)
