"""测试决策数据模型."""

from decimal import Decimal

from xmlx_vlm.ai_trader.decision.decision import Decision, FullDecision


def test_decision_parses_open_long():
    d = Decision(
        action="open_long",
        symbol="BTC/USDC",
        position_size_usd=Decimal("500"),
        leverage=5,
        confidence=75,
        reasoning="breakout",
    )
    assert d.is_open
    assert d.side == "buy"
    assert d.confidence == 75


def test_decision_normalizes_action_and_symbol():
    d = Decision(action="  OPEN_SHORT ", symbol=" btc/usdc ")
    assert d.action == "open_short"
    assert d.symbol == "BTC/USDC"
    assert d.side == "sell"


def test_decision_to_dict_roundtrip():
    d = Decision(
        action="open_long",
        symbol="ETH/USDC",
        position_size_usd=Decimal("1000"),
        price=Decimal("3000"),
        confidence=80,
    )
    data = d.to_dict()
    d2 = Decision.from_dict(data)
    assert d2.action == d.action
    assert d2.symbol == d.symbol
    assert d2.position_size_usd == d.position_size_usd


def test_full_decision_to_dict():
    fd = FullDecision(
        trader_id="t1",
        cycle_number=1,
        decisions=[Decision(action="wait", symbol="BTC/USDC", confidence=0)],
        raw_response="[]",
    )
    data = fd.to_dict()
    assert data["trader_id"] == "t1"
    assert data["cycle_number"] == 1
    assert len(data["decisions"]) == 1
