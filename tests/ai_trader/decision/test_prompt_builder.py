"""测试 PromptBuilder."""

from decimal import Decimal

from xmlx_vlm.ai_trader.decision.context import TradingContext
from xmlx_vlm.ai_trader.decision.prompt_builder import PromptBuilder
from xmlx_vlm.ai_trader.oms.core.account import AccountSnapshot


def test_prompt_builder_renders_account_and_positions():
    account = AccountSnapshot(equity=Decimal("10000"), available_margin=Decimal("8000"))
    ctx = TradingContext(
        current_time="2026-01-01T00:00:00Z",
        runtime_minutes=5,
        cycle_number=1,
        trader_id="t1",
        account=account,
        candidate_symbols=["BTC/USDC"],
    )
    builder = PromptBuilder(variant="conservative")
    prompts = builder.build(ctx)
    assert "AI Trader" in prompts.system_prompt
    assert "保守" in prompts.system_prompt
    assert "10000" in prompts.user_prompt
    assert "BTC/USDC" in prompts.user_prompt


def test_prompt_default_variant():
    account = AccountSnapshot(equity=Decimal("5000"))
    ctx = TradingContext(
        current_time="t",
        runtime_minutes=0,
        cycle_number=0,
        account=account,
    )
    prompts = PromptBuilder().build(ctx)
    assert prompts.variant == "default"
    assert "JSON 数组" in prompts.system_prompt
