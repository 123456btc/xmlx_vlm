"""Tests for PTC (Programmatic Tool Calling), Deterministic Verifier, and Plugin Architecture."""

import asyncio
from decimal import Decimal
import pytest

from xmlx_vlm.ai_trader.agent.config import AgentObjective, PositionConstraint, RiskBudget
from xmlx_vlm.ai_trader.agent.decision import ActionType, SignalEvaluation, TradeProposal
from xmlx_vlm.ai_trader.agent.evaluator import SignalEvaluator
from xmlx_vlm.ai_trader.agent.verifier import (
    DeterministicProposalVerifier,
    KellyCriterionSizer,
    MathematicalRiskRewardVerifier,
    VerificationResult,
)
from xmlx_vlm.ai_trader.benchmarks.replay_benchmark import DeterministicReplayBenchmark
from xmlx_vlm.ai_trader.market_service.events import IndicatorAlertEvent
from xmlx_vlm.ai_trader.runtime.plugins import (
    BaseTraderPlugin,
    PluginContext,
    PluginManager,
    PluginMetadata,
)
from xmlx_vlm.ai_trader.sdk.client import TraderSDK
from xmlx_vlm.ai_trader.tools.code_sandbox import ExecuteCodeTool, sanitize_traceback
from xmlx_vlm.ai_trader.tools.registry import ToolRegistry


class TestPTCCodeSandbox:
    """测试 PTC 代码沙箱执行."""

    def test_execute_simple_math_and_result(self):
        tool = ExecuteCodeTool()
        code = """
a = 10
b = 20
c = math.sqrt(a * b)
result = {"sum": a + b, "sqrt_prod": c}
"""
        res = tool.run(code)
        assert "[Execution Success" in res
        assert '"sum": 30' in res

    def test_execute_print_output(self):
        tool = ExecuteCodeTool()
        code = """
print("Scanning top 5 symbols...")
print("Found candidate BTC")
"""
        res = tool.run(code)
        assert "[Output]:" in res
        assert "Scanning top 5 symbols..." in res
        assert "Found candidate BTC" in res

    def test_execute_with_sdk(self):
        tool = ExecuteCodeTool()
        code = """
ticker = sdk.market.get_ticker("BTC")
result = {"symbol": ticker.get("symbol"), "has_source": "source" in ticker}
"""
        res = tool.run(code)
        assert "[Execution Success" in res
        assert '"symbol": "BTC"' in res

    def test_syntax_error_handling(self):
        tool = ExecuteCodeTool()
        code = "def broken_code(:"
        res = tool.run(code)
        assert "[Execution Error]: SyntaxError" in res

    def test_credential_sanitization(self):
        fake_pk = "0x" + "a" * 64
        fake_addr = "0x" + "b" * 40
        raw = f"Exception at key {fake_pk} and address {fake_addr}"
        sanitized = sanitize_traceback(raw)
        assert "[REDACTED_KEY]" in sanitized
        assert "[REDACTED_ADDR]" in sanitized
        assert fake_pk not in sanitized

    def test_registry_integration(self):
        reg = ToolRegistry(enable_ptc=True)
        tool = reg.get_tool("execute_code")
        assert tool is not None
        output = reg.execute("execute_code", {"code": "result = 42"})
        assert "42" in output


class TestDeterministicVerifier:
    """测试确定性数学与规则验证器."""

    def test_mathematical_risk_reward_verifier_success(self):
        verifier = MathematicalRiskRewardVerifier(min_rr=2.0)
        proposal = TradeProposal(
            action=ActionType.OPEN_LONG,
            symbol="BTC",
            size_usd=Decimal("1000.0"),
            entry_price=Decimal("60000.0"),
            stop_loss=Decimal("59000.0"),    # 风险 = 1000
            take_profit=Decimal("63000.0"),  # 收益 = 3000 -> RR = 3.0
            confidence=85,
        )
        res = verifier.verify(proposal, atr=Decimal("800.0"))
        assert res.passed is True
        assert res.metrics["risk_reward_ratio"] == 3.0

    def test_mathematical_risk_reward_verifier_low_rr_rejected(self):
        verifier = MathematicalRiskRewardVerifier(min_rr=2.0)
        proposal = TradeProposal(
            action=ActionType.OPEN_LONG,
            symbol="ETH",
            size_usd=Decimal("1000.0"),
            entry_price=Decimal("3000.0"),
            stop_loss=Decimal("2900.0"),    # 风险 = 100
            take_profit=Decimal("3100.0"),  # 收益 = 100 -> RR = 1.0 < 2.0
            confidence=90,
        )
        res = verifier.verify(proposal)
        assert res.passed is False
        assert any("盈亏比过低" in r for r in res.rejection_reasons)

    def test_inverted_stop_loss_rejected(self):
        verifier = MathematicalRiskRewardVerifier(min_rr=1.5)
        # 多头但是止损高于入场价
        proposal = TradeProposal(
            action=ActionType.OPEN_LONG,
            symbol="BTC",
            size_usd=Decimal("1000.0"),
            entry_price=Decimal("60000.0"),
            stop_loss=Decimal("61000.0"),
            take_profit=Decimal("65000.0"),
        )
        res = verifier.verify(proposal)
        assert res.passed is False
        assert any("顺序错误" in r for r in res.rejection_reasons)

    def test_kelly_criterion_sizer(self):
        sizer = KellyCriterionSizer(base_win_rate=0.5, fraction=0.25, max_loss_budget_pct=0.02)
        proposal = TradeProposal(
            action=ActionType.OPEN_LONG,
            symbol="BTC",
            size_usd=Decimal("5000.0"),
            entry_price=Decimal("60000.0"),
            stop_loss=Decimal("59000.0"),
            take_profit=Decimal("63000.0"),
        )
        equity = Decimal("100000.0")
        res = sizer.compute_size(proposal, equity=equity, rr=3.0)
        assert res.passed is True
        assert res.metrics["recommended_qty"] > 0
        assert res.metrics["max_loss_dollar"] <= 2000.0  # 2% of 100,000

    def test_full_deterministic_pipeline(self):
        pipeline = DeterministicProposalVerifier(min_rr=1.8, max_risk_pct=0.02)
        proposal = TradeProposal(
            action=ActionType.OPEN_SHORT,
            symbol="SOL",
            size_usd=Decimal("2000.0"),
            entry_price=Decimal("150.0"),
            stop_loss=Decimal("155.0"),    # 风险 = 5
            take_profit=Decimal("135.0"),  # 收益 = 15 -> RR = 3.0
            confidence=80,
        )
        equity = Decimal("50000.0")
        res = pipeline.verify_proposal(proposal, equity=equity, atr=Decimal("3.0"))
        assert res.passed is True
        assert res.metrics["risk_reward_ratio"] == 3.0


class TestPluginArchitecture:
    """测试微内核插件系统与生命周期管理."""

    def test_plugin_load_and_unload_lifecycle(self):
        async def _test():
            manager = PluginManager()

            class MockStrategyPlugin(BaseTraderPlugin):
                metadata = PluginMetadata(name="mock_strategy", version="1.0.0")

                def __init__(self):
                    super().__init__()
                    self.loaded = False
                    self.task_ran = False

                def on_load(self, ctx: PluginContext) -> None:
                    self.loaded = True

                    async def loop():
                        try:
                            while True:
                                self.task_ran = True
                                await asyncio.sleep(0.01)
                        except asyncio.CancelledError:
                            pass

                    ctx.spawn_task("test_loop", loop())

            plugin = MockStrategyPlugin()
            success = manager.load_plugin(plugin)
            assert success is True
            assert plugin.loaded is True
            assert len(manager.list_plugins()) == 1

            # Let task run briefly
            await asyncio.sleep(0.03)
            assert plugin.task_ran is True

            # Unload and verify task cancellation & cleanup
            unload_success = manager.unload_plugin("mock_strategy")
            assert unload_success is True
            assert len(manager.list_plugins()) == 0

        asyncio.run(_test())


class TestReplayBenchmark:
    """测试离线确定性回放评测套件."""

    def test_run_benchmark_scenario(self):
        bench = DeterministicReplayBenchmark()
        report = bench.run_benchmark(scenario_type="bullish_breakout")
        assert report.ticks_processed == 20
        assert report.proposals_generated > 0
        assert report.duration_ms > 0
        assert report.token_savings_estimate_pct > 50.0
