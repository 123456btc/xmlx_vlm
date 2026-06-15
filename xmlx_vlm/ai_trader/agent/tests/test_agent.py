"""Agent 层单元测试."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from functools import wraps
from unittest.mock import AsyncMock, MagicMock

import pytest

from xmlx_vlm.ai_trader.agent.config import AgentObjective, PositionConstraint, RiskBudget


def async_test(func):
    """在没有 pytest-asyncio 时运行 async 测试的包装器."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper
from xmlx_vlm.ai_trader.agent.decision import ActionType, AgentDecision, SignalEvaluation, TradeProposal
from xmlx_vlm.ai_trader.agent.evaluator import SignalEvaluator
from xmlx_vlm.ai_trader.agent.explainability import ExplainabilityBuilder
from xmlx_vlm.ai_trader.agent.governance import ModelGovernance, VariantRegistry
from xmlx_vlm.ai_trader.agent.loop import AutonomousAgentLoop
from xmlx_vlm.ai_trader.agent.modes import AgentMode, ModeController
from xmlx_vlm.ai_trader.market_service.events import IndicatorAlertEvent


class TestAgentObjective:
    def test_risk_budget_effective_risk(self):
        budget = RiskBudget(max_risk_pct_per_trade=Decimal("2.0"))
        assert budget.effective_risk_usd(Decimal("10000")) == Decimal("200")

    def test_risk_budget_cap_usd(self):
        budget = RiskBudget(
            max_risk_pct_per_trade=Decimal("10.0"),
            max_risk_usd_per_trade=Decimal("100"),
        )
        assert budget.effective_risk_usd(Decimal("10000")) == Decimal("100")

    def test_objective_from_dict(self):
        obj = AgentObjective(
            risk_budget={"max_risk_pct_per_trade": "1.5"},
            position_constraint={"min_confidence": 70},
        )
        assert obj.risk_budget.max_risk_pct_per_trade == Decimal("1.5")
        assert obj.position_constraint.min_confidence == 70


class TestSignalEvaluator:
    @pytest.fixture
    def objective(self):
        return AgentObjective(
            risk_budget=RiskBudget(max_risk_pct_per_trade=Decimal("1.0")),
            position_constraint=PositionConstraint(
                max_position_size_usd=Decimal("5000"),
                max_leverage=5,
                min_confidence=60,
                min_risk_reward_ratio=Decimal("1.5"),
            ),
        )

    @pytest.fixture
    def evaluator(self, objective):
        return SignalEvaluator(objective)

    def test_evaluate_long_signal(self, evaluator):
        event = IndicatorAlertEvent(
            symbol="BTC",
            timestamp_ms=1,
            alert_type="ema_breakout",
            payload={"direction": "long", "confidence": 75},
        )
        evaluation = evaluator.evaluate(event, mark_price=Decimal("100000"), atr=Decimal("1000"))
        assert evaluation.signal_type == "ema_breakout"
        assert evaluation.confidence >= 75
        assert evaluation.stop_loss is not None
        assert evaluation.take_profit is not None
        assert evaluation.risk_reward_ratio >= Decimal("1.5")

    def test_evaluate_short_signal(self, evaluator):
        event = IndicatorAlertEvent(
            symbol="ETH",
            timestamp_ms=1,
            alert_type="rsi_overbought",
            payload={"direction": "short", "confidence": 80},
        )
        evaluation = evaluator.evaluate(event, mark_price=Decimal("3000"), atr=Decimal("50"))
        assert evaluation.signal_type == "rsi_overbought"
        assert evaluation.stop_loss > Decimal("3000")
        assert evaluation.take_profit < Decimal("3000")

    def test_build_proposal_passes_constraints(self, evaluator):
        event = IndicatorAlertEvent(
            symbol="BTC",
            timestamp_ms=1,
            alert_type="breakout",
            payload={"direction": "long", "confidence": 80},
        )
        evaluation = evaluator.evaluate(event, mark_price=Decimal("100000"), atr=Decimal("1000"))
        proposal = evaluator.build_proposal(
            evaluation, mark_price=Decimal("100000"), equity=Decimal("10000")
        )
        assert proposal is not None
        assert proposal.action == ActionType.OPEN_LONG
        assert proposal.size_usd > Decimal("0")
        assert proposal.size_usd <= Decimal("5000")
        assert proposal.risk_reward_ratio >= Decimal("1.5")

    def test_build_proposal_fails_low_confidence(self, evaluator):
        event = IndicatorAlertEvent(
            symbol="BTC",
            timestamp_ms=1,
            alert_type="weak_signal",
            payload={"direction": "long", "confidence": 30},
        )
        evaluation = evaluator.evaluate(event, mark_price=Decimal("100000"), atr=Decimal("1000"))
        proposal = evaluator.build_proposal(
            evaluation, mark_price=Decimal("100000"), equity=Decimal("10000")
        )
        assert proposal is None


class TestExplainability:
    @pytest.fixture
    def builder(self):
        obj = AgentObjective(
            position_constraint=PositionConstraint(min_confidence=60, min_risk_reward_ratio=Decimal("1.5"))
        )
        return ExplainabilityBuilder(obj)

    def test_rationale_contains_reasoning(self, builder):
        evaluation = SignalEvaluation(
            signal_type="breakout",
            symbol="BTC",
            confidence=80,
            risk_reward_ratio=Decimal("2.0"),
            stop_loss=Decimal("98000"),
            take_profit=Decimal("104000"),
            expected_return_pct=Decimal("4.0"),
            expected_risk_pct=Decimal("2.0"),
            metadata={"direction": "long"},
        )
        proposal = TradeProposal(
            action=ActionType.OPEN_LONG,
            symbol="BTC",
            size_usd=Decimal("1000"),
            confidence=80,
            stop_loss=Decimal("98000"),
            take_profit=Decimal("104000"),
            expected_return_pct=Decimal("4.0"),
            expected_risk_pct=Decimal("2.0"),
            risk_reward_ratio=Decimal("2.0"),
        )
        rationale = builder.build(evaluation, proposal, should_execute=True)
        assert rationale.symbol == "BTC"
        assert rationale.should_execute is True
        assert "80/100" in rationale.to_markdown()
        assert "2.0" in rationale.risk_reward_ratio

    def test_rationale_no_proposal(self, builder):
        evaluation = SignalEvaluation(
            signal_type="noise",
            symbol="BTC",
            confidence=30,
            risk_reward_ratio=Decimal("0.5"),
            metadata={"direction": "long"},
        )
        rationale = builder.build(evaluation, None, should_execute=False)
        assert rationale.should_execute is False
        assert rationale.action == "wait"


class TestModeController:
    def test_default_mode_observe(self):
        ctrl = ModeController()
        assert ctrl.mode == AgentMode.OBSERVE

    def test_mode_transition(self):
        ctrl = ModeController(AgentMode.ADVISE)
        ctrl.set_mode(AgentMode.FULL_AUTO)
        assert ctrl.mode == AgentMode.FULL_AUTO

    def test_kill_switch(self):
        bus = MagicMock()
        ctrl = ModeController(AgentMode.FULL_AUTO, event_bus=bus)
        ctrl.kill(reason="test")
        assert ctrl.is_killed is True
        assert ctrl.mode == AgentMode.OBSERVE
        assert bus.publish.called

    def test_can_execute_by_mode(self):
        ctrl = ModeController(AgentMode.FULL_AUTO)
        proposal = TradeProposal(
            action=ActionType.OPEN_LONG,
            symbol="BTC",
            size_usd=Decimal("1000"),
            confidence=50,
            risk_reward_ratio=Decimal("1.0"),
            expected_risk_pct=Decimal("2.0"),
        )
        assert ctrl.can_execute(proposal) is True

        ctrl.set_mode(AgentMode.OBSERVE)
        assert ctrl.can_execute(proposal) is False

    def test_semi_auto_only_high_confidence(self):
        ctrl = ModeController(AgentMode.SEMI_AUTO)
        low = TradeProposal(
            action=ActionType.OPEN_LONG,
            symbol="BTC",
            size_usd=Decimal("1000"),
            confidence=70,
            risk_reward_ratio=Decimal("1.5"),
            expected_risk_pct=Decimal("2.0"),
        )
        high = TradeProposal(
            action=ActionType.OPEN_LONG,
            symbol="BTC",
            size_usd=Decimal("1000"),
            confidence=90,
            risk_reward_ratio=Decimal("2.5"),
            expected_risk_pct=Decimal("0.5"),
        )
        assert ctrl.can_execute(low) is False
        assert ctrl.can_execute(high) is True


class TestGovernance:
    def test_registry_default_variant(self):
        reg = VariantRegistry()
        assert reg.get("default") is not None

    def test_shadow_record_and_resolve(self):
        gov = ModelGovernance()
        proposal = TradeProposal(
            action=ActionType.OPEN_LONG,
            symbol="BTC",
            size_usd=Decimal("1000"),
            confidence=80,
            stop_loss=Decimal("98000"),
            take_profit=Decimal("104000"),
        )
        record = gov.record_shadow("v1", "BTC", proposal, Decimal("100000"))
        assert record.realized_pnl_pct is None
        resolved = gov.resolve_shadow(record.record_id, Decimal("103000"))
        assert resolved is not None
        assert resolved.realized_pnl_pct is not None
        assert resolved.realized_pnl_pct > Decimal("0")

    def test_evaluate_variant(self):
        gov = ModelGovernance()
        proposal = TradeProposal(
            action=ActionType.OPEN_LONG,
            symbol="BTC",
            size_usd=Decimal("1000"),
            confidence=80,
        )
        record = gov.record_shadow("v1", "BTC", proposal, Decimal("100000"))
        gov.resolve_shadow(record.record_id, Decimal("101000"))
        stats = gov.evaluate_variant("v1")
        assert stats["sample_count"] == 1
        assert stats["win_rate"] == "100.00%"


class TestAutonomousLoop:
    @pytest.fixture
    def objective(self):
        return AgentObjective(
            risk_budget=RiskBudget(max_risk_pct_per_trade=Decimal("2.0")),
            position_constraint=PositionConstraint(
                max_position_size_usd=Decimal("10000"),
                max_leverage=5,
                min_confidence=60,
                min_risk_reward_ratio=Decimal("1.0"),
            ),
        )

    @pytest.fixture
    def mock_oms(self):
        oms = MagicMock()
        oms.portfolio_summary.return_value = {"account": {"equity": Decimal("10000")}}
        oms.sync = AsyncMock(return_value={"status": "synced"})
        oms.create_order.return_value = MagicMock(
            client_order_id="order1",
            to_dict=lambda: {"client_order_id": "order1"},
        )
        oms.submit_order = AsyncMock(return_value={"status": "dry_run", "order": {"client_order_id": "order1"}})
        oms.portfolio.get_position.return_value = None
        oms.event_bus = MagicMock()
        return oms

    @async_test
    async def test_observe_mode_records_decision(self, objective, mock_oms):
        decisions = []
        loop = AutonomousAgentLoop(
            oms=mock_oms,
            objective=objective,
            reporter=lambda d: decisions.append(d),
            price_provider=lambda s: Decimal("100000"),
            atr_provider=lambda s: Decimal("1000"),
        )
        loop.mode_controller.set_mode(AgentMode.OBSERVE)
        event = IndicatorAlertEvent(
            symbol="BTC",
            timestamp_ms=1,
            alert_type="breakout",
            payload={"direction": "long", "confidence": 80},
        )
        await loop._handle_alert(event)
        assert len(decisions) == 1
        assert decisions[0].symbol == "BTC"
        assert decisions[0].executed is False
        assert decisions[0].mode == "observe"
        assert decisions[0].rationale is not None

    @async_test
    async def test_full_auto_executes_proposal(self, objective, mock_oms):
        decisions = []
        loop = AutonomousAgentLoop(
            oms=mock_oms,
            objective=objective,
            reporter=lambda d: decisions.append(d),
            price_provider=lambda s: Decimal("100000"),
            atr_provider=lambda s: Decimal("1000"),
        )
        loop.mode_controller.set_mode(AgentMode.FULL_AUTO)
        event = IndicatorAlertEvent(
            symbol="BTC",
            timestamp_ms=1,
            alert_type="breakout",
            payload={"direction": "long", "confidence": 80},
        )
        await loop._handle_alert(event)
        assert len(decisions) == 1
        assert decisions[0].executed is True
        assert mock_oms.submit_order.called

    @async_test
    async def test_kill_switch_ignores_alerts(self, objective, mock_oms):
        loop = AutonomousAgentLoop(
            oms=mock_oms,
            objective=objective,
            price_provider=lambda s: Decimal("100000"),
            atr_provider=lambda s: Decimal("1000"),
        )
        loop.mode_controller.set_mode(AgentMode.FULL_AUTO)
        loop.mode_controller.kill("test")
        event = IndicatorAlertEvent(
            symbol="BTC",
            timestamp_ms=1,
            alert_type="breakout",
            payload={"direction": "long", "confidence": 80},
        )
        await loop._handle_alert(event)
        assert not mock_oms.submit_order.called

    @async_test
    async def test_semi_auto_requests_confirm(self, objective, mock_oms):
        confirmed = []
        loop = AutonomousAgentLoop(
            oms=mock_oms,
            objective=objective,
            price_provider=lambda s: Decimal("100000"),
            atr_provider=lambda s: Decimal("1000"),
            mode_controller=ModeController(
                AgentMode.SEMI_AUTO,
                human_confirm_callback=lambda p: (confirmed.append(True) or True),
            ),
        )
        event = IndicatorAlertEvent(
            symbol="BTC",
            timestamp_ms=1,
            alert_type="breakout",
            payload={"direction": "long", "confidence": 80},
        )
        await loop._handle_alert(event)
        assert len(confirmed) == 1
        assert mock_oms.submit_order.called
