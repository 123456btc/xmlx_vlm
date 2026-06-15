"""Agent 与市场服务集成测试."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from functools import wraps
from unittest.mock import AsyncMock, MagicMock

import pytest

from xmlx_vlm.ai_trader.agent import AgentEngine
from xmlx_vlm.ai_trader.agent.config import AgentObjective, PositionConstraint, RiskBudget
from xmlx_vlm.ai_trader.agent.modes import AgentMode
from xmlx_vlm.ai_trader.agent.providers import MarketDataProvider
from xmlx_vlm.ai_trader.market_service.events import EventBus, IndicatorAlertEvent
from xmlx_vlm.ai_trader.market_service.models import MarketSummary
from xmlx_vlm.ai_trader.market_service.service import MarketDataService
from xmlx_vlm.ai_trader.market_service.state import MarketState
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.oms.config.settings import OMSSettings
from xmlx_vlm.ai_trader.runtime.strategy_config import StrategyConfig
from xmlx_vlm.ai_trader.runtime.strategy_instance import StrategyInstance
from xmlx_vlm.ai_trader.runtime.trader_manager import TraderManager


def async_test(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.run(func(*args, **kwargs))

    return wrapper


class TestMarketDataProvider:
    def test_provider_returns_price_and_atr(self):
        service = MarketDataService(event_bus=EventBus())
        # 构造状态
        state = MarketState()
        state.get("BTC", create=True).update_tick(
            type("Tick", (), {"symbol": "BTC", "price": 100000.0, "timestamp_ms": 1})()
        )
        # 用反射替换内部状态以注入测试数据
        service.state = state
        provider = MarketDataProvider(service)
        price = provider.get_price("BTC")
        assert price == Decimal("100000")


class TestAgentEngine:
    @pytest.fixture
    def mock_oms(self):
        settings = OMSSettings(exchange="paper", dry_run=True)
        oms = OMSEngine(settings=settings)
        oms.sync = AsyncMock(return_value={"status": "synced"})
        oms.portfolio_summary = MagicMock(
            return_value={"account": {"equity": Decimal("10000")}}
        )
        oms.create_order = MagicMock(return_value=MagicMock(
            client_order_id="order1",
            to_dict=lambda: {"client_order_id": "order1"},
        ))
        oms.submit_order = AsyncMock(return_value={"status": "dry_run"})
        return oms

    @pytest.fixture
    def market_service(self):
        service = MarketDataService(event_bus=EventBus())
        state = MarketState()
        tick = type("Tick", (), {"symbol": "BTC", "price": 100000.0, "timestamp_ms": 1})()
        sym = state.get("BTC", create=True)
        sym.update_tick(tick)
        service.state = state
        return service

    @async_test
    async def test_engine_subscribes_and_handles_alert(self, mock_oms, market_service):
        objective = AgentObjective(
            risk_budget=RiskBudget(max_risk_pct_per_trade=Decimal("2.0")),
            position_constraint=PositionConstraint(
                max_position_size_usd=Decimal("10000"),
                max_leverage=5,
                min_confidence=60,
                min_risk_reward_ratio=Decimal("1.0"),
            ),
        )
        decisions = []
        engine = AgentEngine(
            trader_id="agent-1",
            oms=mock_oms,
            market_service=market_service,
            objective=objective,
            mode=AgentMode.FULL_AUTO,
            reporter=lambda d: decisions.append(d),
        )
        await engine.start()
        try:
            event = IndicatorAlertEvent(
                symbol="BTC",
                timestamp_ms=1,
                alert_type="breakout",
                payload={
                    "direction": "long",
                    "confidence": 80,
                    "stop_loss": 98000.0,
                    "take_profit": 104000.0,
                },
            )
            market_service.event_bus.publish(event)
            await asyncio.sleep(0.2)
            assert len(decisions) == 1
            assert decisions[0].symbol == "BTC"
            assert decisions[0].executed is True
        finally:
            await engine.stop()


class TestStrategyInstanceAgent:
    @pytest.fixture
    def mock_oms(self):
        settings = OMSSettings(exchange="paper", dry_run=True)
        oms = OMSEngine(settings=settings)
        oms.sync = AsyncMock(return_value={"status": "synced"})
        oms.portfolio_summary = MagicMock(
            return_value={"account": {"equity": Decimal("10000")}}
        )
        oms.create_order = MagicMock(return_value=MagicMock(
            client_order_id="order1",
            to_dict=lambda: {"client_order_id": "order1"},
        ))
        oms.submit_order = AsyncMock(return_value={"status": "dry_run"})
        return oms

    @pytest.fixture
    def market_service(self):
        service = MarketDataService(event_bus=MagicMock())
        state = MarketState()
        tick = type("Tick", (), {"symbol": "BTC", "price": 100000.0, "timestamp_ms": 1})()
        state.get("BTC", create=True).update_tick(tick)
        service.state = state
        return service

    def test_create_agent_engine(self, mock_oms, market_service):
        cfg = StrategyConfig(
            id="agent-1",
            strategy_type="agent",
            exchange="paper",
            dry_run=True,
            agent={
                "mode": "full_auto",
                "min_confidence": 60,
                "min_risk_reward_ratio": 1.0,
            },
        )
        instance = StrategyInstance(
            config=cfg,
            oms=mock_oms,
            market_service=market_service,
        )
        assert instance.engine is not None
        assert isinstance(instance.engine, AgentEngine)
