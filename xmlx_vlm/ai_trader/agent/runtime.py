"""Agent 策略运行时包装."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from xmlx_vlm.ai_trader.agent.config import AgentObjective
from xmlx_vlm.ai_trader.agent.loop import AutonomousAgentLoop
from xmlx_vlm.ai_trader.agent.modes import AgentMode, ModeController
from xmlx_vlm.ai_trader.market_service.events import IndicatorAlertEvent
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.config import DEFAULT_API_KEY, DEFAULT_MODEL, DEFAULT_SERVER_URL
from xmlx_vlm.ai_trader.store.session_db import QuantSessionDB

if TYPE_CHECKING:
    from xmlx_vlm.ai_trader.agent.decision import AgentDecision
    from xmlx_vlm.ai_trader.market_service.service import MarketDataService

logger = logging.getLogger(__name__)


class AgentEngine:
    """把 AutonomousAgentLoop 包装成与 DecisionEngine / GridEngine 一致的策略引擎接口.

    生命周期：start / stop / emergency_stop / is_running
    """

    def __init__(
        self,
        trader_id: str,
        oms: OMSEngine,
        market_service: "MarketDataService",
        objective: AgentObjective,
        mode: AgentMode = AgentMode.OBSERVE,
        reporter: Optional[Callable[["AgentDecision"], None]] = None,
        human_confirm_callback: Optional[Callable[[Any], bool]] = None,
        server_url: str = DEFAULT_SERVER_URL,
        api_key: str = DEFAULT_API_KEY,
        model_name: str = DEFAULT_MODEL,
        db: Optional[QuantSessionDB] = None,
    ):
        self.trader_id = trader_id
        self.oms = oms
        self.market_service = market_service
        self.objective = objective
        self.reporter = reporter

        from xmlx_vlm.ai_trader.agent.providers import MarketDataProvider
        from xmlx_vlm.ai_trader.agent.evaluator import LLMSignalEvaluator

        db = db or QuantSessionDB()
        evaluator = LLMSignalEvaluator(
            objective=objective,
            db=db,
            server_url=server_url,
            api_key=api_key,
            model_name=model_name,
            use_fallback=True,
        )

        provider = MarketDataProvider(market_service)
        self.loop = AutonomousAgentLoop(
            oms=oms,
            objective=objective,
            mode_controller=ModeController(
                initial_mode=mode,
                event_bus=oms.event_bus,
                human_confirm_callback=human_confirm_callback,
            ),
            reporter=reporter,
            price_provider=provider.get_price,
            atr_provider=provider.get_atr,
            evaluator=evaluator,
        )
        self._alert_handler = self.loop.on_alert

    @property
    def is_running(self) -> bool:
        return self.loop.is_running

    @property
    def mode(self) -> AgentMode:
        return self.loop.mode_controller.mode

    def set_mode(self, mode: AgentMode) -> None:
        self.loop.mode_controller.set_mode(mode)

    async def start(self) -> None:
        if self.is_running:
            logger.warning("AgentEngine %s already running", self.trader_id)
            return
        # 订阅行情服务的指标警报事件
        self.market_service.event_bus.subscribe(
            IndicatorAlertEvent, self._alert_handler
        )
        await self.loop.start()
        logger.info("AgentEngine %s started in %s mode", self.trader_id, self.mode.value)

    async def stop(self) -> None:
        if not self.is_running:
            return
        await self.loop.stop()
        try:
            self.market_service.event_bus.unsubscribe(
                IndicatorAlertEvent, self._alert_handler
            )
        except Exception:
            logger.warning("Failed to unsubscribe agent alert handler")
        logger.info("AgentEngine %s stopped", self.trader_id)

    async def emergency_stop(self, flatten: bool = True) -> None:
        await self.loop.emergency_stop()
        logger.info("AgentEngine %s emergency stopped", self.trader_id)

    def summary(self) -> Dict[str, Any]:
        return {
            "trader_id": self.trader_id,
            **self.loop.summary(),
        }
