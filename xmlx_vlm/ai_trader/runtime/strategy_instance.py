"""单个策略实例."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TYPE_CHECKING

from xmlx_vlm.ai_trader.config import LOGS_DIR
from xmlx_vlm.ai_trader.decision.engine import DecisionEngine, DecisionEngineConfig
from xmlx_vlm.ai_trader.decision.llm_client import AutoLLMClient
from xmlx_vlm.ai_trader.oms.config.settings import OMSSettings
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.store.base import DecisionStore
from xmlx_vlm.ai_trader.store.sqlite_store import SQLiteDecisionStore
from xmlx_vlm.ai_trader.runtime.strategy_config import StrategyConfig
from xmlx_vlm.ai_trader.strategies.grid.grid_engine import GridEngine, GridEngineConfig

if TYPE_CHECKING:
    from xmlx_vlm.ai_trader.market_service.service import MarketDataService

logger = logging.getLogger(__name__)


class StrategyInstance:
    """单个策略的运行时表示."""

    def __init__(
        self,
        config: StrategyConfig,
        oms: Optional[OMSEngine] = None,
        store: Optional[DecisionStore] = None,
        engine: Optional[Any] = None,
        market_service: Optional["MarketDataService"] = None,
    ):
        self.config = config
        self._oms = oms
        self._store = store
        self._engine = engine
        self._market_service = market_service
        self._start_time: Optional[datetime] = None
        self._cycle_count = 0

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def oms(self) -> OMSEngine:
        if self._oms is None:
            settings = OMSSettings(**self.config.to_oms_settings_kwargs())
            self._oms = OMSEngine(
                settings=settings,
                order_sync_enabled=self.config.order_sync_enabled,
                order_sync_interval_seconds=self.config.order_sync_interval_seconds,
            )
        return self._oms

    @property
    def store(self) -> DecisionStore:
        if self._store is None:
            self._store = SQLiteDecisionStore(LOGS_DIR / "ai_trader.db")
        return self._store

    @property
    def engine(self) -> Any:
        """根据 strategy_type 返回对应的引擎实例."""
        if self._engine is None:
            if self.config.strategy_type == "grid":
                self._engine = self._create_grid_engine()
            elif self.config.strategy_type == "agent":
                self._engine = self._create_agent_engine()
            else:
                self._engine = self._create_decision_engine()
        return self._engine

    @property
    def is_running(self) -> bool:
        return self.engine.is_running

    async def start(self) -> None:
        if self.is_running:
            logger.warning("Strategy %s already running", self.id)
            return
        self._start_time = datetime.now(timezone.utc)
        if self.config.order_sync_enabled:
            await self.oms.start_order_sync()
        await self.engine.start()
        logger.info("Strategy %s started", self.id)

    async def stop(self) -> None:
        await self.engine.stop()
        if self.config.order_sync_enabled:
            await self.oms.stop_order_sync()
        logger.info("Strategy %s stopped", self.id)

    async def emergency_stop(self, flatten: bool = True) -> None:
        await self.engine.emergency_stop(flatten=flatten)
        logger.info("Strategy %s emergency stopped", self.id)

    def status(self) -> Dict[str, Any]:
        runtime_minutes = 0
        if self._start_time is not None:
            runtime_minutes = int(
                (datetime.now(timezone.utc) - self._start_time).total_seconds() // 60
            )
        status = {
            "id": self.id,
            "name": self.config.name or self.id,
            "strategy_type": self.config.strategy_type,
            "exchange": self.config.exchange,
            "is_running": self.is_running,
            "enabled": self.config.enabled,
            "live_enabled": self.config.live_enabled,
            "dry_run": self.config.dry_run,
            "runtime_minutes": runtime_minutes,
            "scan_interval_seconds": self.config.scan_interval_seconds,
            "symbols": self.config.symbols,
        }
        if self.config.strategy_type == "grid" and self.config.grid:
            status["grid"] = self.config.grid.model_dump()
        return status

    def portfolio_summary(self) -> Dict[str, Any]:
        return self.oms.portfolio_summary()

    def _create_decision_engine(self) -> DecisionEngine:
        engine_config = DecisionEngineConfig(
            trader_id=self.id,
            scan_interval_seconds=self.config.scan_interval_seconds,
            prompt_variant=self.config.prompt_variant,
            max_positions=self.config.max_positions,
            min_confidence=self.config.min_confidence,
            default_leverage=self.config.default_leverage,
            candidate_symbols=self.config.symbols,
            server_url=self.config.server_url,
            api_key=self.config.api_key,
            model_path=self.config.model_path,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            allow_mlx_fallback=self.config.allow_mlx_fallback,
        )
        return DecisionEngine(
            oms=self.oms,
            config=engine_config,
            store=self.store,
            llm_client=AutoLLMClient(
                server_url=engine_config.server_url,
                api_key=engine_config.api_key,
                model_path=engine_config.model_path,
                temperature=engine_config.temperature,
                max_tokens=engine_config.max_tokens,
                allow_mlx_fallback=engine_config.allow_mlx_fallback,
            ),
        )

    def _create_grid_engine(self) -> GridEngine:
        if self.config.grid is None:
            raise ValueError(f"Grid strategy {self.id} requires grid config")
        grid_config = self.config.grid
        engine_config = GridEngineConfig(
            trader_id=self.id,
            symbol=grid_config.symbol,
            upper_price=grid_config.upper_price,
            lower_price=grid_config.lower_price,
            grid_count=grid_config.grid_count,
            total_investment=grid_config.total_investment,
            max_drawdown_pct=grid_config.max_drawdown_pct,
            daily_loss_limit_pct=grid_config.daily_loss_limit_pct,
            scan_interval_seconds=self.config.scan_interval_seconds,
        )
        return GridEngine(
            oms=self.oms,
            config=engine_config,
            store=self.store,
        )

    def _create_agent_engine(self):
        if self._market_service is None:
            raise ValueError(f"Agent strategy {self.id} requires market_service")
        if self.config.agent is None:
            raise ValueError(f"Agent strategy {self.id} requires agent config")
        from xmlx_vlm.ai_trader.agent import AgentEngine, AgentMode
        from xmlx_vlm.ai_trader.agent.config import (
            AgentObjective,
            PositionConstraint,
            RiskBudget,
        )

        ac = self.config.agent
        objective = AgentObjective(
            daily_volatility_target_pct=ac.daily_volatility_target_pct,
            max_drawdown_pct=ac.max_drawdown_pct,
            sharpe_target=ac.sharpe_target,
            max_open_positions=ac.max_open_positions,
            preferred_timeframe=ac.preferred_timeframe,
            risk_budget=RiskBudget(
                max_risk_pct_per_trade=ac.max_risk_pct_per_trade,
                max_risk_usd_per_trade=ac.max_risk_usd_per_trade,
            ),
            position_constraint=PositionConstraint(
                max_position_size_usd=ac.max_position_size_usd,
                max_leverage=ac.max_leverage,
                max_positions=ac.max_open_positions,
                min_confidence=ac.min_confidence,
                min_risk_reward_ratio=ac.min_risk_reward_ratio,
            ),
        )
        return AgentEngine(
            trader_id=self.id,
            oms=self.oms,
            market_service=self._market_service,
            objective=objective,
            mode=AgentMode(ac.mode),
            server_url=self.config.server_url,
            api_key=self.config.api_key,
            model_name=self.config.model_path,
        )
