"""多策略生命周期管理."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from xmlx_vlm.ai_trader.intelligence.brain import Brain, BrainConfig
from xmlx_vlm.ai_trader.intelligence.signal import Signal
from xmlx_vlm.ai_trader.runtime.strategy_config import StrategyConfig
from xmlx_vlm.ai_trader.runtime.strategy_instance import StrategyInstance

if TYPE_CHECKING:
    from xmlx_vlm.ai_trader.market_service.service import MarketDataService

logger = logging.getLogger(__name__)


class TraderManager:
    """管理多个策略实例的生命周期与状态聚合."""

    def __init__(
        self,
        brain: Optional[Brain] = None,
        market_service: Optional["MarketDataService"] = None,
    ):
        self._strategies: Dict[str, StrategyInstance] = {}
        self._load_errors: Dict[str, str] = {}
        self._brain = brain
        self._market_service = market_service
        if self._brain is not None:
            self._brain.register_handler(self._on_signal)

    @property
    def brain(self) -> Optional[Brain]:
        return self._brain

    def attach_brain(self, brain: Brain) -> None:
        """附加一个 Brain 实例."""
        if self._brain is not None:
            self._brain.unregister_handler(self._on_signal)
        self._brain = brain
        brain.register_handler(self._on_signal)

    def _on_signal(self, signal: Signal) -> None:
        """Brain 信号回调：记录日志，未来可扩展为影响策略决策."""
        logger.info(
            "[signal] %s %s %s: %s",
            signal.severity.value,
            signal.type,
            signal.symbol,
            signal.title,
        )
        # P1：仅记录；P2 可通过共享事件总线影响策略 prompt

    def register(
        self,
        config: StrategyConfig,
        instance: Optional[StrategyInstance] = None,
    ) -> StrategyInstance:
        """注册一个策略配置."""
        if instance is None:
            instance = StrategyInstance(
                config,
                market_service=self._market_service,
            )
        self._strategies[config.id] = instance
        self._load_errors.pop(config.id, None)
        logger.info("Registered strategy %s (%s)", config.id, config.name)
        return instance

    def unregister(self, strategy_id: str) -> bool:
        """注销策略."""
        instance = self._strategies.pop(strategy_id, None)
        if instance is None:
            return False
        if instance.is_running:
            asyncio.create_task(instance.stop())
        logger.info("Unregistered strategy %s", strategy_id)
        return True

    def get(self, strategy_id: str) -> Optional[StrategyInstance]:
        return self._strategies.get(strategy_id)

    def list_ids(self) -> List[str]:
        return list(self._strategies.keys())

    def list_all(self) -> List[StrategyInstance]:
        return list(self._strategies.values())

    async def start(self, strategy_id: str) -> bool:
        instance = self._strategies.get(strategy_id)
        if instance is None:
            logger.warning("Strategy %s not found", strategy_id)
            return False
        await instance.start()
        return True

    async def stop(self, strategy_id: str) -> bool:
        instance = self._strategies.get(strategy_id)
        if instance is None:
            return False
        await instance.stop()
        return True

    async def start_all(self) -> None:
        logger.info("Starting all enabled strategies...")
        if self._brain is not None:
            await self._brain.start()
        for instance in self._strategies.values():
            if instance.config.enabled:
                try:
                    await instance.start()
                except Exception as exc:
                    logger.exception("Failed to start strategy %s", instance.id)
                    self._load_errors[instance.id] = str(exc)

    async def stop_all(self) -> None:
        logger.info("Stopping all strategies...")
        for instance in self._strategies.values():
            try:
                await instance.stop()
            except Exception as exc:
                logger.exception("Failed to stop strategy %s", instance.id)
        if self._brain is not None:
            await self._brain.stop()

    async def emergency_stop(self, strategy_id: Optional[str] = None, flatten: bool = True) -> None:
        if strategy_id is not None:
            instance = self._strategies.get(strategy_id)
            if instance is not None:
                await instance.emergency_stop(flatten=flatten)
            return
        for instance in self._strategies.values():
            try:
                await instance.emergency_stop(flatten=flatten)
            except Exception as exc:
                logger.exception("Emergency stop failed for strategy %s", instance.id)

    def get_status(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        instance = self._strategies.get(strategy_id)
        if instance is None:
            return None
        status = instance.status()
        status["load_error"] = self._load_errors.get(strategy_id)
        return status

    def get_all_status(self) -> List[Dict[str, Any]]:
        return [self.get_status(sid) for sid in self._strategies]

    def get_comparison(self) -> Dict[str, Any]:
        """返回所有策略的对比数据."""
        traders = []
        for instance in self._strategies.values():
            status = instance.status()
            try:
                summary = instance.portfolio_summary()
                account = summary.get("account", {})
                traders.append(
                    {
                        "id": status["id"],
                        "name": status["name"],
                        "exchange": status["exchange"],
                        "is_running": status["is_running"],
                        "total_equity": account.get("equity", "0"),
                        "available_margin": account.get("available_margin", "0"),
                        "margin_utilization_pct": account.get("margin_utilization_pct", "0"),
                        "unrealized_pnl": summary.get("unrealized_pnl", "0"),
                        "position_count": len(summary.get("positions", [])),
                        "runtime_minutes": status["runtime_minutes"],
                    }
                )
            except Exception as exc:
                logger.warning("Failed to get comparison for %s: %s", instance.id, exc)
                traders.append(
                    {
                        "id": status["id"],
                        "name": status["name"],
                        "exchange": status["exchange"],
                        "is_running": status["is_running"],
                        "error": str(exc),
                    }
                )
        return {"traders": traders, "count": len(traders)}
