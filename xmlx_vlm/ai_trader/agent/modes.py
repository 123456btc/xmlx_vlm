"""Agent 运行模式与急停控制."""

from __future__ import annotations

import logging
import threading
from enum import Enum
from typing import Callable, List, Optional

from xmlx_vlm.ai_trader.oms.events.bus import EventBus, SyncEventBus
from xmlx_vlm.ai_trader.oms.events.types import KillSwitchEvent
from xmlx_vlm.ai_trader.oms.constants import EventType

logger = logging.getLogger(__name__)


class AgentMode(str, Enum):
    """Agent 运行模式.

    - OBSERVE: 只观察、记录，不生成交易提案
    - ADVISE: 生成提案并报告人类，但不执行
    - SEMI_AUTO: 生成提案，低风险/高置信度自动执行，其他请求确认
    - FULL_AUTO: 完全自主执行
    """

    OBSERVE = "observe"
    ADVISE = "advise"
    SEMI_AUTO = "semi_auto"
    FULL_AUTO = "full_auto"


class ModeController:
    """管理 Agent 模式切换、急停、确认回调."""

    def __init__(
        self,
        initial_mode: AgentMode = AgentMode.OBSERVE,
        event_bus: Optional[EventBus] = None,
        human_confirm_callback: Optional[Callable[["TradeProposal"], bool]] = None,
    ):
        self._lock = threading.RLock()
        self._mode = AgentMode(initial_mode)
        self._killed = False
        self._event_bus = event_bus or SyncEventBus()
        self._human_confirm_callback = human_confirm_callback

    @property
    def mode(self) -> AgentMode:
        with self._lock:
            return self._mode

    @property
    def is_killed(self) -> bool:
        with self._lock:
            return self._killed

    def set_mode(self, mode: AgentMode) -> None:
        with self._lock:
            old = self._mode
            self._mode = AgentMode(mode)
            logger.info("Agent mode changed: %s -> %s", old.value, self._mode.value)

    def kill(self, reason: str = "manual kill", triggered_by: str = "user") -> None:
        """一键全局急停."""
        with self._lock:
            if self._killed:
                return
            self._killed = True
            self._mode = AgentMode.OBSERVE
        logger.critical("KILL SWITCH triggered by %s: %s", triggered_by, reason)
        self._event_bus.publish(
            KillSwitchEvent(
                event_type=EventType.KILL_SWITCH_TRIGGERED,
                triggered_by=triggered_by,
                reason=reason,
                flatten_positions=True,
            )
        )

    def reset_kill(self) -> None:
        """仅在人工确认后重置急停状态（不会自动恢复模式）."""
        with self._lock:
            self._killed = False
            logger.warning("Kill switch reset manually")

    def can_execute(self, proposal) -> bool:
        """根据当前模式判断提案是否可执行."""
        with self._lock:
            if self._killed:
                return False
            mode = self._mode

        if mode == AgentMode.FULL_AUTO:
            return True
        if mode == AgentMode.OBSERVE:
            return False
        if mode == AgentMode.ADVISE:
            return False
        if mode == AgentMode.SEMI_AUTO:
            # 半自动：高置信度 + 低风险敞口才自动执行
            return (
                proposal.confidence >= 80
                and proposal.risk_reward_ratio >= 2
                and proposal.expected_risk_pct <= Decimal("1.0")
            )
        return False

    def needs_human_confirm(self, proposal) -> bool:
        """是否需要人类确认."""
        with self._lock:
            if self._killed:
                return False
            mode = self._mode
        if mode == AgentMode.SEMI_AUTO:
            return not self.can_execute(proposal)
        return False

    def request_human_confirm(self, proposal) -> bool:
        """请求人类确认；未配置回调时默认拒绝."""
        if self._human_confirm_callback is None:
            logger.warning("No human confirm callback configured; rejecting %s", proposal.symbol)
            return False
        try:
            return bool(self._human_confirm_callback(proposal))
        except Exception:
            logger.exception("Human confirm callback failed for %s", proposal.symbol)
            return False


# Avoid circular import type hints
from xmlx_vlm.ai_trader.agent.decision import TradeProposal  # noqa: E402
from decimal import Decimal  # noqa: E402
