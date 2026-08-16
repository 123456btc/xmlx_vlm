# SPDX-License-Identifier: Apache-2.0
"""
Dynamic Chandelier Trailing Stop and Breakeven Position Lifecycle Manager.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional

from xmlx_vlm.ai_trader.oms.constants import PositionSide
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO
from xmlx_vlm.ai_trader.oms.utils.symbol import normalize_symbol

logger = logging.getLogger(__name__)


@dataclass
class TrailingPositionState:
    """单个持仓的动态追踪状态."""

    symbol: str
    side: PositionSide
    entry_price: Decimal
    highest_price: Decimal
    lowest_price: Decimal
    current_stop_loss: Decimal
    take_profit: Optional[Decimal]
    atr: Decimal
    trailing_mult: Decimal
    breakeven_triggered: bool = False
    chandelier_active: bool = False


@dataclass
class TrailingStopSignal:
    """追踪止损触发信号."""

    should_close: bool
    symbol: str
    trigger_type: str  # 'chandelier_stop', 'breakeven_stop', 'hard_stop', 'take_profit', 'none'
    current_price: Decimal
    effective_stop_price: Decimal
    reason: str


class TrailingStopManager:
    """动态吊灯追踪止损与保本止损管理器 (Chandelier Trailing Exit & Break-Even)."""

    def __init__(
        self,
        default_trailing_mult: float = 3.0,     # 吊灯止损倍数 3.0 * ATR
        breakeven_trigger_r: float = 1.0,       # 达到 1.0 R (1 * ATR 浮盈) 自动将止损拉升至开仓价保本
    ):
        self.default_trailing_mult = to_decimal(default_trailing_mult)
        self.breakeven_trigger_r = to_decimal(breakeven_trigger_r)
        self._positions: Dict[str, TrailingPositionState] = {}

    def register_position(
        self,
        symbol: str,
        side: PositionSide | str,
        entry_price: Decimal | float | str,
        initial_stop_loss: Optional[Decimal | float | str] = None,
        take_profit: Optional[Decimal | float | str] = None,
        atr: Optional[Decimal | float | str] = None,
        trailing_mult: Optional[float] = None,
    ) -> TrailingPositionState:
        """注册或更新持仓追踪生命周期."""
        canonical_symbol = normalize_symbol(symbol)
        pos_side = PositionSide(side) if isinstance(side, str) else side
        entry = to_decimal(entry_price)
        stop = to_decimal(initial_stop_loss) if initial_stop_loss is not None else ZERO
        tp = to_decimal(take_profit) if take_profit is not None else None
        atr_val = to_decimal(atr) if atr is not None and to_decimal(atr) > ZERO else (entry * Decimal("0.02"))
        mult = to_decimal(trailing_mult) if trailing_mult is not None else self.default_trailing_mult

        state = TrailingPositionState(
            symbol=canonical_symbol,
            side=pos_side,
            entry_price=entry,
            highest_price=entry,
            lowest_price=entry,
            current_stop_loss=stop,
            take_profit=tp,
            atr=atr_val,
            trailing_mult=mult,
            breakeven_triggered=False,
            chandelier_active=False,
        )
        self._positions[canonical_symbol] = state
        logger.info(
            "Registered trailing state for %s (%s) @ %s (Initial Stop=%s, ATR=%s)",
            canonical_symbol, pos_side.value, entry, stop, atr_val
        )
        return state

    def update_price(
        self,
        symbol: str,
        current_price: Decimal | float | str,
        atr: Optional[Decimal | float | str] = None,
    ) -> TrailingStopSignal:
        """根据最新市价更新持仓状态并判定是否触发止盈止损."""
        canonical_symbol = normalize_symbol(symbol)
        state = self._positions.get(canonical_symbol)
        price = to_decimal(current_price)

        if not state or price <= ZERO:
            return TrailingStopSignal(
                should_close=False,
                symbol=canonical_symbol,
                trigger_type="none",
                current_price=price,
                effective_stop_price=ZERO,
                reason="Position not tracked or invalid price",
            )

        if atr is not None and to_decimal(atr) > ZERO:
            state.atr = to_decimal(atr)

        is_long = (state.side == PositionSide.LONG)

        # 1. 更新极值价格 (High-water mark / Low-water mark)
        if is_long:
            if price > state.highest_price:
                state.highest_price = price
        else:
            if price < state.lowest_price or state.lowest_price == ZERO:
                state.lowest_price = price

        # 2. 检查保本机制 (Break-Even Trigger): 浮盈达到 1R 时，将止损拉至开仓价
        be_distance = state.atr * self.breakeven_trigger_r
        if not state.breakeven_triggered:
            if is_long and (price - state.entry_price >= be_distance):
                state.breakeven_triggered = True
                state.current_stop_loss = max(state.current_stop_loss, state.entry_price)
                logger.info(
                    "Auto Break-Even activated for LONG %s! Stop loss raised to entry: %s",
                    canonical_symbol, state.entry_price
                )
            elif not is_long and (state.entry_price - price >= be_distance):
                state.breakeven_triggered = True
                if state.current_stop_loss > ZERO:
                    state.current_stop_loss = min(state.current_stop_loss, state.entry_price)
                else:
                    state.current_stop_loss = state.entry_price
                logger.info(
                    "Auto Break-Even activated for SHORT %s! Stop loss lowered to entry: %s",
                    canonical_symbol, state.entry_price
                )

        # 3. 计算吊灯追踪止损 (Chandelier Exit)
        chandelier_dist = state.atr * state.trailing_mult
        if is_long:
            chandelier_stop = state.highest_price - chandelier_dist
            if chandelier_stop > state.current_stop_loss:
                state.current_stop_loss = chandelier_stop
                state.chandelier_active = True
        else:
            chandelier_stop = state.lowest_price + chandelier_dist
            if state.current_stop_loss == ZERO or chandelier_stop < state.current_stop_loss:
                state.current_stop_loss = chandelier_stop
                state.chandelier_active = True

        # 4. 判定是否触发硬止盈 (Take-Profit Hit)
        if state.take_profit is not None and state.take_profit > ZERO:
            if is_long and price >= state.take_profit:
                return TrailingStopSignal(
                    should_close=True,
                    symbol=canonical_symbol,
                    trigger_type="take_profit",
                    current_price=price,
                    effective_stop_price=state.take_profit,
                    reason=f"Target Take-Profit {state.take_profit} reached (Current={price})",
                )
            elif not is_long and price <= state.take_profit:
                return TrailingStopSignal(
                    should_close=True,
                    symbol=canonical_symbol,
                    trigger_type="take_profit",
                    current_price=price,
                    effective_stop_price=state.take_profit,
                    reason=f"Target Take-Profit {state.take_profit} reached (Current={price})",
                )

        # 5. 判定是否触发止损 (Chandelier / Hard / Breakeven Stop Hit)
        if state.current_stop_loss > ZERO:
            if is_long and price <= state.current_stop_loss:
                trigger_type = "chandelier_stop" if state.chandelier_active else (
                    "breakeven_stop" if state.breakeven_triggered else "hard_stop"
                )
                return TrailingStopSignal(
                    should_close=True,
                    symbol=canonical_symbol,
                    trigger_type=trigger_type,
                    current_price=price,
                    effective_stop_price=state.current_stop_loss,
                    reason=f"Stop Loss hit at {state.current_stop_loss} ({trigger_type}, Current={price})",
                )
            elif not is_long and price >= state.current_stop_loss:
                trigger_type = "chandelier_stop" if state.chandelier_active else (
                    "breakeven_stop" if state.breakeven_triggered else "hard_stop"
                )
                return TrailingStopSignal(
                    should_close=True,
                    symbol=canonical_symbol,
                    trigger_type=trigger_type,
                    current_price=price,
                    effective_stop_price=state.current_stop_loss,
                    reason=f"Stop Loss hit at {state.current_stop_loss} ({trigger_type}, Current={price})",
                )

        return TrailingStopSignal(
            should_close=False,
            symbol=canonical_symbol,
            trigger_type="none",
            current_price=price,
            effective_stop_price=state.current_stop_loss,
            reason="Holding position within trailing boundary",
        )

    def remove_position(self, symbol: str) -> None:
        """移除平仓后的追踪标的."""
        canonical_symbol = normalize_symbol(symbol)
        self._positions.pop(canonical_symbol, None)

    def get_position_state(self, symbol: str) -> Optional[TrailingPositionState]:
        """获取指定标的的追踪状态."""
        canonical_symbol = normalize_symbol(symbol)
        return self._positions.get(canonical_symbol)
