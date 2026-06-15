"""网格策略引擎."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.oms.constants import OrderSide
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO
from xmlx_vlm.ai_trader.store.base import DecisionStore, EquitySnapshot
from xmlx_vlm.ai_trader.strategies.grid.grid_state import GridState
from xmlx_vlm.ai_trader.tools.market import MarketDataTool

logger = logging.getLogger(__name__)


@dataclass
class GridEngineConfig:
    """网格引擎配置."""

    trader_id: str
    symbol: str
    upper_price: Decimal
    lower_price: Decimal
    grid_count: int = 5
    total_investment: Decimal = Decimal("1000")
    max_drawdown_pct: Decimal = Decimal("5")
    daily_loss_limit_pct: Decimal = Decimal("2")
    scan_interval_seconds: int = 60
    enable_equity_snapshot: bool = True


class GridEngine:
    """等差网格策略引擎."""

    def __init__(
        self,
        oms: OMSEngine,
        config: GridEngineConfig,
        store: DecisionStore,
        market_data: Optional[MarketDataTool] = None,
    ):
        self.oms = oms
        self.config = config
        self.store = store
        self.market_data = market_data or MarketDataTool()
        self.state = GridState(
            symbol=config.symbol,
            upper_price=config.upper_price,
            lower_price=config.lower_price,
            grid_count=config.grid_count,
            total_investment=config.total_investment,
            max_drawdown_pct=config.max_drawdown_pct,
            daily_loss_limit_pct=config.daily_loss_limit_pct,
        )

        self._cycle_count = 0
        self._start_time = datetime.now(timezone.utc)
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._active_orders: Dict[str, Order] = {}

    @property
    def trader_id(self) -> str:
        return self.config.trader_id

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("GridEngine %s started", self.trader_id)

    async def stop(self) -> None:
        if not self.is_running:
            return
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("GridEngine %s stopped", self.trader_id)

    async def emergency_stop(self, flatten: bool = True) -> None:
        await self.stop()
        await self._cancel_all_grid_orders()
        if flatten:
            await self.oms.emergency_stop(flatten=True)
        else:
            await self.oms.emergency_stop(flatten=False)

    async def run_cycle(self) -> Dict[str, Any]:
        """执行一个网格周期."""
        self._cycle_count += 1
        await self.oms.sync()

        if self.state.is_paused:
            return {"status": "paused", "cycle": self._cycle_count}

        summary = self.market_data.get_summary_object(self.config.symbol)
        if summary is None:
            logger.warning("No market data for grid %s", self.trader_id)
            return {"status": "no_data", "cycle": self._cycle_count}

        price = to_decimal(summary.mark_price)

        # 风控检查
        portfolio_summary = self.oms.portfolio_summary()
        account = portfolio_summary.get("account", {})
        equity = to_decimal(account.get("equity", "0"))

        self.state.reset_daily(datetime.now(timezone.utc).strftime("%Y-%m-%d"))

        if self.state.check_max_drawdown(equity):
            logger.warning("Grid %s max drawdown reached, emergency exit", self.trader_id)
            await self.emergency_stop(flatten=True)
            return {"status": "max_drawdown", "cycle": self._cycle_count}

        if self.state.check_daily_loss_limit():
            logger.warning("Grid %s daily loss limit reached, pausing", self.trader_id)
            self.state.is_paused = True
            await self._cancel_all_grid_orders()
            return {"status": "daily_loss_limit", "cycle": self._cycle_count}

        breakout = self.state.check_breakout(price)
        if breakout:
            logger.warning("Grid %s breakout detected: %s", self.trader_id, breakout)
            # P1：突破后暂停网格，等待人工或下一周期重新评估
            self.state.is_paused = True
            self.state.breakout_direction = breakout
            await self._cancel_all_grid_orders()
            return {"status": "breakout", "direction": breakout, "cycle": self._cycle_count}

        # 挂网格单
        await self._manage_grid_orders(price)

        if self.config.enable_equity_snapshot:
            await self._save_equity_snapshot()

        return {
            "status": "ok",
            "cycle": self._cycle_count,
            "price": str(price),
            "levels": len(self.state.levels),
        }

    async def _run_loop(self) -> None:
        interval = max(5, self.config.scan_interval_seconds)
        while not self._stop_event.is_set():
            try:
                await self.run_cycle()
            except Exception:
                logger.exception("Grid cycle failed for %s", self.trader_id)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def _manage_grid_orders(self, current_price: Decimal) -> None:
        """在当前价上下挂网格限价单."""
        # 计算每格名义金额
        per_level_notional = self.config.total_investment / Decimal(self.state.grid_count or 1)
        if per_level_notional <= ZERO:
            return

        # 取消远离当前价的旧单
        await self._cancel_distant_orders(current_price)

        # 在当前价下方最近的档位挂买单，上方挂卖单
        for level in self.state.levels:
            if level.price >= current_price:
                continue
            if level.buy_order_id and level.buy_order_id in self._active_orders:
                continue
            qty = per_level_notional / level.price
            order = self.oms.create_order(
                symbol=self.config.symbol,
                side="buy",
                qty=qty,
                order_type="limit",
                price=level.price,
            )
            result = await self.oms.submit_order(order)
            if result.get("status") in ("submitted", "dry_run"):
                level.buy_order_id = order.client_order_id
                self._active_orders[order.client_order_id] = order

        for level in self.state.levels:
            if level.price <= current_price:
                continue
            if level.sell_order_id and level.sell_order_id in self._active_orders:
                continue
            qty = per_level_notional / level.price
            order = self.oms.create_order(
                symbol=self.config.symbol,
                side="sell",
                qty=qty,
                order_type="limit",
                price=level.price,
            )
            result = await self.oms.submit_order(order)
            if result.get("status") in ("submitted", "dry_run"):
                level.sell_order_id = order.client_order_id
                self._active_orders[order.client_order_id] = order

    async def _cancel_distant_orders(self, current_price: Decimal) -> None:
        """取消偏离当前价太远的旧限价单."""
        # 简化：取消所有不在当前相邻网格区间内的未成交单
        keep_ids = set()
        nearest = self.state.level_for_price(current_price)
        if nearest:
            keep_ids.add(nearest.buy_order_id)
            keep_ids.add(nearest.sell_order_id)

        to_cancel = [
            cid for cid, order in self._active_orders.items()
            if cid not in keep_ids and not order.state.is_done()
        ]
        for cid in to_cancel:
            await self.oms.cancel_order(cid)
            self._active_orders.pop(cid, None)

    async def _cancel_all_grid_orders(self) -> None:
        """取消所有网格单."""
        for cid, order in list(self._active_orders.items()):
            if not order.state.is_done():
                await self.oms.cancel_order(cid)
        self._active_orders.clear()
        for level in self.state.levels:
            level.buy_order_id = None
            level.sell_order_id = None

    async def _save_equity_snapshot(self) -> None:
        summary = self.oms.portfolio_summary()
        account = summary.get("account", {})
        snapshot = EquitySnapshot(
            trader_id=self.trader_id,
            timestamp_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
            total_equity=to_decimal(account.get("equity", "0")),
            available_margin=to_decimal(account.get("available_margin", "0")),
            unrealized_pnl=to_decimal(summary.get("unrealized_pnl", "0")),
            realized_pnl=to_decimal(summary.get("realized_pnl", "0")),
            margin_used_pct=to_decimal(account.get("margin_utilization_pct", "0")),
            position_count=len(self.oms.portfolio.list_positions()),
        )
        self.store.save_equity_snapshot(snapshot)
