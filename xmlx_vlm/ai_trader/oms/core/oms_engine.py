"""OMS 编排引擎."""

from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.oms.audit.events import AuditEvent
from xmlx_vlm.ai_trader.oms.audit.auditor import Auditor
from xmlx_vlm.ai_trader.oms.circuit.api_error_breaker import ApiErrorCircuitBreaker
from xmlx_vlm.ai_trader.oms.circuit.consecutive_loss_breaker import ConsecutiveLossCircuitBreaker
from xmlx_vlm.ai_trader.oms.circuit.daily_loss_breaker import DailyLossCircuitBreaker
from xmlx_vlm.ai_trader.oms.circuit.kill_switch import KillSwitch
from xmlx_vlm.ai_trader.oms.config.settings import OMSSettings
from xmlx_vlm.ai_trader.oms.constants import AuditEventType, EventType, OrderSide, OrderState, RiskDecisionType
from xmlx_vlm.ai_trader.oms.core.account import AccountSnapshot
from xmlx_vlm.ai_trader.oms.core.order import Fill, Order
from xmlx_vlm.ai_trader.oms.core.portfolio import Portfolio
from xmlx_vlm.ai_trader.oms.core.trade import Trade
from xmlx_vlm.ai_trader.oms.events.bus import EventBus, SyncEventBus
from xmlx_vlm.ai_trader.oms.events.types import (
    CircuitEvent,
    FillEvent,
    KillSwitchEvent,
    OrderEvent,
    PortfolioEvent,
    RiskEvent,
)
from xmlx_vlm.ai_trader.oms.exceptions import (
    CircuitTrippedError,
    LiveTradingNotEnabledError,
    RiskRejectedError,
)
from xmlx_vlm.ai_trader.oms.execution.algo.base import ParentOrder
from xmlx_vlm.ai_trader.oms.execution.algo.registry import get_algo
from xmlx_vlm.ai_trader.oms.execution.algo.scheduler import AlgoScheduler
from xmlx_vlm.ai_trader.oms.execution.factory import ExecutionAdapterFactory
from xmlx_vlm.ai_trader.oms.impact.market_impact import AlmgrenChrissImpactModel
from xmlx_vlm.ai_trader.oms.market_data.provider import MarketDataProvider
from xmlx_vlm.ai_trader.oms.interfaces.audit_sink import AuditSink
from xmlx_vlm.ai_trader.oms.interfaces.execution_adapter import ExecutionAdapter
from xmlx_vlm.ai_trader.oms.interfaces.risk_engine import RiskContext
from xmlx_vlm.ai_trader.oms.order_sync.factory import create_order_sync_worker
from xmlx_vlm.ai_trader.oms.risk.risk_manager import RiskManager
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO
from xmlx_vlm.ai_trader.oms.utils.time import utc_now_ms

logger = logging.getLogger(__name__)


class OMSEngine:
    """OMS 编排器：订单生命周期、风控、执行、审计、熔断的统一入口."""

    def __init__(
        self,
        settings: OMSSettings,
        adapter: Optional[ExecutionAdapter] = None,
        risk_manager: Optional[RiskManager] = None,
        portfolio: Optional[Portfolio] = None,
        event_bus: Optional[EventBus] = None,
        auditor: Optional[Any] = None,
        market_data_tool=None,
        market_data_provider: Optional[MarketDataProvider] = None,
        order_sync_enabled: bool = False,
        order_sync_interval_seconds: int = 5,
    ):
        self.settings = settings
        self.event_bus = event_bus or SyncEventBus()
        self.portfolio = portfolio or Portfolio()

        # 执行适配器
        # paper / local_sim 是本地仿真机构盘，与 hyperliquid 实盘地位相同
        self._adapter = adapter or ExecutionAdapterFactory.create(
            exchange=settings.exchange,
            market_data_tool=market_data_tool,
            market_data_provider=market_data_provider,
            wallet_address=settings.wallet_address,
            private_key=settings.private_key,
            signer_endpoint=settings.signer_endpoint,
            testnet=settings.testnet,
            timeout=settings.request_timeout,
            fill_slippage_pct=settings.paper_fill_slippage_pct,
            default_price=settings.paper_default_price,
            latency_ms=settings.paper_latency_ms,
        )

        # 市场数据与冲击模型 / 路由
        self._market_data_provider = market_data_provider
        self._impact_model = AlmgrenChrissImpactModel()
        from xmlx_vlm.ai_trader.oms.routing.router import SmartOrderRouter

        self._router = SmartOrderRouter(
            self._adapter,
            self._impact_model,
            default_max_slippage_pct=settings.max_slippage_pct,
        )
        self._algo_scheduler = AlgoScheduler(self._router)

        # 风控
        self.risk_manager = risk_manager or RiskManager.from_profile(
            settings.risk_profile_dict()
        )

        # 审计
        self.auditor = auditor
        if self.auditor is None:
            from xmlx_vlm.ai_trader.oms.audit.sinks.file_sink import FileAuditSink
            from xmlx_vlm.ai_trader.oms.audit.sinks.sqlite_sink import SQLiteAuditSink

            self.auditor = Auditor(
                sinks=[
                    FileAuditSink(settings.audit_log_dir),
                    SQLiteAuditSink(settings.audit_db_path),
                ]
            )

        # 熔断器
        self._daily_loss_breaker = DailyLossCircuitBreaker(settings.max_daily_loss_pct)
        self._api_error_breaker = ApiErrorCircuitBreaker(
            max_errors=settings.max_api_errors
        )
        self._consecutive_loss_breaker = ConsecutiveLossCircuitBreaker(
            max_consecutive_losses=settings.max_consecutive_losses
        )
        self.kill_switch = KillSwitch(self.event_bus)

        # 状态
        self._orders: Dict[str, Order] = {}
        self._starting_equity = ZERO
        self._load_state()
        self._subscribe_events()

        # 后台订单同步（默认关闭）
        self._order_sync_enabled = order_sync_enabled
        self._order_sync_interval_seconds = order_sync_interval_seconds
        self._order_sync_worker = create_order_sync_worker(
            self._adapter, self, self._order_sync_interval_seconds
        )
        if self._order_sync_enabled and self._order_sync_worker is not None:
            # OMSEngine 自身不启动 worker，由上层 StrategyInstance / 用户显式启动
            logger.info("Order sync worker created (not started yet)")

    # ── 公共属性 ──
    @property
    def adapter(self) -> ExecutionAdapter:
        return self._adapter

    @property
    def is_live(self) -> bool:
        """是否连接真实交易所（hyperliquid 实盘）."""
        return self.settings.live_enabled and self._adapter.is_live

    @property
    def venue_type(self) -> str:
        """交易场所类型：live_exchange 或 local_simulation."""
        if self._adapter.is_live:
            return "live_exchange"
        return "local_simulation"

    # ── 订单入口 ──
    def create_order(
        self,
        symbol: str,
        side: str,
        qty: Any,
        order_type: str = "market",
        price: Optional[Any] = None,
        stop_px: Optional[Any] = None,
        time_in_force: str = "GTC",
        client_order_id: Optional[str] = None,
    ) -> Order:
        """创建 DRAFT 订单."""
        if client_order_id is None:
            import uuid
            client_order_id = uuid.uuid4().hex
        return Order(
            symbol=symbol.upper(),
            side=OrderSide(side.lower()),
            order_type=order_type,
            qty=to_decimal(qty),
            price=to_decimal(price) if price is not None else None,
            stop_px=to_decimal(stop_px) if stop_px is not None else None,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )

    async def submit_order(
        self,
        order: Order,
        mark_price: Optional[Any] = None,
        oracle_price: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """提交订单：风控 → 执行 → 更新状态 → 审计."""
        # 0. 全局锁定检查
        if self.kill_switch.is_locked:
            raise CircuitTrippedError("kill_switch", self.kill_switch.reason)

        circuit_reason = self._check_circuits()
        if circuit_reason:
            raise CircuitTrippedError("circuit", circuit_reason)

        # 1. 实盘保护：仅真实交易所（hyperliquid）需要显式启用 live
        # paper / local_sim 是本地仿真机构盘，地位相同，无需 live_enabled
        if self._adapter.is_live and not self.settings.live_enabled:
            raise LiveTradingNotEnabledError(
                "live trading not enabled; set AI_TRADER_LIVE=1 and pass --live"
            )

        # 2. 发布订单创建事件
        self._publish_order_event(order, EventType.ORDER_CREATED)
        self._audit(
            AuditEventType.ORDER_INTENT,
            order,
            payload=order.to_dict(),
        )

        # 3. 事前风控
        context = RiskContext(
            portfolio=self.portfolio,
            mark_price=to_decimal(mark_price) if mark_price else None,
            oracle_price=to_decimal(oracle_price) if oracle_price else None,
            account_equity=self.portfolio.account.equity,
        )
        risk_decision = self.risk_manager.evaluate_pre_trade(order, context)
        self._publish_risk_event(risk_decision, order.client_order_id)
        self._audit(
            AuditEventType.RISK_DECISION,
            order,
            payload=risk_decision.to_dict(),
        )
        if risk_decision.rejected:
            order.transition_to(OrderState.REJECTED, reason=risk_decision.reason)
            self._orders[order.client_order_id] = order
            raise RiskRejectedError(risk_decision.rule_name, risk_decision.reason)

        order.transition_to(OrderState.PRE_TRADE_OK)

        # 4. dry-run 模式
        if self.settings.dry_run:
            return {
                "status": "dry_run",
                "order": order.to_dict(),
                "risk_decision": risk_decision.to_dict(),
            }

        # 5. 提交到交易所
        order.transition_to(OrderState.SUBMITTED)
        self._publish_order_event(order, EventType.ORDER_SUBMITTED)
        try:
            ack = await self._adapter.submit(order)
        except Exception as exc:
            self._api_error_breaker.record_error()
            if not order.is_done():
                order.transition_to(OrderState.REJECTED, reason=str(exc))
            self._orders[order.client_order_id] = order
            self._audit(
                AuditEventType.ORDER_UPDATE,
                order,
                payload={"error": str(exc)},
            )
            raise

        self._orders[order.client_order_id] = order
        self._audit(
            AuditEventType.ORDER_SUBMIT,
            order,
            payload={"ack": {"success": ack.success, "order_id": ack.order_id}},
        )

        # 6. 处理成交
        if order.state == OrderState.FILLED or order.state == OrderState.PARTIAL_FILLED:
            for fill in order.fills:
                self._process_fill(order, fill)

        return {
            "status": "submitted",
            "order": order.to_dict(),
            "ack": {"success": ack.success, "order_id": ack.order_id, "message": ack.message},
        }

    async def cancel_order(self, client_order_id: str) -> Dict[str, Any]:
        order = self._orders.get(client_order_id)
        if order is None:
            return {"status": "error", "message": "order not found"}
        if not order.is_done():
            order.transition_to(OrderState.CANCEL_REQUESTED)
            self._publish_order_event(order, EventType.ORDER_CANCEL_REQUESTED)
        ack = await self._adapter.cancel(client_order_id, client_order_id)
        if ack.success:
            order.transition_to(OrderState.CANCELLED)
            self._publish_order_event(order, EventType.ORDER_CANCEL_ACKED)
        return {"status": "cancelled" if ack.success else "failed", "ack": ack.__dict__}

    async def submit_algo(
        self,
        parent_order: ParentOrder,
        mark_price: Optional[Any] = None,
        oracle_price: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """提交算法单.

        流程：风控（对 parent）→ 启动 AlgoScheduler → 返回 algo_id。
        """
        # 0. 全局锁定检查
        if self.kill_switch.is_locked:
            raise CircuitTrippedError("kill_switch", self.kill_switch.reason)
        circuit_reason = self._check_circuits()
        if circuit_reason:
            raise CircuitTrippedError("circuit", circuit_reason)

        # 1. 实盘保护：仅真实交易所（hyperliquid）需要显式启用 live
        # paper / local_sim 是本地仿真机构盘，地位相同，无需 live_enabled
        if self._adapter.is_live and not self.settings.live_enabled:
            raise LiveTradingNotEnabledError(
                "live trading not enabled; set AI_TRADER_LIVE=1 and pass --live"
            )

        # 2. 用一张临时 Order 做 pre-trade risk 检查
        risk_price = to_decimal(mark_price) if mark_price else (to_decimal(oracle_price) if oracle_price else None)
        risk_order = Order(
            symbol=parent_order.symbol,
            side=parent_order.side,
            qty=parent_order.total_qty,
            order_type="market" if risk_price is None else "limit",
            price=risk_price,
            algo_id=parent_order.order_id,
        )
        context = RiskContext(
            portfolio=self.portfolio,
            mark_price=to_decimal(mark_price) if mark_price else None,
            oracle_price=to_decimal(oracle_price) if oracle_price else None,
            account_equity=self.portfolio.account.equity,
        )
        risk_decision = self.risk_manager.evaluate_pre_trade(risk_order, context)
        self._publish_risk_event(risk_decision, parent_order.order_id)
        self._audit(
            AuditEventType.RISK_DECISION,
            risk_order,
            payload={"parent_order": parent_order.to_dict(), "risk_decision": risk_decision.to_dict()},
        )
        if risk_decision.rejected:
            parent_order.state = OrderState.REJECTED
            parent_order.reject_reason = risk_decision.reason
            raise RiskRejectedError(risk_decision.rule_name, risk_decision.reason)

        # 3. dry-run 模式
        if self.settings.dry_run:
            return {
                "status": "dry_run",
                "parent_order": parent_order.to_dict(),
                "risk_decision": risk_decision.to_dict(),
            }

        # 4. 启动算法
        algo_id = self._algo_scheduler.start_algo(
            parent_order,
            on_child_update=self._on_algo_child_update,
        )
        self._audit(
            AuditEventType.ORDER_SUBMIT,
            None,
            payload={"parent_order": parent_order.to_dict(), "algo_id": algo_id},
        )
        return {
            "status": "started",
            "algo_id": algo_id,
            "parent_order": parent_order.to_dict(),
        }

    async def cancel_algo(self, algo_id: str) -> Dict[str, Any]:
        """取消算法单."""
        success = await self._algo_scheduler.cancel_algo(algo_id)
        return {"status": "cancelled" if success else "not_found", "algo_id": algo_id}

    def get_algo_status(self, algo_id: str) -> Optional[Dict[str, Any]]:
        """获取算法单状态."""
        algo = self._algo_scheduler.get_algo(algo_id)
        parent = self._algo_scheduler.get_parent(algo_id)
        if parent is None:
            return None
        return {
            "algo_id": algo_id,
            "name": algo.name if algo else None,
            "is_done": algo.is_done if algo else True,
            "parent_order": parent.to_dict(),
        }

    def list_algos(self) -> Dict[str, Dict[str, Any]]:
        """列出所有算法单状态."""
        return self._algo_scheduler.list_algos()

    def _on_algo_child_update(self, child: Order) -> None:
        """算法 child 成交/状态变更回调."""
        self._orders[child.client_order_id] = child
        if child.filled_qty > ZERO:
            for fill in child.fills:
                self._process_fill(child, fill)
        self._publish_order_event(child, self._event_type_for_state(child.state))

    def _event_type_for_state(self, state: OrderState) -> EventType:
        mapping = {
            OrderState.SUBMITTED: EventType.ORDER_SUBMITTED,
            OrderState.SENT: EventType.ORDER_SENT,
            OrderState.ACKNOWLEDGED: EventType.ORDER_ACKED,
            OrderState.PARTIAL_FILLED: EventType.ORDER_PARTIAL_FILLED,
            OrderState.FILLED: EventType.ORDER_FILLED,
            OrderState.REJECTED: EventType.ORDER_REJECTED,
            OrderState.CANCEL_REQUESTED: EventType.ORDER_CANCEL_REQUESTED,
            OrderState.CANCELLED: EventType.ORDER_CANCELLED,
            OrderState.EXPIRED: EventType.ORDER_EXPIRED,
        }
        return mapping.get(state, EventType.ORDER_CREATED)

    async def emergency_stop(self, flatten: bool = True) -> Dict[str, Any]:
        """触发急停，可选市价全平."""
        self.kill_switch.trigger(
            triggered_by="user",
            reason="emergency stop requested",
            flatten_positions=flatten,
        )
        self._audit(
            AuditEventType.KILL_SWITCH,
            None,
            payload={"flatten": flatten, "reason": "emergency stop"},
        )

        if flatten and self.settings.auto_flatten_on_kill:
            # 市价平掉所有持仓
            results = []
            for pos in self.portfolio.list_positions():
                close_side = "sell" if pos.is_long() else "buy"
                order = self.create_order(
                    symbol=pos.symbol,
                    side=close_side,
                    qty=pos.qty,
                    order_type="market",
                )
                try:
                    result = await self.submit_order(order)
                    results.append(result)
                except Exception as exc:
                    logger.error("flatten failed for %s: %s", pos.symbol, exc)
                    results.append({"symbol": pos.symbol, "error": str(exc)})
            return {"status": "killed", "flatten_results": results}
        return {"status": "killed"}

    async def close_position(self, symbol: str) -> Optional[Order]:
        """Close an active position for the given symbol, offsetting long/short directions."""
        await self.sync()
        
        target = symbol.upper()
        pos = None
        for p in self.portfolio.list_positions():
            if p.symbol.upper() == target or p.symbol.split('/')[0].upper() == target:
                pos = p
                break
                
        if pos is None or pos.is_flat():
            logger.info("No active position to close for symbol: %s", symbol)
            return None
            
        close_side = "sell" if pos.is_long() else "buy"
        order = self.create_order(
            symbol=pos.symbol,
            side=close_side,
            qty=pos.qty,
            order_type="market",
        )
        
        # Determine mark price for wind-down order validation
        mark_price = None
        if self._market_data_provider:
            try:
                summary = self._market_data_provider.get_summary(pos.symbol)
                if summary:
                    mark_price = Decimal(str(summary.mark_price))
            except Exception:
                pass
                
        if mark_price is None:
            try:
                from xmlx_vlm.ai_trader.tools.market import MarketDataTool
                market = MarketDataTool()
                import re
                text = market.get_ticker(pos.symbol, "hyperliquid")
                match = re.search(r"(?:mark|last)=([\d,]+\.?\d*)", text)
                if match:
                    mark_price = Decimal(match.group(1).replace(",", ""))
            except Exception:
                pass
                
        await self.submit_order(order, mark_price=mark_price)
        return order

    async def sync(self) -> Dict[str, Any]:
        """同步账户、持仓."""
        try:
            positions = await self._adapter.sync_positions()
            account = await self._adapter.sync_account()
            self.portfolio.sync_positions(positions)
            self.portfolio.sync_account(account)
            self._starting_equity = max(self._starting_equity, account.equity)
            self._daily_loss_breaker.update(self._starting_equity, account.equity)
            self.event_bus.publish(
                PortfolioEvent(event_type=EventType.PORTFOLIO_SYNCED, data=self.portfolio.summary())
            )
            return {"status": "synced", "account": account.to_dict(), "positions": len(positions)}
        except Exception as exc:
            self._api_error_breaker.record_error()
            logger.error("oms sync failed: %s", exc)
            return {"status": "error", "message": str(exc)}

    def portfolio_summary(self) -> Dict[str, Any]:
        return self.portfolio.summary()

    def get_order(self, client_order_id: str) -> Optional[Order]:
        return self._orders.get(client_order_id)

    def list_orders(self) -> List[Order]:
        return list(self._orders.values())

    async def start_order_sync(self) -> None:
        """启动后台订单同步."""
        if self._order_sync_worker is not None:
            await self._order_sync_worker.start()

    async def stop_order_sync(self) -> None:
        """停止后台订单同步."""
        if self._order_sync_worker is not None:
            await self._order_sync_worker.stop()

    @property
    def order_sync_worker(self):
        """返回订单同步 worker（可能为 None）."""
        return self._order_sync_worker

    def close(self) -> None:
        if self._order_sync_worker is not None and self._order_sync_worker.is_running:
            import asyncio

            try:
                asyncio.get_running_loop().create_task(self._order_sync_worker.stop())
            except RuntimeError:
                pass
        self._save_state()
        if self.auditor:
            self.auditor.flush()
            self.auditor.close()
        try:
            self._adapter.close()
        except Exception as exc:
            logger.warning("adapter close failed: %s", exc)

    @property
    def router(self) -> "SmartOrderRouter":
        return self._router

    @property
    def algo_scheduler(self) -> AlgoScheduler:
        return self._algo_scheduler

    # ── 内部方法 ──
    def _process_fill(self, order: Order, fill: Fill) -> None:
        trade = Trade(
            trade_id=fill.fill_id,
            order_id=order.order_id or order.client_order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            qty=fill.qty,
            price=fill.price,
            fee=fill.fee,
            timestamp_ms=fill.timestamp_ms,
            exchange=order.exchange,
        )
        self.portfolio.update_with_trade(trade)

        # 事后风控
        post_decision = self.risk_manager.evaluate_post_trade(trade, self.portfolio)
        if post_decision:
            self._publish_risk_event(post_decision, order.client_order_id)
            self._audit(
                AuditEventType.RISK_DECISION,
                order,
                payload=post_decision.to_dict(),
            )

        # 连续亏损熔断
        self._consecutive_loss_breaker.record_trade_pnl(
            self.portfolio.get_position(order.symbol).realized_pnl
            if self.portfolio.get_position(order.symbol)
            else ZERO
        )

        # 账户快照
        account = self.portfolio.account
        self._daily_loss_breaker.update(self._starting_equity, account.equity)
        self._audit(
            AuditEventType.ORDER_FILL,
            order,
            payload={"fill": fill.to_dict(), "trade": trade.to_dict()},
        )

    def _check_circuits(self) -> str:
        for breaker in [
            self._daily_loss_breaker,
            self._api_error_breaker,
            self._consecutive_loss_breaker,
        ]:
            reason = breaker.check()
            if reason:
                self.event_bus.publish(
                    CircuitEvent(
                        event_type=EventType.CIRCUIT_TRIPPED,
                        circuit_name=breaker.name,
                        reason=reason,
                    )
                )
                return reason
        return ""

    def _publish_order_event(self, order: Order, event_type: EventType) -> None:
        self.event_bus.publish(
            OrderEvent(
                event_type=event_type,
                client_order_id=order.client_order_id,
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                state=order.state,
                qty=order.qty,
                filled_qty=order.filled_qty,
                price=order.price,
                reason=order.reject_reason,
            )
        )

    def _publish_risk_event(self, decision, client_order_id: str) -> None:
        event_type = (
            EventType.RISK_PASSED
            if decision.passed
            else EventType.RISK_REJECTED
            if decision.rejected
            else EventType.RISK_WARNING
        )
        self.event_bus.publish(
            RiskEvent(
                event_type=event_type,
                decision=decision.decision,
                rule_name=decision.rule_name,
                client_order_id=client_order_id,
                reason=decision.reason,
                metadata=decision.metadata,
            )
        )

    def _audit(
        self,
        event_type: AuditEventType,
        order: Optional[Order],
        payload: Optional[Dict[str, Any]] = None,
        raw: Optional[Any] = None,
    ) -> None:
        if self.auditor is None:
            return
        event = AuditEvent(
            event_type=event_type,
            client_order_id=order.client_order_id if order else None,
            order_id=order.order_id if order else None,
            symbol=order.symbol if order else None,
            payload=payload or {},
            raw=raw,
        )
        self.auditor.record(event)

    def _subscribe_events(self) -> None:
        # 可在此订阅事件做额外处理（如告警）
        pass

    # ── 持久化 ──
    def _state_path(self) -> Path:
        return self.settings.state_path

    def _load_state(self) -> None:
        path = self._state_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._starting_equity = to_decimal(data.get("starting_equity", "0"))
        except Exception as exc:
            logger.warning("failed to load oms state: %s", exc)

    def _save_state(self) -> None:
        try:
            data = {
                "starting_equity": str(self._starting_equity),
                "saved_at_ms": utc_now_ms(),
            }
            self._state_path().write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as exc:
            logger.warning("failed to save oms state: %s", exc)
