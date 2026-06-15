"""自主决策闭环.

AutonomousAgentLoop 订阅市场事件，评估信号，生成交易提案，
根据当前模式决定：仅记录、报告人类、请求确认或直接执行。
"""

from __future__ import annotations

import asyncio
import logging
import uuid
import json
import requests
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from xmlx_vlm.ai_trader.agent.config import AgentObjective
from xmlx_vlm.ai_trader.agent.decision import (
    ActionType,
    AgentDecision,
    SignalEvaluation,
    TradeProposal,
)
from xmlx_vlm.ai_trader.agent.evaluator import SignalEvaluator, LLMSignalEvaluator
from xmlx_vlm.ai_trader.config import DEFAULT_API_KEY, DEFAULT_MODEL, DEFAULT_SERVER_URL
from xmlx_vlm.ai_trader.agent.explainability import ExplainabilityBuilder
from xmlx_vlm.ai_trader.agent.governance import ModelGovernance
from xmlx_vlm.ai_trader.agent.modes import AgentMode, ModeController
from xmlx_vlm.ai_trader.market_service.events import IndicatorAlertEvent
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO

logger = logging.getLogger(__name__)

AgentReporter = Callable[[AgentDecision], None]


class AutonomousAgentLoop:
    """Agent 自主决策循环."""

    def __init__(
        self,
        oms: OMSEngine,
        objective: AgentObjective,
        governance: Optional[ModelGovernance] = None,
        mode_controller: Optional[ModeController] = None,
        reporter: Optional[Callable[[AgentDecision], None]] = None,
        price_provider: Optional[Callable[[str], Decimal]] = None,
        atr_provider: Optional[Callable[[str], Optional[Decimal]]] = None,
        evaluator: Optional[SignalEvaluator] = None,
    ):
        self.oms = oms
        self.objective = objective
        self.evaluator = evaluator or SignalEvaluator(objective)
        self.explain_builder = ExplainabilityBuilder(objective)
        self.governance = governance or ModelGovernance()
        self.mode_controller = mode_controller or ModeController(
            initial_mode=AgentMode.OBSERVE,
            event_bus=oms.event_bus if oms else None,
        )
        self.reporter = reporter
        self.price_provider = price_provider
        self.atr_provider = atr_provider

        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._alert_queue: asyncio.Queue[IndicatorAlertEvent] = asyncio.Queue()
        self._handlers: List[Callable[[IndicatorAlertEvent], None]] = []

    # ── 生命周期 ──
    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.is_running:
            logger.warning("AutonomousAgentLoop already running")
            return
        self._stop_event.clear()
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._run())
        logger.info("AutonomousAgentLoop started in %s mode", self.mode_controller.mode.value)

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
        self._loop = None
        logger.info("AutonomousAgentLoop stopped")

    async def emergency_stop(self, reason: str = "agent emergency stop") -> None:
        self.mode_controller.kill(reason=reason, triggered_by="agent")
        await self.stop()
        try:
            await self.oms.emergency_stop(flatten=True)
        except Exception:
            logger.exception("OMS emergency stop failed")

    # ── 事件处理 ──
    def on_alert(self, event: IndicatorAlertEvent) -> None:
        """事件总线回调，将同步事件线程安全地投递到异步队列."""
        loop = self._loop
        if loop is None:
            logger.warning("Agent loop not started, dropping %s alert", event.symbol)
            return
        try:
            loop.call_soon_threadsafe(self._alert_queue.put_nowait, event)
        except Exception:
            logger.warning("Agent alert queue full or closed, dropping %s alert", event.symbol)

    def register_alert_handler(self, handler: Callable[[IndicatorAlertEvent], None]) -> None:
        self._handlers.append(handler)

    # ── 主循环 ──
    async def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = await asyncio.wait_for(
                    self._alert_queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            try:
                await self._handle_alert(event)
            except Exception:
                logger.exception("Agent failed to handle alert for %s", event.symbol)

    async def _handle_alert(self, event: IndicatorAlertEvent) -> None:
        if self.mode_controller.is_killed:
            logger.info("Kill switch active; ignoring alert %s", event.symbol)
            return

        symbol = event.symbol.upper()
        mark_price = await self._get_price(symbol)
        if mark_price is None or mark_price <= ZERO:
            logger.warning("No mark price for %s, skipping alert", symbol)
            return

        atr = await self._get_atr(symbol)
        await self.oms.sync()
        portfolio_summary = self.oms.portfolio_summary()
        equity = to_decimal(portfolio_summary.get("account", {}).get("equity", "0"))

        # 1. 信号评估
        evaluation = self.evaluator.evaluate(
            event=event,
            mark_price=mark_price,
            atr=atr,
            portfolio_summary=portfolio_summary,
        )

        # 2. 生成交易提案
        proposal = self.evaluator.build_proposal(
            evaluation=evaluation,
            mark_price=mark_price,
            equity=equity,
            variant_id="default",
        )

        # 3. 记录 shadow（如果变体不止默认）
        if proposal is not None and len(self.governance.registry.list_variants()) > 1:
            self.governance.record_shadow(
                variant_id=proposal.variant_id,
                symbol=symbol,
                proposal=proposal,
                mark_price=mark_price,
            )

        # 4. 模式驱动执行
        decision_id = uuid.uuid4().hex[:16]
        decision = AgentDecision(
            decision_id=decision_id,
            symbol=symbol,
            proposal=proposal or TradeProposal(
                action=ActionType.WAIT,
                symbol=symbol,
                reason="no proposal generated",
                variant_id="default",
            ),
            evaluation=evaluation,
            mode=self.mode_controller.mode.value,
            variant_id="default",
        )

        if proposal is None:
            decision.rationale = self.explain_builder.build(
                evaluation, None, should_execute=False
            ).to_dict()
            decision.executed = False
            self._emit_decision(decision)
            return

        should_execute = self.mode_controller.can_execute(proposal)
        rejected_reason: Optional[str] = None

        if self.mode_controller.needs_human_confirm(proposal):
            confirmed = self.mode_controller.request_human_confirm(proposal)
            if confirmed:
                should_execute = True
            else:
                should_execute = False
                rejected_reason = "human declined"

        execution_result: Optional[Dict[str, Any]] = None
        if should_execute:
            execution_result = await self._execute_proposal(proposal)
            if execution_result and execution_result.get("status") in ("submitted", "dry_run"):
                decision.executed = True
                if proposal.is_close:
                    asyncio.create_task(self._run_post_trade_reflection(proposal, execution_result))
            else:
                should_execute = False
                rejected_reason = execution_result.get("error") if execution_result else "execution failed"

        decision.rationale = self.explain_builder.build(
            evaluation,
            proposal,
            should_execute=decision.executed,
            rejected_reason=rejected_reason,
        ).to_dict()
        decision.execution_result = execution_result
        decision.rejected_reason = rejected_reason

        self._emit_decision(decision)

    async def _execute_proposal(self, proposal: TradeProposal) -> Dict[str, Any]:
        """将 TradeProposal 提交到 OMS."""
        try:
            mark_price = await self._get_price(proposal.symbol)
            if proposal.is_close:
                position = self.oms.portfolio.get_position(proposal.symbol)
                if position is None or position.is_flat():
                    return {"status": "skipped", "reason": "no position to close"}
                close_side = "sell" if position.is_long() else "buy"
                qty = position.qty
                if proposal.size_usd > ZERO and mark_price is not None and mark_price > ZERO:
                    target_qty = proposal.size_usd / mark_price
                    if target_qty < position.qty:
                        qty = target_qty
                order = self.oms.create_order(
                    symbol=proposal.symbol,
                    side=close_side,
                    qty=qty,
                    order_type="market",
                    client_order_id=f"agent-close-{uuid.uuid4().hex[:8]}",
                )
            else:
                if mark_price is None or mark_price <= ZERO:
                    return {"status": "error", "error": "no mark price"}
                qty = proposal.size_usd / mark_price
                order_type = "limit" if proposal.take_profit else "market"
                price = proposal.take_profit if order_type == "limit" else None
                order = self.oms.create_order(
                    symbol=proposal.symbol,
                    side=proposal.side,
                    qty=qty,
                    order_type=order_type,
                    price=price,
                    client_order_id=f"agent-open-{uuid.uuid4().hex[:8]}",
                )
            result = await self.oms.submit_order(order, mark_price=mark_price)
            return result
        except Exception as exc:
            logger.exception("Agent execution failed for %s", proposal.symbol)
            return {"status": "error", "error": str(exc)}

    # ── 数据获取 ──
    async def _get_price(self, symbol: str) -> Optional[Decimal]:
        if self.price_provider is not None:
            try:
                price = self.price_provider(symbol)
                if asyncio.iscoroutine(price):
                    price = await price
                return to_decimal(price) if price is not None else None
            except Exception:
                logger.exception("Price provider failed for %s", symbol)
        try:
            if self.oms and hasattr(self.oms, "market_data_tool") and self.oms.market_data_tool:
                summary = self.oms.market_data_tool.get_summary_object(symbol)
                if summary:
                    return to_decimal(summary.mark_price)
        except Exception:
            logger.debug("No market data tool available for %s", symbol)
        return None

    async def _get_atr(self, symbol: str) -> Optional[Decimal]:
        if self.atr_provider is not None:
            try:
                atr = self.atr_provider(symbol)
                if asyncio.iscoroutine(atr):
                    atr = await atr
                return to_decimal(atr) if atr is not None else None
            except Exception:
                logger.exception("ATR provider failed for %s", symbol)
        return None

    # ── 辅助 ──
    def _emit_decision(self, decision: AgentDecision) -> None:
        self.governance.record_decision(decision)
        if self.reporter:
            try:
                self.reporter(decision)
            except Exception:
                logger.exception("Agent reporter failed")
        for handler in self._handlers:
            try:
                handler(decision)
            except Exception:
                logger.exception("Agent alert handler failed")

    # ── 便捷方法 ──
    def add_variant(
        self,
        name: str,
        description: str,
        prompt_template: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        variant = self.governance.registry.create_variant(
            name=name,
            description=description,
            prompt_template=prompt_template,
            config_overrides=config_overrides or {},
        )
        return variant.variant_id

    async def _run_post_trade_reflection(self, proposal: TradeProposal, execution_result: Dict[str, Any]):
        """Run post-trade reflection using the local LLM server after a trade closes."""
        try:
            # Let's wait a brief moment for the order to be filled/processed in the portfolio
            await asyncio.sleep(2.0)

            # Fetch position or trade log to calculate PnL
            position = self.oms.portfolio.get_position(proposal.symbol)
            pnl = float(position.realized_pnl) if position else 0.0

            # Formulate prompt for local LLM to do post-trade reflection
            prompt = f"""You are a professional trading analyst. Reflect on the outcome of the closed trade.

### Trade Summary:
- Symbol: {proposal.symbol}
- Action: {proposal.action.value}
- Size (USD): {proposal.size_usd}
- Realized PnL: {pnl} USD
- Original Reason: {proposal.reason}

Identify the key lesson learned from this trade (e.g., entry timing, market context, leverage sizing, or rule violation). Keep the lesson concise (under 2 sentences).

Your output MUST be a valid JSON block inside ```json ... ``` code fence with a single field:
```json
{{
  "lesson": "The main lesson learned from this trade."
}}
```
"""
            # Call local LLM server
            headers = {"Content-Type": "application/json"}
            api_key = getattr(self.evaluator, "api_key", DEFAULT_API_KEY)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            server_url = getattr(self.evaluator, "server_url", DEFAULT_SERVER_URL)
            model_name = getattr(self.evaluator, "model_name", DEFAULT_MODEL)

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a professional quantitative contract trading expert."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 150,
                "stream": False
            }

            def call_server():
                return requests.post(
                    f"{server_url.rstrip('/')}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=20.0
                )

            resp = await asyncio.to_thread(call_server)
            lesson = "Check execution entry/exit timing."
            if resp.status_code == 200:
                resp_json = resp.json()
                content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                import re
                match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(1))
                        lesson = data.get("lesson", lesson)
                    except Exception:
                        pass
                else:
                    match_braces = re.search(r"(\{.*\})", content, re.DOTALL)
                    if match_braces:
                        try:
                            data = json.loads(match_braces.group(1))
                            lesson = data.get("lesson", lesson)
                        except Exception:
                            pass

            # Save reflection to DB
            db = getattr(self.evaluator, "db", None)
            if db:
                db.add_reflection(
                    symbol=proposal.symbol,
                    pnl=pnl,
                    trade_details=proposal.to_dict(),
                    lesson=lesson
                )
                logger.info("Saved trade reflection for %s: %s", proposal.symbol, lesson)
        except Exception as e:
            logger.exception("Failed to run post-trade reflection: %s", e)

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": self.mode_controller.mode.value,
            "killed": self.mode_controller.is_killed,
            "objective": self.objective.to_dict(),
            "variant_count": len(self.governance.registry.list_variants()),
            "decision_count": len(self.governance.list_decisions()),
            "shadow_count": len(self.governance.list_shadow_records()),
            "best_variant": self.governance.best_variant(),
        }
