"""AI 决策引擎：周期循环、LLM 调用、OMS 执行."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Protocol

from xmlx_vlm.ai_trader.config import DEFAULT_API_KEY, DEFAULT_MODEL, DEFAULT_SERVER_URL
from xmlx_vlm.ai_trader.decision.context import TradingContext, TradingStats
from xmlx_vlm.ai_trader.decision.decision import Decision, FullDecision
from xmlx_vlm.ai_trader.decision.prompt_builder import PromptBuilder
from xmlx_vlm.ai_trader.market_service.models import MarketSummary
from xmlx_vlm.ai_trader.oms.config.settings import OMSSettings
from xmlx_vlm.ai_trader.oms.constants import OrderSide
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO
from xmlx_vlm.ai_trader.store.base import DecisionStore, EquitySnapshot
from xmlx_vlm.ai_trader.tools.market import MarketDataTool

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """LLM 完成接口."""

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass
class DecisionEngineConfig:
    """决策引擎配置."""

    trader_id: str
    scan_interval_seconds: int = 300
    prompt_variant: str = "default"
    max_positions: int = 3
    min_confidence: int = 60
    default_leverage: int = 3
    candidate_symbols: Optional[List[str]] = None
    enable_equity_snapshot: bool = True
    # 本地推理配置（默认值来自 xmlx_vlm.ai_trader.config）
    server_url: str = DEFAULT_SERVER_URL
    api_key: Optional[str] = DEFAULT_API_KEY
    model_path: Optional[str] = DEFAULT_MODEL
    temperature: float = 0.3
    max_tokens: int = 2048
    allow_mlx_fallback: bool = True


class DecisionEngine:
    """趋势型 AI 决策引擎."""

    def __init__(
        self,
        oms: OMSEngine,
        config: DecisionEngineConfig,
        store: DecisionStore,
        llm_client: LLMClient,
        market_data: Optional[MarketDataTool] = None,
    ):
        self.oms = oms
        self.config = config
        self.store = store
        self.llm_client = llm_client
        self.market_data = market_data or MarketDataTool()
        self.prompt_builder = PromptBuilder(variant=config.prompt_variant)

        self._cycle_count = 0
        self._start_time = datetime.now(timezone.utc)
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    @property
    def trader_id(self) -> str:
        return self.config.trader_id

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.is_running:
            logger.warning("DecisionEngine %s already running", self.trader_id)
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("DecisionEngine %s started", self.trader_id)

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
        logger.info("DecisionEngine %s stopped", self.trader_id)

    async def emergency_stop(self, flatten: bool = True) -> None:
        await self.stop()
        await self.oms.emergency_stop(flatten=flatten)

    async def run_cycle(self) -> FullDecision:
        """执行一个完整决策周期."""
        self._cycle_count += 1
        cycle_start = time.time()

        # 1. 同步账户与持仓
        await self.oms.sync()

        # 2. 构建上下文
        context = await self._build_context()

        # 3. 生成 prompt
        prompts = self.prompt_builder.build(context)

        # 4. 调用 LLM
        llm_start = time.time()
        try:
            raw_response = await self.llm_client.complete(
                prompts.system_prompt, prompts.user_prompt
            )
        except Exception as exc:
            logger.exception("LLM call failed for %s", self.trader_id)
            raw_response = f"[LLM_ERROR: {exc}]"

        latency_ms = int((time.time() - llm_start) * 1000)

        # 5. 解析决策
        decisions = self._parse_decisions(raw_response)

        # 6. 过滤低 confidence（只针对开仓/平仓决策，保留 hold/wait）
        decisions = [
            d for d in decisions
            if d.action in ("hold", "wait") or d.confidence >= self.config.min_confidence
        ]

        # 7. 执行决策
        execution_results = await self._execute_decisions(decisions)

        # 8. 保存权益快照
        if self.config.enable_equity_snapshot:
            await self._save_equity_snapshot()

        # 9. 组装完整决策记录
        full = FullDecision(
            trader_id=self.trader_id,
            cycle_number=self._cycle_count,
            timestamp=datetime.now(timezone.utc),
            system_prompt=prompts.system_prompt,
            user_prompt=prompts.user_prompt,
            cot_trace=self._extract_cot(raw_response),
            raw_response=raw_response,
            decisions=decisions,
            latency_ms=latency_ms,
        )
        self.store.save_decision(full)

        total_ms = int((time.time() - cycle_start) * 1000)
        logger.info(
            "Cycle %d for %s completed in %dms, decisions=%d, executed=%d",
            self._cycle_count,
            self.trader_id,
            total_ms,
            len(decisions),
            sum(1 for r in execution_results if r.get("executed")),
        )
        return full

    async def _run_loop(self) -> None:
        interval = max(10, self.config.scan_interval_seconds)
        while not self._stop_event.is_set():
            try:
                await self.run_cycle()
            except Exception:
                logger.exception("Decision cycle failed for %s", self.trader_id)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    async def _build_context(self) -> TradingContext:
        portfolio_summary = self.oms.portfolio_summary()
        account = self.oms.portfolio.account
        positions = self.oms.portfolio.list_positions()

        symbols = self.config.candidate_symbols or ["BTC/USDC", "ETH/USDC"]
        market_data: Dict[str, MarketSummary] = {}
        for symbol in symbols:
            try:
                summary = self.market_data.get_summary_object(symbol)
                if summary is not None:
                    market_data[symbol] = summary
            except Exception as exc:
                logger.debug("Market data fetch failed for %s: %s", symbol, exc)

        runtime = datetime.now(timezone.utc) - self._start_time
        return TradingContext(
            current_time=datetime.now(timezone.utc).isoformat(),
            runtime_minutes=int(runtime.total_seconds() // 60),
            cycle_number=self._cycle_count,
            trader_id=self.trader_id,
            account=account,
            positions=positions,
            candidate_symbols=symbols,
            market_data=market_data,
            trading_stats=TradingStats(),  # P0 留空，P2 从 store 计算
            recent_orders=[],
            prompt_variant=self.config.prompt_variant,
        )

    def _parse_decisions(self, raw_response: str) -> List[Decision]:
        """从 LLM 输出中解析 JSON 决策数组，支持各种格式和缺失括号的容错."""
        text = raw_response.strip()
        if not text:
            return []

        # 尝试提取 Markdown 代码块
        code_block = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if code_block:
            text = code_block.group(1).strip()

        # 尝试直接解析
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                data = [data]
            if isinstance(data, list):
                return self._parse_items(data)
        except json.JSONDecodeError:
            pass

        # 容错 1：尝试提取方括号数组
        array_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", text)
        if array_match:
            try:
                data = json.loads(array_match.group(0))
                if isinstance(data, list):
                    return self._parse_items(data)
            except json.JSONDecodeError:
                pass

        # 容错 2：如果是一个或多个以逗号/换行分隔的独立 JSON 对象（缺失外层方括号），尝试包装后解析
        try:
            wrapped = f"[{text}]"
            data = json.loads(wrapped)
            if isinstance(data, list):
                return self._parse_items(data)
        except json.JSONDecodeError:
            pass

        # 容错 3：利用大括号扫描，逐个提取 JSON 对象
        decisions: List[Decision] = []
        brackets = []
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if not brackets:
                    start_idx = i
                brackets.append('{')
            elif char == '}':
                if brackets:
                    brackets.pop()
                    if not brackets and start_idx != -1:
                        obj_str = text[start_idx:i+1]
                        try:
                            item = json.loads(obj_str)
                            if isinstance(item, dict):
                                decisions.append(Decision.from_dict(item))
                        except Exception:
                            pass
        return decisions

    def _parse_items(self, data: List[Any]) -> List[Decision]:
        decisions: List[Decision] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                decisions.append(Decision.from_dict(item))
            except Exception as exc:
                logger.warning("Invalid decision item: %s (%s)", item, exc)
        return decisions

    def _extract_cot(self, raw_response: str) -> str:
        """尝试提取 <think>...</think> 中的 CoT."""
        match = re.search(r"<think>([\s\S]*?)</think>", raw_response)
        return match.group(1).strip() if match else ""

    async def _execute_decisions(self, decisions: List[Decision]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for decision in decisions:
            result: Dict[str, Any] = {"decision": decision.to_dict(), "executed": False}
            try:
                if decision.is_open:
                    order = await self._open_position(decision)
                    result["order"] = order.to_dict() if order else None
                    result["executed"] = order is not None
                elif decision.is_close:
                    order = await self._close_position(decision)
                    result["order"] = order.to_dict() if order else None
                    result["executed"] = order is not None
                else:
                    result["executed"] = True  # hold / wait 视为已处理
            except Exception as exc:
                logger.warning("Decision execution failed: %s", exc)
                result["error"] = str(exc)
            results.append(result)
        return results

    async def _open_position(self, decision: Decision) -> Optional[Order]:
        if decision.position_size_usd is None or decision.position_size_usd <= ZERO:
            logger.warning("Skipping open decision without position_size_usd")
            return None

        # 仓位数量估算：size_usd / mark_price
        summary = self.market_data.get_summary_object(decision.symbol)
        if summary is None:
            logger.warning("No market data for %s, cannot size order", decision.symbol)
            return None
        mark_price = to_decimal(summary.mark_price)
        if mark_price <= ZERO:
            return None
        qty = decision.position_size_usd / mark_price

        side = OrderSide.BUY if decision.side == "buy" else OrderSide.SELL
        order_type = "limit" if decision.price else "market"
        order = self.oms.create_order(
            symbol=decision.symbol,
            side=side.value,
            qty=qty,
            order_type=order_type,
            price=decision.price,
        )
        result = await self.oms.submit_order(order, mark_price=mark_price)
        if result.get("status") in ("submitted", "dry_run"):
            return order
        return None

    async def _close_position(self, decision: Decision) -> Optional[Order]:
        position = self.oms.portfolio.get_position(decision.symbol)
        if not position or position.is_flat():
            logger.info("No position to close for %s", decision.symbol)
            return None

        # 判断关闭方向是否与持仓方向匹配
        close_side = "buy" if decision.action == "close_short" else "sell"
        if (close_side == "buy" and not position.is_short()) or (
            close_side == "sell" and not position.is_long()
        ):
            logger.warning(
                "Close side mismatch: %s vs position %s", close_side, position.side
            )
            return None

        summary = self.market_data.get_summary_object(decision.symbol)
        mark_price = to_decimal(summary.mark_price) if summary else ZERO

        qty = position.qty
        if decision.position_size_usd is not None and decision.position_size_usd > ZERO and mark_price > ZERO:
            target_qty = decision.position_size_usd / mark_price
            if target_qty < position.qty:
                qty = target_qty

        order = self.oms.create_order(
            symbol=decision.symbol,
            side=close_side,
            qty=qty,
            order_type="market",
        )
        result = await self.oms.submit_order(order)
        if result.get("status") in ("submitted", "dry_run"):
            return order
        return None

    async def _save_equity_snapshot(self) -> None:
        summary = self.oms.portfolio_summary()
        account = summary.get("account", {})
        snapshot = EquitySnapshot(
            trader_id=self.trader_id,
            timestamp_ms=int(time.time() * 1000),
            total_equity=to_decimal(account.get("equity", "0")),
            available_margin=to_decimal(account.get("available_margin", "0")),
            unrealized_pnl=to_decimal(summary.get("unrealized_pnl", "0")),
            realized_pnl=to_decimal(summary.get("realized_pnl", "0")),
            margin_used_pct=to_decimal(account.get("margin_utilization_pct", "0")),
            position_count=len(self.oms.portfolio.list_positions()),
        )
        self.store.save_equity_snapshot(snapshot)
