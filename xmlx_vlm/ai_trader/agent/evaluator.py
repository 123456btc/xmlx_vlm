"""信号评估器：置信度、RR、止损、仓位."""

from __future__ import annotations

import logging
import requests
import json
import time
from decimal import Decimal
from typing import Any, Dict, Optional, List

from xmlx_vlm.ai_trader.config import DEFAULT_API_KEY, DEFAULT_SERVER_URL, DEFAULT_MODEL
from xmlx_vlm.ai_trader.store.session_db import QuantSessionDB
from xmlx_vlm.ai_trader.agent.config import AgentObjective
from xmlx_vlm.ai_trader.agent.decision import ActionType, SignalEvaluation, TradeProposal
from xmlx_vlm.ai_trader.market_service.events import IndicatorAlertEvent
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO
from xmlx_vlm.ai_trader.agent.telemetry import QuantTracer

logger = logging.getLogger(__name__)


class SignalEvaluator:
    """评估原始市场信号，转换为结构化交易提案."""

    def __init__(self, objective: AgentObjective):
        self.objective = objective

    def evaluate(
        self,
        event: IndicatorAlertEvent,
        mark_price: Decimal,
        atr: Optional[Decimal] = None,
        portfolio_summary: Optional[Dict[str, Any]] = None,
    ) -> SignalEvaluation:
        """根据技术指标警报生成信号评估.

        这是规则化启发式评估；生产环境可替换为 LLM 或更复杂的模型。
        """
        symbol = event.symbol.upper()
        alert_type = event.alert_type
        payload = event.payload or {}

        # 默认值
        confidence = int(payload.get("confidence", 50))
        direction = str(payload.get("direction", "neutral")).lower()
        stop_loss = self._extract_price(payload, "stop_loss")
        take_profit = self._extract_price(payload, "take_profit")

        notes: list[str] = []
        confidence = self._adjust_confidence(alert_type, payload, confidence, notes)

        if stop_loss is None and atr is not None:
            stop_loss = self._default_stop(mark_price, direction, atr)
            notes.append(f"使用 ATR 默认止损: {stop_loss}")

        if take_profit is None and stop_loss is not None:
            take_profit = self._default_take_profit(mark_price, stop_loss, direction)
            notes.append(f"使用默认止盈: {take_profit}")

        rr = self._compute_rr(mark_price, stop_loss, take_profit, direction)
        expected_return_pct, expected_risk_pct = self._compute_expected_pct(
            mark_price, stop_loss, take_profit, direction
        )

        return SignalEvaluation(
            signal_type=alert_type,
            symbol=symbol,
            confidence=confidence,
            risk_reward_ratio=rr,
            stop_loss=stop_loss,
            take_profit=take_profit,
            expected_return_pct=expected_return_pct,
            expected_risk_pct=expected_risk_pct,
            notes=notes,
            metadata={
                "direction": direction,
                "alert_payload": payload,
                "portfolio_summary": portfolio_summary,
            },
        )

    def build_proposal(
        self,
        evaluation: SignalEvaluation,
        mark_price: Decimal,
        equity: Decimal,
        variant_id: str = "default",
    ) -> Optional[TradeProposal]:
        """从信号评估生成交易提案，如果不满足约束则返回 None."""
        mark_price = to_decimal(mark_price)
        equity = to_decimal(equity)
        direction = evaluation.metadata.get("direction", "neutral")

        if direction not in ("long", "short"):
            logger.debug("Neutral direction, no proposal for %s", evaluation.symbol)
            return None

        action = ActionType.OPEN_LONG if direction == "long" else ActionType.OPEN_SHORT
        constraint = self.objective.position_constraint
        budget = self.objective.risk_budget

        # 约束检查
        if not evaluation.passes(constraint.min_confidence, constraint.min_risk_reward_ratio):
            logger.debug(
                "Signal %s failed gate: confidence=%d, rr=%s",
                evaluation.symbol,
                evaluation.confidence,
                evaluation.risk_reward_ratio,
            )
            return None

        # 仓位大小：按风险预算 / 止损距离估算
        size_usd = self._size_position(
            equity=equity,
            entry=mark_price,
            stop=evaluation.stop_loss,
            risk_budget=budget.effective_risk_usd(equity),
            max_size=constraint.max_position_size_usd,
        )

        if size_usd <= ZERO:
            logger.debug("Sized to zero for %s", evaluation.symbol)
            return None

        return TradeProposal(
            action=action,
            symbol=evaluation.symbol,
            size_usd=size_usd,
            leverage=min(constraint.max_leverage, max(1, int(size_usd / equity * 2))),
            confidence=evaluation.confidence,
            stop_loss=evaluation.stop_loss,
            take_profit=evaluation.take_profit,
            expected_return_pct=evaluation.expected_return_pct,
            expected_risk_pct=evaluation.expected_risk_pct,
            risk_reward_ratio=evaluation.risk_reward_ratio,
            reason=", ".join(evaluation.notes),
            variant_id=variant_id,
        )

    # ── 内部辅助 ──
    def _extract_price(self, payload: Dict[str, Any], key: str) -> Optional[Decimal]:
        value = payload.get(key)
        if value is None:
            return None
        try:
            return to_decimal(value)
        except Exception:
            return None

    def _adjust_confidence(
        self,
        alert_type: str,
        payload: Dict[str, Any],
        base_confidence: int,
        notes: list[str],
    ) -> int:
        confidence = max(0, min(100, int(base_confidence)))
        if "breakout" in alert_type.lower():
            confidence = min(100, confidence + 5)
        if "divergence" in alert_type.lower():
            confidence = min(100, confidence + 3)
        if payload.get("volume_confirmed"):
            confidence = min(100, confidence + 5)
            notes.append("成交量确认")
        if payload.get("oi_rising"):
            confidence = min(100, confidence + 3)
            notes.append("持仓量配合")
        return confidence

    def _default_stop(
        self, mark_price: Decimal, direction: str, atr: Decimal
    ) -> Optional[Decimal]:
        if atr is None or atr <= ZERO:
            return None
        mark_price = to_decimal(mark_price)
        atr = to_decimal(atr)
        multiplier = Decimal("2.0")
        if direction == "long":
            return mark_price - atr * multiplier
        if direction == "short":
            return mark_price + atr * multiplier
        return None

    def _default_take_profit(
        self, mark_price: Decimal, stop_loss: Decimal, direction: str
    ) -> Optional[Decimal]:
        mark_price = to_decimal(mark_price)
        stop_loss = to_decimal(stop_loss)
        risk = abs(mark_price - stop_loss)
        if risk <= ZERO:
            return None
        rr = max(self.objective.position_constraint.min_risk_reward_ratio, Decimal("1.5"))
        if direction == "long":
            return mark_price + risk * rr
        if direction == "short":
            return mark_price - risk * rr
        return None

    def _compute_rr(
        self,
        mark_price: Decimal,
        stop_loss: Optional[Decimal],
        take_profit: Optional[Decimal],
        direction: str,
    ) -> Decimal:
        if stop_loss is None or take_profit is None:
            return ZERO
        mark_price = to_decimal(mark_price)
        risk = abs(mark_price - stop_loss)
        reward = abs(take_profit - mark_price)
        if risk <= ZERO:
            return ZERO
        return to_decimal(reward / risk)

    def _compute_expected_pct(
        self,
        mark_price: Decimal,
        stop_loss: Optional[Decimal],
        take_profit: Optional[Decimal],
        direction: str,
    ) -> tuple[Decimal, Decimal]:
        mark_price = to_decimal(mark_price)
        if stop_loss is None or take_profit is None or mark_price <= ZERO:
            return ZERO, ZERO
        risk = abs(mark_price - stop_loss) / mark_price * Decimal("100")
        reward = abs(take_profit - mark_price) / mark_price * Decimal("100")
        return to_decimal(reward), to_decimal(risk)

    def _size_position(
        self,
        equity: Decimal,
        entry: Decimal,
        stop: Optional[Decimal],
        risk_budget: Decimal,
        max_size: Decimal,
    ) -> Decimal:
        equity = to_decimal(equity)
        entry = to_decimal(entry)
        risk_budget = to_decimal(risk_budget)
        max_size = to_decimal(max_size)

        if entry <= ZERO or risk_budget <= ZERO:
            return ZERO

        if stop is None or stop <= ZERO:
            # 无止损时按权益百分比上限
            size = equity * Decimal("0.05")
        else:
            stop = to_decimal(stop)
            risk_per_unit = abs(entry - stop)
            if risk_per_unit <= ZERO:
                return ZERO
            size = risk_budget / risk_per_unit * entry

        return min(size, max_size)


class AgentChatSession:
    """Manages multi-agent conversation history, system prompt swapping, and LLM completions."""

    def __init__(self, evaluator: LLMSignalEvaluator, system_prompt: str):
        self.evaluator = evaluator
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})

    def get_last_response(self) -> Optional[str]:
        if self.messages and self.messages[-1]["role"] == "assistant":
            return self.messages[-1]["content"]
        return None

    def call(self, prompt: str, tracer: Optional[QuantTracer] = None) -> Optional[Dict[str, Any]]:
        """Appends the prompt as a user message, runs completions, and parses the response."""
        self.add_message("user", prompt)
        
        headers = {"Content-Type": "application/json"}
        if self.evaluator.api_key:
            headers["Authorization"] = f"Bearer {self.evaluator.api_key}"

        payload_data = {
            "model": self.evaluator.model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt}
            ] + self.messages,
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": False
        }
        
        t0 = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.evaluator.server_url.rstrip('/')}/v1/chat/completions",
                json=payload_data,
                headers=headers,
                timeout=30.0
            )
            duration = time.perf_counter() - t0
            
            if resp.status_code == 200:
                resp_json = resp.json()
                content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                # Trace LLM Call
                prompt_tokens = resp_json.get("usage", {}).get("prompt_tokens", 0)
                completion_tokens = resp_json.get("usage", {}).get("completion_tokens", 0)
                if tracer:
                    tracer.add_tokens(prompt_tokens, completion_tokens)
                    tracer.log_step(
                        "llm_chat_call",
                        duration,
                        {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
                    )
                
                self.add_message("assistant", content)
                return self.evaluator._parse_json_response(content)
            else:
                logger.warning("Local LLM server returned status code %d", resp.status_code)
        except Exception as e:
            logger.warning("Local LLM server invocation failed: %s", e)
        return None


class LLMSignalEvaluator(SignalEvaluator):
    """LLM-based Signal Evaluator.

    Uses a local LLM to run a multi-agent consensus debate between a Strategy Analyst
    and a Risk Officer, incorporating past reflections/lessons learned from the SQLite database.
    """

    def __init__(
        self,
        objective: AgentObjective,
        db: Optional[QuantSessionDB] = None,
        server_url: str = DEFAULT_SERVER_URL,
        api_key: str = DEFAULT_API_KEY,
        model_name: str = DEFAULT_MODEL,
        use_fallback: bool = True
    ):
        super().__init__(objective)
        self.db = db or QuantSessionDB()
        self.server_url = server_url
        self.api_key = api_key
        self.model_name = model_name
        self.use_fallback = use_fallback

    def evaluate(
        self,
        event: IndicatorAlertEvent,
        mark_price: Decimal,
        atr: Optional[Decimal] = None,
        portfolio_summary: Optional[Dict[str, Any]] = None,
    ) -> SignalEvaluation:
        # Initialize Tracer
        tracer = QuantTracer()
        tracer.start_span("LLMSignalEvaluator.evaluate", {"symbol": event.symbol})
        
        try:
            # 1. Fetch recent reflections from DB
            recent_reflections = []
            try:
                recent_reflections = self.db.get_recent_reflections(limit=5)
            except Exception as e:
                logger.warning("Failed to fetch recent reflections: %s", e)

            # 2. Format reflections for prompt
            reflection_str = "No past reflections found."
            if recent_reflections:
                reflection_lines = []
                for r in recent_reflections:
                    reflection_lines.append(
                        f"- Symbol: {r['symbol']}, PnL: {r['pnl']} USD, Lesson: {r['lesson']}"
                    )
                reflection_str = "\n".join(reflection_lines)

            # 3. Extract event parameters
            symbol = event.symbol.upper()
            alert_type = event.alert_type
            payload = event.payload or {}
            equity = "0"
            if portfolio_summary:
                equity = str(portfolio_summary.get("account", {}).get("equity", "0"))

            # Initialize debate state
            notes = []
            metadata = {
                "portfolio_summary": portfolio_summary,
                "debate_rounds": []
            }
            final_direction = "neutral"
            final_confidence = 50
            final_stop_loss = None
            final_take_profit = None
            final_rationale = "No consensus reached."
            success = False

            # Create Agent Chat Sessions
            analyst_session = AgentChatSession(self, "You are a professional quantitative contract trading expert.")
            risk_session = AgentChatSession(self, "You are a strict risk management officer for a trading desk.")

            # Multi-Agent Debate Loop (Max 2 Rounds)
            for round_idx in range(1, 3):
                notes.append(f"--- Debate Round {round_idx} ---")
                
                # Step A: Strategy Analyst Proposes
                if round_idx == 1:
                    analyst_prompt = f"""You are acting as the Strategy Analyst of a professional quantitative contract trading desk.
Your task is to analyze the following trading signal, market context, and historical reflections, and propose a trading decision (long, short, or neutral).

### Recent Lessons Learned (Reflections from Past Trades):
{reflection_str}

### Current Market Context:
- Symbol: {symbol}
- Current Mark Price: {mark_price}
- ATR (14): {atr or 'N/A'}
- Alert Type: {alert_type}
- Alert Details: {json.dumps(payload, ensure_ascii=False)}
- Portfolio Equity: {equity} USD

### Instructions:
Formulate your trade setup. You MUST reply ONLY with a valid JSON block inside ```json ... ``` code fence matching the structure below:
```json
{{
  "direction": "long/short/neutral",
  "confidence": 75,
  "stop_loss": 60500.0,
  "take_profit": 63000.0,
  "rationale": "Reason for long/short/neutral choice based on context."
}}
```
"""
                else:
                    # Round 2: Analyst gets feedback from Risk Officer
                    prev_feedback = metadata["debate_rounds"][-1]["risk_officer"]
                    analyst_prompt = f"""You are acting as the Strategy Analyst. The Risk Officer has rejected your previous trade proposal with the following feedback:
{json.dumps(prev_feedback, ensure_ascii=False)}

Please adjust your trade proposal (e.g. modify direction, stop loss, or take profit) to address the Risk Officer's concerns, or decide to stay neutral/flat.

### Current Market Context:
- Symbol: {symbol}
- Current Mark Price: {mark_price}
- ATR (14): {atr or 'N/A'}

You MUST reply ONLY with a valid JSON block inside ```json ... ``` code fence matching the structure below:
```json
{{
  "direction": "long/short/neutral",
  "confidence": 70,
  "stop_loss": 60700.0,
  "take_profit": 63000.0,
  "rationale": "Adjusted rationale addressing the risk officer's concerns."
}}
```
"""

                analyst_res = analyst_session.call(analyst_prompt, tracer=tracer)
                if not analyst_res:
                    notes.append("Strategy Analyst failed to respond.")
                    break
                
                notes.append(f"Strategy Analyst Proposal: {json.dumps(analyst_res, ensure_ascii=False)}")
                
                direction = str(analyst_res.get("direction", "neutral")).lower()
                confidence = int(analyst_res.get("confidence", 50))
                stop_loss = analyst_res.get("stop_loss")
                take_profit = analyst_res.get("take_profit")
                rationale = analyst_res.get("rationale", "")

                if direction == "neutral":
                    notes.append("Strategy Analyst decided to stay neutral/flat.")
                    final_direction = "neutral"
                    final_confidence = confidence
                    final_stop_loss = None
                    final_take_profit = None
                    final_rationale = rationale
                    success = True
                    break

                # Step B: Risk Officer Prompt and Call
                risk_prompt = f"""You are acting as the Risk Officer of a professional quantitative contract trading desk.
Your task is to review the trade proposal submitted by the Strategy Analyst. You must enforce strict wind-control rules to ensure capital safety.

### Trading Constraints & Risk Rules:
- Account Equity: {equity} USD
- Minimum Risk-Reward (RR) Ratio: {self.objective.position_constraint.min_risk_reward_ratio} (Take Profit distance / Stop Loss distance must be >= this value)
- Stop Loss Sizing: Risk per unit (entry price - stop loss) should be proportional to ATR (typically 1.5 - 2.5 times ATR) and not exceed {self.objective.risk_budget.max_risk_pct_per_trade}% of total equity.
- Current Mark Price: {mark_price}
- ATR (14): {atr or 'N/A'}

### Proposal under review:
{json.dumps(analyst_res, ensure_ascii=False)}

### Instructions:
Determine if the proposal is safe to execute. If it violates any risk rules (e.g., RR ratio too low, stop loss too tight/wide, or risk too high), you must reject it and provide constructive feedback on how the Analyst should adjust the parameters (e.g., specific stop loss level or size).

You MUST reply ONLY with a valid JSON block inside ```json ... ``` code fence matching the structure below:
```json
{{
  "approved": true/false,
  "feedback": "Why you approved or rejected the proposal.",
  "adjusted_stop_loss": 60700.0,
  "adjusted_take_profit": 63000.0,
  "adjusted_size_usd": 1000.0
}}
```
"""

                risk_res = risk_session.call(risk_prompt, tracer=tracer)
                if not risk_res:
                    notes.append("Risk Officer failed to respond.")
                    break

                notes.append(f"Risk Officer Review: {json.dumps(risk_res, ensure_ascii=False)}")
                
                metadata["debate_rounds"].append({
                    "round": round_idx,
                    "strategy_analyst": analyst_res,
                    "risk_officer": risk_res
                })

                approved = bool(risk_res.get("approved", False))
                if approved:
                    final_direction = direction
                    final_confidence = confidence
                    final_stop_loss = risk_res.get("adjusted_stop_loss") if risk_res.get("adjusted_stop_loss") is not None else stop_loss
                    final_take_profit = risk_res.get("adjusted_take_profit") if risk_res.get("adjusted_take_profit") is not None else take_profit
                    final_rationale = f"Consensus reached: {rationale} (Risk feedback: {risk_res.get('feedback', '')})"
                    success = True
                    notes.append("Risk Officer approved the proposal. Consensus reached!")
                    break
                else:
                    notes.append(f"Risk Officer rejected the proposal. Feedback: {risk_res.get('feedback', '')}")

            if not success and metadata["debate_rounds"]:
                final_direction = "neutral"
                final_confidence = 50
                final_stop_loss = None
                final_take_profit = None
                final_rationale = "No consensus reached after debate rounds. Staying neutral."
                success = True
                notes.append("No consensus reached. Defaulting to neutral/flat.")

            if not success:
                if self.use_fallback:
                    logger.info("Falling back to rule-based SignalEvaluator for %s", symbol)
                    fallback_res = super().evaluate(event, mark_price, atr, portfolio_summary)
                    tracer.end_span("fallback_success")
                    return fallback_res
                else:
                    raise RuntimeError("LLMSignalEvaluator failed and fallback is disabled.")

            sl_decimal = to_decimal(final_stop_loss) if final_stop_loss is not None else None
            tp_decimal = to_decimal(final_take_profit) if final_take_profit is not None else None

            rr = self._compute_rr(mark_price, sl_decimal, tp_decimal, final_direction)
            expected_return_pct, expected_risk_pct = self._compute_expected_pct(
                mark_price, sl_decimal, tp_decimal, final_direction
            )

            metadata["direction"] = final_direction
            metadata["rationale"] = final_rationale

            eval_res = SignalEvaluation(
                signal_type=alert_type,
                symbol=symbol,
                confidence=final_confidence,
                risk_reward_ratio=rr,
                stop_loss=sl_decimal,
                take_profit=tp_decimal,
                expected_return_pct=expected_return_pct,
                expected_risk_pct=expected_risk_pct,
                notes=notes,
                metadata=metadata
            )
            tracer.end_span("success")
            return eval_res

        except Exception as e:
            tracer.end_span("failed", str(e))
            raise e

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON block from LLM output."""
        try:
            # Try parsing whole text
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Look for code block markers
        import re
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Fallback: find first { and last }
        match_braces = re.search(r"(\{.*\})", text, re.DOTALL)
        if match_braces:
            try:
                return json.loads(match_braces.group(1))
            except json.JSONDecodeError:
                pass

        return None
