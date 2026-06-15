"""信号评估器：置信度、RR、止损、仓位."""

from __future__ import annotations

import logging
import requests
import json
from decimal import Decimal
from typing import Any, Dict, Optional, List

from xmlx_vlm.ai_trader.config import DEFAULT_API_KEY, DEFAULT_SERVER_URL, DEFAULT_MODEL
from xmlx_vlm.ai_trader.store.session_db import QuantSessionDB
from xmlx_vlm.ai_trader.agent.config import AgentObjective
from xmlx_vlm.ai_trader.agent.decision import ActionType, SignalEvaluation, TradeProposal
from xmlx_vlm.ai_trader.market_service.events import IndicatorAlertEvent
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO

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


class LLMSignalEvaluator(SignalEvaluator):
    """LLM-based Signal Evaluator.

    Uses a local LLM to run a single-request Bull/Bear debate and return a consensus decision,
    incorporating past reflections/lessons learned from the SQLite database.
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

        # 4. Formulate the debate prompt
        prompt = f"""You are a professional quantitative contract trading desk.
Evaluate the following trading signal and make a decision.

To prevent confirmation bias, you must perform a structured debate between a Bull Analyst (seeking long arguments) and a Bear Analyst (seeking short/flat arguments).

### Recent Lessons Learned (Reflections from Past Trades):
{reflection_str}

### Current Market Context:
- Symbol: {symbol}
- Current Mark Price: {mark_price}
- ATR (14): {atr or 'N/A'}
- Alert Type: {alert_type}
- Alert Details: {json.dumps(payload, ensure_ascii=False)}
- Portfolio Equity: {equity} USD

### Debate Instructions:
1. **Bull Argument**: Why we should go LONG.
2. **Bear Argument**: Why we should go SHORT or stay FLAT.
3. **Consensus & Rebuttal**: Reconcile both sides.
4. **Final Decision**: Determine the direction (long, short, or neutral) and confidence. Suggest stop loss and take profit if any.

You MUST reply ONLY with a valid JSON block inside ```json ... ``` code fence matching the structure below:
```json
{{
  "debate": {{
    "bull_case": "Bull case details",
    "bear_case": "Bear case details",
    "rebuttal": "Consensus and rebuttal details"
  }},
  "decision": {{
    "direction": "long",
    "confidence": 75,
    "stop_loss": 60500.0,
    "take_profit": 63000.0,
    "rationale": "Rationale details"
  }}
}}
```
"""

        # 5. Invoke local LLM server
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload_data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a professional quantitative contract trading expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
            "stream": False
        }

        success = False
        decision_data = {}
        try:
            resp = requests.post(
                f"{self.server_url.rstrip('/')}/v1/chat/completions",
                json=payload_data,
                headers=headers,
                timeout=30.0
            )
            if resp.status_code == 200:
                resp_json = resp.json()
                content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                decision_data = self._parse_json_response(content)
                if decision_data:
                    success = True
            else:
                logger.warning("Local LLM server returned status code %d", resp.status_code)
        except Exception as e:
            logger.warning("Local LLM server invocation failed: %s", e)

        # 6. Fallback if failed or disabled
        if not success:
            if self.use_fallback:
                logger.info("Falling back to rule-based SignalEvaluator for %s", symbol)
                return super().evaluate(event, mark_price, atr, portfolio_summary)
            else:
                raise RuntimeError("LLMSignalEvaluator failed and fallback is disabled.")

        # 7. Map parsed JSON to SignalEvaluation
        dec = decision_data.get("decision", {})
        dir_val = str(dec.get("direction", "neutral")).lower()
        confidence = int(dec.get("confidence", 50))
        stop_loss = dec.get("stop_loss")
        take_profit = dec.get("take_profit")
        rationale = dec.get("rationale", "")

        notes = [
            f"LLM Consensus: {dir_val.upper()} (Confidence: {confidence})",
            f"Rationale: {rationale}",
            f"Bull Case: {decision_data.get('debate', {}).get('bull_case', '')}",
            f"Bear Case: {decision_data.get('debate', {}).get('bear_case', '')}"
        ]

        # Convert SL / TP to Decimal if present
        sl_decimal = to_decimal(stop_loss) if stop_loss is not None else None
        tp_decimal = to_decimal(take_profit) if take_profit is not None else None

        # Compute risk reward & expected return
        rr = self._compute_rr(mark_price, sl_decimal, tp_decimal, dir_val)
        expected_return_pct, expected_risk_pct = self._compute_expected_pct(
            mark_price, sl_decimal, tp_decimal, dir_val
        )

        return SignalEvaluation(
            signal_type=alert_type,
            symbol=symbol,
            confidence=confidence,
            risk_reward_ratio=rr,
            stop_loss=sl_decimal,
            take_profit=tp_decimal,
            expected_return_pct=expected_return_pct,
            expected_risk_pct=expected_risk_pct,
            notes=notes,
            metadata={
                "direction": dir_val,
                "debate": decision_data.get("debate", {}),
                "rationale": rationale,
                "portfolio_summary": portfolio_summary,
            }
        )

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
