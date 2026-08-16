"""
LLM Alpha Researcher & Co-STEER Feedback Diagnostic Agent (RD-Agent Architecture).

Features:
1. Hypothesis-Driven Alpha Generation: Translates market microstructure & macroeconomic
   dynamics into formulaic AST expressions.
2. Experience-Conditioned Prompting: Feeds success templates and negative constraints
   from FactorMemory into LLM context.
3. Co-STEER Diagnostic Feedback Loop: Analyzes backtest diagnostics (IC decay, IR, turnover)
   to produce refined and mutated formula candidates.
4. Robust Fallback Engine: High-quality rule-based microstructure alpha generation when
   offline or when LLM API is unavailable.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .factor_memory import ExperienceMemory
from .symbolic_mining import DEFAULT_FIELDS, parse_formula

logger = logging.getLogger(__name__)


@dataclass
class QuantitativeHypothesis:
    """A financial / market microstructure hypothesis with formula implementation."""
    name: str
    description: str
    target_mechanism: str
    formula_candidate: str
    expected_direction: int = 1  # +1 positive correlation with forward return, -1 negative

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "target_mechanism": self.target_mechanism,
            "formula_candidate": self.formula_candidate,
            "expected_direction": self.expected_direction,
        }


# Catalog of domain-expert microstructure alpha templates for offline fallback
FALLBACK_ALPHA_TEMPLATES = [
    QuantitativeHypothesis(
        name="CVD_Momentum_Reversal",
        description="When Cumulative Volume Delta (CVD) accelerates faster than price, liquidity absorption is occurring, signaling mean-reversion.",
        target_mechanism="orderflow_reversal",
        formula_candidate="ratio_diff(ts_delta(cvd, 5), ts_std(volume, 10))",
        expected_direction=-1,
    ),
    QuantitativeHypothesis(
        name="Funding_OI_Pressure_Squeeze",
        description="Extreme positive funding rate coupled with surging open interest indicates overleveraged longs vulnerable to cascade liquidation.",
        target_mechanism="funding_squeeze",
        formula_candidate="(ts_zscore(funding, 20) * ts_delta(oi, 5))",
        expected_direction=-1,
    ),
    QuantitativeHypothesis(
        name="Imbalance_Weighted_Trend",
        description="Persistent orderbook bid/ask imbalance smoothed by decay linear weights predicts short-term directional momentum.",
        target_mechanism="orderflow_momentum",
        formula_candidate="ts_decay_linear(imbalance, 10)",
        expected_direction=1,
    ),
    QuantitativeHypothesis(
        name="Volatility_Normalized_Price_Delta",
        description="Price momentum normalized by rolling Parkinson/standard volatility to filter out false breakouts during low-volatility regimes.",
        target_mechanism="volatility_breakout",
        formula_candidate="(ts_delta(close, 5) / (ts_std(close, 20) + 1.00))",
        expected_direction=1,
    ),
    QuantitativeHypothesis(
        name="Volume_Price_Correlation_Exhaustion",
        description="High rolling correlation between price and volume near local extremes indicates trend exhaustion.",
        target_mechanism="trend_exhaustion",
        formula_candidate="ts_corr(close, volume, 15)",
        expected_direction=-1,
    ),
    QuantitativeHypothesis(
        name="Liquidity_Spread_Mean_Reversion",
        description="Discrepancy between high-low spread and volume indicates illiquidity premium mean-reversion.",
        target_mechanism="liquidity_premium",
        formula_candidate="ratio_diff((high - low), ts_mean(volume, 20))",
        expected_direction=1,
    ),
]


class LLMAlphaResearcher:
    """
    LLM-powered Quantitative Researcher that formulates hypotheses and iterates factors.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        memory: Optional[ExperienceMemory] = None,
    ):
        self.llm_client = llm_client
        self.memory = memory or ExperienceMemory()

    def generate_hypotheses(
        self,
        market_summary: Optional[Dict[str, Any]] = None,
        count: int = 3,
    ) -> List[QuantitativeHypothesis]:
        """
        Generate financial hypotheses and formula candidates using LLM (or fallback catalog).
        """
        if self.llm_client is not None:
            try:
                hypotheses = self._llm_generate(market_summary, count)
                if hypotheses:
                    return hypotheses
            except Exception as e:
                logger.warning("LLM hypothesis generation failed, falling back to rule-based catalog: %s", e)

        # Fallback to rich template engine
        selected = list(FALLBACK_ALPHA_TEMPLATES)
        if count < len(selected):
            import random
            selected = random.sample(selected, count)
        return selected

    def _llm_generate(
        self,
        market_summary: Optional[Dict[str, Any]],
        count: int,
    ) -> List[QuantitativeHypothesis]:
        """Internal prompt-based hypothesis generation."""
        success_templates = self.memory.get_success_templates(top_n=3)
        failure_skeletons = [fc.ast_skeleton for fc in self.memory.failure_constraints[:3]]

        system_prompt = (
            "You are a Senior Quantitative Researcher specialized in Crypto & Derivatives Alpha Generation. "
            "Your task is to formulate high-conviction mathematical trading alpha formulas based on market microstructure. "
            "Allowed Fields: open, high, low, close, volume, cvd, oi, funding, imbalance.\n"
            "Allowed Operators:\n"
            "- Unary: ts_mean(x, w), ts_std(x, w), ts_delta(x, w), ts_rank(x, w), ts_decay_linear(x, w), ts_zscore(x, w), abs(x), neg(x), sign(x)\n"
            "- Binary: add(x, y), sub(x, y), mul(x, y), div(x, y), ratio_diff(x, y), ts_corr(x, y, w)\n"
            "Output JSON with a list of hypotheses."
        )

        user_prompt = f"""Generate {count} distinct quantitative alpha hypotheses.
Market Summary: {json.dumps(market_summary or {})}
Known Successful Patterns: {json.dumps(success_templates)}
Known Overfitted/Failed Patterns (DO NOT REPEAT): {json.dumps(failure_skeletons)}

Format your response strictly as JSON:
[
  {{
    "name": "Hypothesis_Name",
    "description": "Economic/Microstructure rationale",
    "target_mechanism": "orderflow_momentum/reversal/volatility/funding",
    "formula_candidate": "valid_formula_string",
    "expected_direction": 1
  }}
]
"""
        # Execute LLM call synchronously if possible or via client
        response_text = ""
        if hasattr(self.llm_client, "complete_sync"):
            response_text = self.llm_client.complete_sync(system_prompt, user_prompt)
        elif hasattr(self.llm_client, "complete"):
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        response_text = pool.submit(
                            lambda: asyncio.run(self.llm_client.complete(system_prompt, user_prompt))
                        ).result(timeout=10.0)
                else:
                    response_text = loop.run_until_complete(
                        self.llm_client.complete(system_prompt, user_prompt)
                    )
            except Exception as ex:
                logger.debug("Async complete failed: %s", ex)

        if not response_text:
            return []

        # Parse JSON
        parsed_json = self._extract_json(response_text)
        results: List[QuantitativeHypothesis] = []
        for item in parsed_json:
            if "formula_candidate" in item:
                # Validate syntax via parse_formula
                try:
                    parse_formula(item["formula_candidate"])
                    results.append(QuantitativeHypothesis(
                        name=item.get("name", "LLM_Alpha"),
                        description=item.get("description", ""),
                        target_mechanism=item.get("target_mechanism", "composite"),
                        formula_candidate=item["formula_candidate"],
                        expected_direction=int(item.get("expected_direction", 1)),
                    ))
                except Exception as e:
                    logger.debug("Syntax validation failed for formula %s: %s", item.get("formula_candidate"), e)

        return results

    def diagnose_and_refine(
        self,
        factor_report: Dict[str, Any],
        current_formula: str,
    ) -> Tuple[str, str]:
        """
        Co-STEER Feedback Diagnostician:
        Analyzes factor metrics (Rank IC, IR, t-stat, turnover) and returns (refined_formula, diagnosis_reason).
        """
        rank_ic = float(factor_report.get("rank_ic", factor_report.get("residual_rank_ic", 0.0)))
        ir = float(factor_report.get("ir", 0.0))
        t_stat = float(factor_report.get("t_stat", 0.0))

        # Rule-based diagnostic logic
        if abs(rank_ic) >= 0.04 and abs(ir) < 0.35:
            # High IC but noisy/unstable -> add smoothing with ts_decay_linear
            refined = f"ts_decay_linear({current_formula}, 10)"
            diagnosis = "High Rank IC but low IR indicates signal volatility; applying linear decay smoothing (ts_decay_linear) to reduce noise."
        elif abs(rank_ic) < 0.02 and abs(t_stat) < 1.5:
            # Low signal -> normalize by rolling volatility (ts_zscore)
            refined = f"ts_zscore({current_formula}, 20)"
            diagnosis = "Marginal Rank IC and t-stat; converting signal to rolling Z-Score to isolate regime departures."
        elif "cvd" in current_formula and "volume" not in current_formula:
            # Orderflow unnormalized -> normalize by volume
            refined = f"ratio_diff({current_formula}, ts_mean(volume, 10))"
            diagnosis = "Raw orderflow volume delta lacks relative volume normalization; dividing by rolling average volume."
        else:
            # Wrap in rank normalizer
            refined = f"ts_rank({current_formula}, 15)"
            diagnosis = "Signal distribution contains outliers; applying rolling percentile ranking."

        return refined, diagnosis

    def _extract_json(self, text: str) -> List[Dict[str, Any]]:
        """Extract JSON array from LLM response text."""
        text = text.strip()
        match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        # Try direct parse
        return json.loads(text)
