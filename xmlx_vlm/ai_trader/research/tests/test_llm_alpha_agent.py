"""Unit tests for LLMAlphaResearcher, prompt parsing, and Co-STEER feedback."""

import pytest
from xmlx_vlm.ai_trader.research.factor_memory import ExperienceMemory
from xmlx_vlm.ai_trader.research.llm_alpha_agent import (
    LLMAlphaResearcher,
    QuantitativeHypothesis,
)
from xmlx_vlm.ai_trader.research.symbolic_mining import parse_formula


class DummyLLMClient:
    """Mock LLM client returning structured JSON hypotheses."""

    def complete_sync(self, system_prompt: str, user_prompt: str) -> str:
        return """
[
  {
    "name": "Funding_CVD_Arbitrage",
    "description": "Funding rate divergence combined with CVD trend",
    "target_mechanism": "funding_squeeze",
    "formula_candidate": "ts_corr(funding, cvd, 10)",
    "expected_direction": -1
  },
  {
    "name": "Vol_Adjusted_Breakout",
    "description": "Normalized price breakout",
    "target_mechanism": "volatility_breakout",
    "formula_candidate": "(ts_delta(close, 5) / (ts_std(close, 20) + 1.00))",
    "expected_direction": 1
  }
]
"""


class TestLLMAlphaResearcher:
    """Test suite for LLM-driven alpha generation and diagnostic feedback."""

    def test_fallback_hypotheses_generation(self):
        researcher = LLMAlphaResearcher()
        hypotheses = researcher.generate_hypotheses(count=3)
        assert len(hypotheses) == 3
        for h in hypotheses:
            assert isinstance(h, QuantitativeHypothesis)
            assert h.formula_candidate != ""
            # Verify formula is syntactically valid AST
            ast_node = parse_formula(h.formula_candidate)
            assert ast_node is not None

    def test_llm_hypotheses_generation(self):
        mock_client = DummyLLMClient()
        researcher = LLMAlphaResearcher(llm_client=mock_client)
        hypotheses = researcher.generate_hypotheses(market_summary={"trend": "bullish"}, count=2)
        assert len(hypotheses) == 2
        assert hypotheses[0].name == "Funding_CVD_Arbitrage"
        assert hypotheses[0].formula_candidate == "ts_corr(funding, cvd, 10)"

    def test_co_steer_diagnostic_refinement(self):
        researcher = LLMAlphaResearcher()

        # Case 1: High IC but low IR (noisy) -> should add ts_decay_linear smoothing
        noisy_report = {"rank_ic": 0.06, "ir": 0.25, "t_stat": 1.8}
        refined, diagnosis = researcher.diagnose_and_refine(noisy_report, "ts_delta(close, 5)")
        assert "ts_decay_linear" in refined
        assert "smoothing" in diagnosis

        # Case 2: Low signal -> should add z-score
        weak_report = {"rank_ic": 0.015, "ir": 0.1, "t_stat": 0.9}
        refined_weak, diagnosis_weak = researcher.diagnose_and_refine(weak_report, "(high - low)")
        assert "ts_zscore" in refined_weak
        assert "Z-Score" in diagnosis_weak
