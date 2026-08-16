"""Unit tests for EvolvingAlphaEngine, Portfolio Synergy Sharpe, and self-evolution loop."""

import pytest
import numpy as np

from xmlx_vlm.ai_trader.research.evolutionary_miner import (
    EvolutionaryMiningConfig,
    EvolvingAlphaEngine,
    _calculate_series_turnover,
    _calculate_sharpe,
)
from xmlx_vlm.ai_trader.research.factor_memory import ExperienceMemory
from xmlx_vlm.ai_trader.research.symbolic_mining import parse_formula, crossover_trees


class TestEvolutionaryMiner:
    """Test suite for Self-Evolving Alpha Factor Mining Engine."""

    def test_crossover_trees(self):
        t1 = parse_formula("ts_mean(close, 5)")
        t2 = parse_formula("ts_std(volume, 10)")
        c1, c2 = crossover_trees(t1, t2)
        assert c1.to_formula() != ""
        assert c2.to_formula() != ""

    def test_turnover_and_sharpe_calculations(self):
        # Stable series -> low turnover
        stable_series = [float(i) for i in range(100)]
        turnover_low = _calculate_series_turnover(stable_series)
        assert turnover_low < 0.05

        # Oscillating series -> high turnover
        noisy_series = [1.0 if i % 2 == 0 else -1.0 for i in range(100)]
        turnover_high = _calculate_series_turnover(noisy_series)
        assert turnover_high > 0.3

        # Consistent positive returns -> high Sharpe
        pos_returns = [0.01 + 0.001 * (i % 3) for i in range(50)]
        sharpe = _calculate_sharpe(pos_returns)
        assert sharpe > 1.0

    def test_synergy_sharpe_evaluation(self):
        memory = ExperienceMemory()
        engine = EvolvingAlphaEngine(memory=memory)

        n = 100
        forward_returns = [0.002 * (1 if i % 2 == 0 else -0.5) for i in range(n)]
        candidate_values = [float(i) for i in range(n)]

        # 1. No active factors -> candidate is baseline
        sharpe, delta_sharpe = engine.evaluate_synergy_sharpe(
            candidate_values=candidate_values,
            active_factors={},
            forward_returns=forward_returns,
        )
        assert sharpe == delta_sharpe

        # 2. With existing active factors
        active_factors = {
            "factor_1": [float(i * 0.5) for i in range(n)],
        }
        comp_sharpe, delta_sharpe_2 = engine.evaluate_synergy_sharpe(
            candidate_values=candidate_values,
            active_factors=active_factors,
            forward_returns=forward_returns,
        )
        assert isinstance(comp_sharpe, float)
        assert isinstance(delta_sharpe_2, float)

    def test_full_evolutionary_mining_cycle(self):
        # Synthetic market data
        n = 120
        data = {
            "open": [50000.0 + i * 40.0 for i in range(n)],
            "high": [50050.0 + i * 40.0 for i in range(n)],
            "low": [49950.0 + i * 40.0 for i in range(n)],
            "close": [50000.0 + i * 42.0 for i in range(n)],
            "volume": [100.0 + (i % 10) * 15.0 for i in range(n)],
            "cvd": [5.0 + i * 2.0 for i in range(n)],
            "oi": [1000.0 + i * 10.0 for i in range(n)],
            "funding": [0.0001 for _ in range(n)],
            "imbalance": [0.2 if i % 3 == 0 else -0.1 for i in range(n)],
        }
        # Forward returns positively correlated with price momentum & CVD
        forward_returns = [0.0005 * i + 0.001 * (1 if i % 2 == 0 else -1) for i in range(n)]

        memory = ExperienceMemory()
        config = EvolutionaryMiningConfig(
            population_size=15,
            generations=2,
            min_rank_ic=0.02,
            min_ir=0.15,
        )
        engine = EvolvingAlphaEngine(memory=memory, config=config)

        results = engine.evolve_factors(
            data=data,
            forward_returns=forward_returns,
            market_summary={"regime": "trend_following"},
        )

        assert "total_factors_discovered" in results
        assert "top_factors" in results
        assert "generation_history" in results
        assert results["generations_completed"] == 2
        assert len(results["generation_history"]) == 2

        # Check memory has recorded entries
        mem_summary = results["experience_memory_summary"]
        assert mem_summary["total_success_factors"] + mem_summary["total_failure_constraints"] > 0
