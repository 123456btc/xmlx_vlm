"""Unit tests for FactorEvaluator, IC/IR reporting, Quantile Layering, and Symbolic Mining."""

import math
import pytest
from xmlx_vlm.ai_trader.research.factor_analyzer import (
    FactorEvaluator,
    _pearson_corr,
    _spearman_rank_corr,
    gram_schmidt_orthogonalize,
    symmetric_orthogonalize,
)
from xmlx_vlm.ai_trader.research.symbolic_mining import (
    BinaryOpNode,
    ConstNode,
    FieldNode,
    SymbolicFactorGenerator,
    UnaryOpNode,
)


class TestFactorMining:
    """Test suite for IC/Rank IC analysis, Quantile layering, and Symbolic Factor Discovery."""

    def test_correlation_and_rank_corr(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        assert pytest.approx(_pearson_corr(x, y), 0.001) == 1.0
        assert pytest.approx(_spearman_rank_corr(x, y), 0.001) == 1.0

        # Monotonic non-linear
        y2 = [1.0, 10.0, 100.0, 1000.0, 10000.0]
        assert pytest.approx(_spearman_rank_corr(x, y2), 0.001) == 1.0

    def test_time_series_ic_evaluation(self):
        # Synthetic factor with high positive correlation with forward return
        factor_values = [float(i) + (0.5 if i % 2 == 0 else -0.5) for i in range(100)]
        forward_returns = [0.01 * float(i) + (0.005 if i % 2 == 0 else -0.005) for i in range(100)]

        report = FactorEvaluator.evaluate_time_series_ic(factor_values, forward_returns, rolling_window=20)
        assert report.mean_rank_ic > 0.5
        assert report.sample_size == 80
        assert report.is_significant is True
        summary_text = report.summary()
        assert "Mean IC" in summary_text
        assert "Rank IC" in summary_text

    def test_quantile_layering_monotonicity(self):
        # Cross-sectional matrix with 10 symbols across 50 timesteps
        symbols = [f"COIN_{i}" for i in range(10)]
        factor_matrix = {}
        returns_matrix = {}

        for idx, s in enumerate(symbols):
            # Higher index symbol has higher factor value and higher return
            factor_matrix[s] = [float(idx * 2) + 0.1 * t for t in range(50)]
            returns_matrix[s] = [0.01 * float(idx) + 0.001 * t for t in range(50)]

        q_report = FactorEvaluator.evaluate_quantiles(factor_matrix, returns_matrix, num_quantiles=5)
        assert q_report.num_quantiles == 5
        assert "Q1" in q_report.quantile_returns
        assert "Q5" in q_report.quantile_returns
        assert q_report.long_short_spread_pct > 0.0
        assert q_report.is_monotonic is True
        assert q_report.monotonicity_score > 0.8

    def test_symbolic_expression_eval(self):
        data = {
            "close": [10.0, 12.0, 15.0, 14.0, 16.0, 18.0],
            "volume": [100.0, 120.0, 150.0, 110.0, 130.0, 170.0],
        }

        # AST: ts_mean(close, 3)
        mean_node = UnaryOpNode(op="ts_mean", child=FieldNode("close"), param=3)
        assert mean_node.to_formula() == "ts_mean(close, 3)"
        res = mean_node.eval(data)
        assert len(res) == 6
        assert pytest.approx(res[-1], 0.01) == (14.0 + 16.0 + 18.0) / 3.0

        # AST: (close / volume)
        div_node = BinaryOpNode(op="div", left=FieldNode("close"), right=FieldNode("volume"))
        assert div_node.to_formula() == "(close / volume)"
        div_res = div_node.eval(data)
        assert len(div_res) == 6
        assert pytest.approx(div_res[0], 0.001) == 10.0 / 100.0

    def test_symbolic_factor_generator_mining(self):
        # Generate synthetic market data
        n = 100
        data = {
            "close": [50000.0 + i * 50.0 for i in range(n)],
            "volume": [100.0 + i * 2.0 for i in range(n)],
            "cvd": [10.0 + i for i in range(n)],
            "oi": [1000.0 + i * 5.0 for i in range(n)],
            "funding": [0.0001 for _ in range(n)],
            "imbalance": [0.1 for _ in range(n)],
        }
        forward_returns = [0.001 * i for i in range(n)]

        generator = SymbolicFactorGenerator(random_seed=42)
        discovered = generator.mine_factors(
            data=data,
            forward_returns=forward_returns,
            population_size=15,
            generations=2,
            min_rank_ic=0.01,
        )

    def test_gram_schmidt_exact_orthogonality(self):
        # Base factor 1: Linear trend
        f1 = [float(i) for i in range(100)]
        # Base factor 2: Sine wave
        f2 = [math.sin(i * 0.1) * 10.0 for i in range(100)]
        
        # Candidate factor: A mix of f1, f2 plus some independent signal
        f_cand = [0.8 * f1[i] + 0.5 * f2[i] + math.cos(i * 0.2) * 5.0 for i in range(100)]

        # Verify raw candidate has high correlation with base factors
        assert abs(_pearson_corr(f_cand, f1)) > 0.5

        # Orthogonalize candidate against [f1, f2]
        f_ortho = gram_schmidt_orthogonalize(f_cand, [f1, f2])

        # Assert orthogonalized factor has virtually 0 correlation with both base factors
        corr_with_f1 = abs(_pearson_corr(f_ortho, f1))
        corr_with_f2 = abs(_pearson_corr(f_ortho, f2))
        assert corr_with_f1 < 1e-4, f"Expected corr ~0 with f1, got {corr_with_f1}"
        assert corr_with_f2 < 1e-4, f"Expected corr ~0 with f2, got {corr_with_f2}"

    def test_symmetric_orthogonalization(self):
        # 3 correlated factors
        f1 = [float(i) for i in range(80)]
        f2 = [float(i) * 1.5 + (2.0 if i % 2 == 0 else -2.0) for i in range(80)]
        f3 = [float(i) * 0.8 + math.sin(i * 0.2) * 5.0 for i in range(80)]

        ortho_set = symmetric_orthogonalize([f1, f2, f3])
        assert len(ortho_set) == 3

        # Verify each pair in ortho_set has zero mutual correlation
        c12 = abs(_pearson_corr(ortho_set[0], ortho_set[1]))
        c13 = abs(_pearson_corr(ortho_set[0], ortho_set[2]))
        c23 = abs(_pearson_corr(ortho_set[1], ortho_set[2]))

        assert c12 < 1e-3, f"Expected ortho c12 ~0, got {c12}"
        assert c13 < 1e-3, f"Expected ortho c13 ~0, got {c13}"
        assert c23 < 1e-3, f"Expected ortho c23 ~0, got {c23}"

    def test_incremental_alpha_evaluation_report(self):
        # Base factor: Momentum
        base_f = [float(i) for i in range(100)]
        # Forward returns driven by an independent non-linear feature
        indep_f = [math.sin(i * 0.15) * 10.0 for i in range(100)]
        forward_returns = [0.01 * indep_f[i] for i in range(100)]

        # Candidate A: Clone of base factor (collinear, no incremental alpha)
        clone_cand = [float(i) * 2.0 + 1.0 for i in range(100)]
        rep_clone = FactorEvaluator.evaluate_incremental_alpha(clone_cand, [base_f], forward_returns)
        assert rep_clone.r_squared_overlap > 0.95
        assert rep_clone.is_true_incremental_alpha is False

        # Candidate B: The true independent factor (genuine incremental alpha)
        rep_indep = FactorEvaluator.evaluate_incremental_alpha(indep_f, [base_f], forward_returns)
        assert rep_indep.r_squared_overlap < 0.20
        assert rep_indep.is_true_incremental_alpha is True
        summary = rep_indep.summary()
        assert "纯增量 Alpha" in summary

    def test_symbolic_mining_with_base_factors(self):
        n = 100
        # Base factor present in system
        base_factor = [float(i) * 10.0 for i in range(n)]
        
        data = {
            "close": [50000.0 + i * 50.0 for i in range(n)],
            "volume": [100.0 + (50.0 if i % 3 == 0 else -30.0) for i in range(n)],
            "cvd": [10.0 + i for i in range(n)],
            "oi": [1000.0 + i * 5.0 for i in range(n)],
            "funding": [0.0001 for _ in range(n)],
            "imbalance": [0.1 for _ in range(n)],
        }
        forward_returns = [0.002 * (data["volume"][i] - 100.0) for i in range(n)]

        generator = SymbolicFactorGenerator(random_seed=42)
        discovered = generator.mine_factors(
            data=data,
            forward_returns=forward_returns,
            base_factors=[base_factor],
            population_size=15,
            generations=2,
            min_rank_ic=0.01,
        )

        assert isinstance(discovered, list)
        if discovered:
            first = discovered[0]
            assert "formula" in first
            assert "residual_rank_ic" in first
            assert "r_squared_overlap" in first

