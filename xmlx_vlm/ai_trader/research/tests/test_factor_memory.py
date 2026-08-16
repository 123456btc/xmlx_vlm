"""Unit tests for ExperienceMemory, AST skeleton extraction, similarity, and pruning."""

import os
import tempfile
import pytest

from xmlx_vlm.ai_trader.research.factor_memory import (
    ExperienceMemory,
    FactorPattern,
    FailureConstraint,
    compute_ast_similarity,
    extract_ast_skeleton,
)
from xmlx_vlm.ai_trader.research.symbolic_mining import (
    BinaryOpNode,
    FieldNode,
    UnaryOpNode,
    parse_formula,
)


class TestFactorMemory:
    """Test suite for Factor Experience Memory (FactorMiner architecture)."""

    def test_ast_skeleton_extraction(self):
        node1 = parse_formula("ts_std(close, 5)")
        skel1 = extract_ast_skeleton(node1)
        assert skel1 == "ts_std(FIELD, W)"

        node2 = parse_formula("ts_std(volume, 10)")
        skel2 = extract_ast_skeleton(node2)
        assert skel1 == skel2  # Both share same normalized structural skeleton

        node3 = parse_formula("(close / volume)")
        skel3 = extract_ast_skeleton(node3)
        assert skel3 == "(FIELD div FIELD)"

    def test_ast_similarity(self):
        node1 = parse_formula("ts_mean(close, 5)")
        node2 = parse_formula("ts_mean(close, 5)")
        assert compute_ast_similarity(node1, node2) == 1.0

        node3 = parse_formula("ts_std(volume, 20)")
        sim = compute_ast_similarity(node1, node3)
        assert 0.0 <= sim < 0.5

    def test_experience_memory_record_and_pruning(self):
        memory = ExperienceMemory()

        # Record a successful factor
        factor_data = {
            "formula": "ts_mean(close, 5)",
            "rank_ic": 0.05,
            "ir": 0.45,
            "t_stat": 3.2,
            "category": "momentum",
            "is_true_incremental_alpha": True,
        }
        pat = memory.record_success(factor_data, is_active_pool=True)
        assert pat.formula == "ts_mean(close, 5)"
        assert len(memory.success_factors) == 1
        assert len(memory.active_factor_pool) == 1

        # Check duplicate formula is blocked
        is_blocked, reason = memory.is_redundant_or_constrained("ts_mean(close, 5)")
        assert is_blocked is True
        assert "already present in success memory" in reason

        # Record a failure constraint
        memory.record_failure("ts_delta(oi, 2)", reason="high_decay")
        assert len(memory.failure_constraints) == 1

        is_blocked_fail, fail_reason = memory.is_redundant_or_constrained("ts_delta(oi, 2)")
        assert is_blocked_fail is True
        assert "failure constraint" in fail_reason

        # Check new distinct candidate passes
        is_blocked_ok, _ = memory.is_redundant_or_constrained("ts_corr(close, volume, 15)")
        assert is_blocked_ok is False

    def test_memory_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "test_factor_memory.json")
            mem = ExperienceMemory(persistence_path=json_path)

            mem.record_success({
                "formula": "ratio_diff(high, low)",
                "rank_ic": 0.04,
                "ir": 0.38,
                "t_stat": 2.8,
            })
            mem.record_failure("ts_min(close, 3)", reason="zero_variance")

            assert os.path.exists(json_path)

            # Reload into a fresh instance
            mem2 = ExperienceMemory(persistence_path=json_path)
            assert len(mem2.success_factors) == 1
            assert "ratio_diff(high, low)" in mem2.success_factors
            assert len(mem2.failure_constraints) == 1

            summary = mem2.export_summary()
            assert summary["total_success_factors"] == 1
            assert summary["total_failure_constraints"] == 1
