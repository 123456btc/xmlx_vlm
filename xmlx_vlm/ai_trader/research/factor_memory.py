"""
Factor Experience Memory & Pattern Library (FactorMiner Inspired).

Features:
1. Long-term & short-term Alpha Experience Memory (Success Patterns & Failure Constraints).
2. Structural AST Skeleton Extraction and AST Jaccard/Edit Distance Similarity.
3. Pre-evaluation Redundancy Pruning (Prevents falling into the "Correlation Red Sea").
4. JSON Persistence for cross-session knowledge accumulation and self-evolution.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .symbolic_mining import (
    BinaryOpNode,
    ConstNode,
    FieldNode,
    SymbolicNode,
    UnaryOpNode,
    parse_formula,
)

logger = logging.getLogger(__name__)


def extract_ast_skeleton(node: SymbolicNode) -> str:
    """
    Extract a normalized structural skeleton of an AST tree by replacing field names and constants with placeholders.
    Example: `ts_std(close, 5)` -> `ts_std(FIELD, W)`
             `(close / volume)` -> `(FIELD / FIELD)`
    """
    if isinstance(node, FieldNode):
        return "FIELD"
    elif isinstance(node, ConstNode):
        return "CONST"
    elif isinstance(node, UnaryOpNode):
        child_skel = extract_ast_skeleton(node.child)
        return f"{node.op}({child_skel}, W)"
    elif isinstance(node, BinaryOpNode):
        left_skel = extract_ast_skeleton(node.left)
        right_skel = extract_ast_skeleton(node.right)
        if node.op in ("add", "sub", "mul", "div"):
            return f"({left_skel} {node.op} {right_skel})"
        return f"{node.op}({left_skel}, {right_skel}, W)"
    return "NODE"


def get_ast_feature_multiset(node: SymbolicNode) -> List[str]:
    """Extract a bag of structural tokens from an AST for similarity computation."""
    tokens: List[str] = []
    if isinstance(node, FieldNode):
        tokens.append(f"field:{node.field_name}")
    elif isinstance(node, ConstNode):
        tokens.append("const")
    elif isinstance(node, UnaryOpNode):
        tokens.append(f"uop:{node.op}")
        tokens.append(f"param:{node.param}")
        tokens.extend(get_ast_feature_multiset(node.child))
    elif isinstance(node, BinaryOpNode):
        tokens.append(f"bop:{node.op}")
        tokens.append(f"param:{node.param}")
        tokens.extend(get_ast_feature_multiset(node.left))
        tokens.extend(get_ast_feature_multiset(node.right))
    return tokens


def compute_ast_similarity(node1: SymbolicNode, node2: SymbolicNode) -> float:
    """
    Compute Jaccard similarity between two AST feature sets.
    Returns float in [0.0, 1.0].
    """
    t1 = set(get_ast_feature_multiset(node1))
    t2 = set(get_ast_feature_multiset(node2))
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    intersection = len(t1.intersection(t2))
    union = len(t1.union(t2))
    return intersection / union if union > 0 else 0.0


@dataclass
class FactorPattern:
    """Metadata record of a successful discovered factor."""
    formula: str
    ast_skeleton: str
    category: str = "composite"  # momentum, reversal, volatility, orderflow, funding
    rank_ic: float = 0.0
    ir: float = 0.0
    t_stat: float = 0.0
    turnover: float = 0.0
    r_squared_overlap: float = 0.0
    is_true_incremental_alpha: bool = True
    generation: int = 0
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FailureConstraint:
    """Pattern record of a failed, overfitted, or high-decay factor."""
    formula: str
    ast_skeleton: str
    failure_reason: str  # high_decay, high_correlation, low_ic, high_turnover, overfitting
    metrics: Dict[str, float] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperienceMemory:
    """
    Long-Term Factor Experience Memory and Pruning Engine (FactorMiner Architecture).
    """

    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = persistence_path
        self.success_factors: Dict[str, FactorPattern] = {}
        self.failure_constraints: List[FailureConstraint] = []
        self.active_factor_pool: Dict[str, FactorPattern] = {}
        self._skeleton_counts: Dict[str, int] = {}

        if self.persistence_path and os.path.exists(self.persistence_path):
            self.load_from_json(self.persistence_path)

    def record_success(self, factor_data: Dict[str, Any], is_active_pool: bool = False) -> FactorPattern:
        """Register a successful factor into the experience memory."""
        formula = factor_data.get("formula", "").strip()
        if not formula:
            raise ValueError("Formula string cannot be empty.")

        ast_node = parse_formula(formula)
        skeleton = extract_ast_skeleton(ast_node)

        category = factor_data.get("category") or self._classify_category(formula)

        pattern = FactorPattern(
            formula=formula,
            ast_skeleton=skeleton,
            category=category,
            rank_ic=float(factor_data.get("rank_ic", factor_data.get("residual_rank_ic", 0.0))),
            ir=float(factor_data.get("ir", 0.0)),
            t_stat=float(factor_data.get("t_stat", 0.0)),
            turnover=float(factor_data.get("turnover", 0.0)),
            r_squared_overlap=float(factor_data.get("r_squared_overlap", 0.0)),
            is_true_incremental_alpha=bool(factor_data.get("is_true_incremental_alpha", True)),
            generation=int(factor_data.get("generation", 0)),
            tags=factor_data.get("tags", []),
        )

        self.success_factors[formula] = pattern
        self._skeleton_counts[skeleton] = self._skeleton_counts.get(skeleton, 0) + 1

        if is_active_pool:
            self.active_factor_pool[formula] = pattern

        if self.persistence_path:
            self.save_to_json(self.persistence_path)

        return pattern

    def record_failure(self, formula: str, reason: str, metrics: Optional[Dict[str, float]] = None) -> FailureConstraint:
        """Record a failure / overfitted pattern into memory to guide future mutations."""
        ast_node = parse_formula(formula)
        skeleton = extract_ast_skeleton(ast_node)

        constraint = FailureConstraint(
            formula=formula,
            ast_skeleton=skeleton,
            failure_reason=reason,
            metrics=metrics or {},
        )
        self.failure_constraints.append(constraint)

        if self.persistence_path:
            self.save_to_json(self.persistence_path)

        return constraint

    def is_redundant_or_constrained(
        self,
        candidate_formula: str,
        candidate_ast: Optional[SymbolicNode] = None,
        max_similarity: float = 0.85,
        max_skeleton_quota: int = 3,
    ) -> Tuple[bool, str]:
        """
        Check if a candidate factor is structurally redundant or matches known failure constraints.
        Returns (is_blocked, reason_string).
        """
        node = candidate_ast or parse_formula(candidate_formula)
        formula = candidate_formula.strip()

        # 1. Exact match in success or failure
        if formula in self.success_factors:
            return True, f"Exact formula already present in success memory (Rank IC={self.success_factors[formula].rank_ic:.4f})"

        for fc in self.failure_constraints:
            if fc.formula == formula:
                return True, f"Formula matches known failure constraint: {fc.failure_reason}"

        # 2. Skeleton quota check (prevent over-mining same structural skeleton)
        skel = extract_ast_skeleton(node)
        if self._skeleton_counts.get(skel, 0) >= max_skeleton_quota:
            return True, f"Skeleton '{skel}' exceeded maximum diversity quota ({max_skeleton_quota})"

        # 3. AST Jaccard similarity check against top active factors
        for exist_formula, pat in self.active_factor_pool.items():
            exist_node = parse_formula(exist_formula)
            sim = compute_ast_similarity(node, exist_node)
            if sim >= max_similarity:
                return True, f"AST similarity {sim:.2f} >= threshold {max_similarity} with active factor '{exist_formula}'"

        return False, "Candidate passed experience memory validation"

    def get_success_templates(self, top_n: int = 5) -> List[str]:
        """Return formulas of top-performing factors to use as LLM prompts / mutation seeds."""
        sorted_factors = sorted(
            self.success_factors.values(),
            key=lambda x: (abs(x.ir), abs(x.rank_ic)),
            reverse=True,
        )
        return [f.formula for f in sorted_factors[:top_n]]

    def get_top_factors(self, n: int = 10, sort_by: str = "ir") -> List[FactorPattern]:
        """Retrieve top factors sorted by IR or Rank IC."""
        if sort_by == "rank_ic":
            key_fn = lambda x: abs(x.rank_ic)
        else:
            key_fn = lambda x: abs(x.ir)
        return sorted(self.success_factors.values(), key=key_fn, reverse=True)[:n]

    def _classify_category(self, formula: str) -> str:
        f_lower = formula.lower()
        if "cvd" in f_lower or "imbalance" in f_lower:
            return "orderflow"
        elif "funding" in f_lower or "oi" in f_lower:
            return "liquidity_leverage"
        elif "ts_std" in f_lower or "high" in f_lower and "low" in f_lower:
            return "volatility"
        elif "ts_delta" in f_lower or "ts_rank" in f_lower:
            return "momentum"
        return "composite"

    def save_to_json(self, filepath: str) -> None:
        """Persist memory to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "success_factors": {k: v.to_dict() for k, v in self.success_factors.items()},
            "failure_constraints": [fc.to_dict() for fc in self.failure_constraints],
            "active_factor_pool": {k: v.to_dict() for k, v in self.active_factor_pool.items()},
            "skeleton_counts": self._skeleton_counts,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.debug("Saved experience memory to %s", filepath)

    def load_from_json(self, filepath: str) -> None:
        """Load memory from a JSON file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                payload = json.load(f)

            self.success_factors = {
                k: FactorPattern(**v) for k, v in payload.get("success_factors", {}).items()
            }
            self.failure_constraints = [
                FailureConstraint(**v) for v in payload.get("failure_constraints", [])
            ]
            self.active_factor_pool = {
                k: FactorPattern(**v) for k, v in payload.get("active_factor_pool", {}).items()
            }
            self._skeleton_counts = payload.get("skeleton_counts", {})
            logger.info("Loaded %d success factors from %s", len(self.success_factors), filepath)
        except Exception as e:
            logger.warning("Failed to load experience memory from %s: %s", filepath, e)

    def export_summary(self) -> Dict[str, Any]:
        return {
            "total_success_factors": len(self.success_factors),
            "total_failure_constraints": len(self.failure_constraints),
            "active_factor_pool_size": len(self.active_factor_pool),
            "top_categories": {
                cat: sum(1 for f in self.success_factors.values() if f.category == cat)
                for cat in {"orderflow", "liquidity_leverage", "volatility", "momentum", "composite"}
            },
        }
