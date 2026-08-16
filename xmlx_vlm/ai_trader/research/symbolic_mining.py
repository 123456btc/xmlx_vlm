"""
Symbolic Factor Mining & Genetic Discovery Engine (Symbolic Alpha Generator).

Features:
1. Vectorized AST-based Expression Tree with Time-Series & Cross-Sectional Operators.
2. Grammar primitives: ts_mean, ts_std, ts_delta, ts_max, ts_min, ts_rank, ts_decay_linear,
   ts_zscore, ts_corr, sub_ratio, ratio_diff, etc.
3. String-to-AST Formula Parser for seamless LLM formula interpretation.
4. Genetic operators: AST Subtree Crossover, Semantic Mutation, and Elitism.
"""

from __future__ import annotations

import ast
import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .factor_analyzer import FactorEvaluator, ICReport, _mean, _pearson_corr, _rank, _spearman_rank_corr, _std

logger = logging.getLogger(__name__)

# Available dataset field terminals
DEFAULT_FIELDS = ["open", "high", "low", "close", "volume", "cvd", "oi", "funding", "imbalance"]
DEFAULT_WINDOWS = [3, 5, 10, 20, 40]


def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-8 else 0.0


def _decay_linear_weights(w: int) -> np.ndarray:
    """Compute normalized linear decay weights [1, 2, ..., w] / sum."""
    weights = np.arange(1, w + 1, dtype=np.float64)
    return weights / np.sum(weights)


class SymbolicNode(ABC):
    """Abstract Base Class for AST nodes in Symbolic Factor Expressions."""

    @abstractmethod
    def eval(self, data: Dict[str, Any]) -> List[float]:
        """Evaluate node and return python List[float]."""
        pass

    @abstractmethod
    def eval_array(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Vectorized evaluation returning NumPy 1D array."""
        pass

    @abstractmethod
    def to_formula(self) -> str:
        """Return standardized string representation of the formula."""
        pass

    @abstractmethod
    def clone(self) -> SymbolicNode:
        """Deep copy of the node."""
        pass

    def get_all_nodes(self) -> List[SymbolicNode]:
        """Return all subnodes including self."""
        nodes: List[SymbolicNode] = [self]
        if isinstance(self, UnaryOpNode):
            nodes.extend(self.child.get_all_nodes())
        elif isinstance(self, BinaryOpNode):
            nodes.extend(self.left.get_all_nodes())
            nodes.extend(self.right.get_all_nodes())
        return nodes


@dataclass
class FieldNode(SymbolicNode):
    """Leaf node representing an input time-series field."""
    field_name: str

    def eval(self, data: Dict[str, Any]) -> List[float]:
        val = data.get(self.field_name, [])
        if isinstance(val, np.ndarray):
            return val.tolist()
        return list(val)

    def eval_array(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        val = data.get(self.field_name)
        if val is None:
            return np.array([], dtype=np.float64)
        if isinstance(val, np.ndarray):
            return val.astype(np.float64, copy=False)
        return np.asarray(val, dtype=np.float64)

    def to_formula(self) -> str:
        return self.field_name

    def clone(self) -> FieldNode:
        return FieldNode(field_name=self.field_name)


@dataclass
class ConstNode(SymbolicNode):
    """Leaf node representing a constant scalar."""
    value: float

    def eval(self, data: Dict[str, Any]) -> List[float]:
        n = len(next(iter(data.values()))) if data else 0
        return [self.value] * n

    def eval_array(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        n = len(next(iter(data.values()))) if data else 0
        return np.full(n, self.value, dtype=np.float64)

    def to_formula(self) -> str:
        return f"{self.value:.2f}"

    def clone(self) -> ConstNode:
        return ConstNode(value=self.value)


@dataclass
class UnaryOpNode(SymbolicNode):
    """Unary Operator Node (e.g. ts_mean, ts_std, ts_delta, ts_rank, ts_decay_linear, ts_zscore, log_abs, sign)."""
    op: str
    child: SymbolicNode
    param: int = 5

    def eval_array(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        s = self.child.eval_array(data)
        n = len(s)
        if n == 0:
            return np.array([], dtype=np.float64)

        w = max(1, self.param)
        ps = pd.Series(s)

        if self.op == "ts_mean":
            res = ps.rolling(window=w, min_periods=1).mean().to_numpy()
        elif self.op == "ts_std":
            res = ps.rolling(window=w, min_periods=1).std(ddof=1).fillna(0.0).to_numpy()
        elif self.op == "ts_delta":
            diff = ps.diff(periods=w)
            fallback = ps - ps.iloc[0]
            res = diff.fillna(fallback).to_numpy()
        elif self.op == "ts_max":
            res = ps.rolling(window=w, min_periods=1).max().to_numpy()
        elif self.op == "ts_min":
            res = ps.rolling(window=w, min_periods=1).min().to_numpy()
        elif self.op == "ts_rank":
            def _rolling_rank(window_vals: np.ndarray) -> float:
                curr = window_vals[-1]
                return float(np.sum(window_vals < curr) / len(window_vals))
            res = ps.rolling(window=w, min_periods=1).apply(_rolling_rank, raw=True).to_numpy()
        elif self.op == "ts_decay_linear":
            weights = _decay_linear_weights(w)
            def _w_mean(window_vals: np.ndarray) -> float:
                k = len(window_vals)
                if k == w:
                    return float(np.dot(window_vals, weights))
                sub_w = _decay_linear_weights(k)
                return float(np.dot(window_vals, sub_w))
            res = ps.rolling(window=w, min_periods=1).apply(_w_mean, raw=True).to_numpy()
        elif self.op == "ts_zscore":
            rmean = ps.rolling(window=w, min_periods=1).mean()
            rstd = ps.rolling(window=w, min_periods=1).std(ddof=1).fillna(1e-8)
            res = ((ps - rmean) / (rstd + 1e-8)).to_numpy()
        elif self.op == "abs":
            res = np.abs(s)
        elif self.op == "neg":
            res = -s
        elif self.op == "sign":
            res = np.sign(s)
        elif self.op == "log_abs":
            res = np.log(np.abs(s) + 1e-8)
        else:
            res = s

        return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)

    def eval(self, data: Dict[str, Any]) -> List[float]:
        arr_data = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}
        return self.eval_array(arr_data).tolist()

    def to_formula(self) -> str:
        if self.op.startswith("ts_"):
            return f"{self.op}({self.child.to_formula()}, {self.param})"
        return f"{self.op}({self.child.to_formula()})"

    def clone(self) -> UnaryOpNode:
        return UnaryOpNode(op=self.op, child=self.child.clone(), param=self.param)


@dataclass
class BinaryOpNode(SymbolicNode):
    """Binary Operator Node (e.g. add, sub, mul, div, ts_corr, sub_ratio)."""
    op: str
    left: SymbolicNode
    right: SymbolicNode
    param: int = 5

    def eval_array(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        s1 = self.left.eval_array(data)
        s2 = self.right.eval_array(data)
        n = min(len(s1), len(s2))
        if n == 0:
            return np.array([], dtype=np.float64)
        s1, s2 = s1[:n], s2[:n]
        w = max(2, self.param)

        if self.op == "add":
            res = s1 + s2
        elif self.op == "sub":
            res = s1 - s2
        elif self.op == "mul":
            res = s1 * s2
        elif self.op == "div":
            denom = np.where(np.abs(s2) > 1e-8, s2, np.nan)
            res = np.nan_to_num(s1 / denom, nan=0.0, posinf=0.0, neginf=0.0)
        elif self.op in ("sub_ratio", "ratio_diff"):
            diff = s1 - s2
            tot = np.abs(s1) + np.abs(s2)
            denom = np.where(tot > 1e-8, tot, np.nan)
            res = np.nan_to_num(diff / denom, nan=0.0, posinf=0.0, neginf=0.0)
        elif self.op == "ts_corr":
            ps1 = pd.Series(s1)
            ps2 = pd.Series(s2)
            res = ps1.rolling(window=w, min_periods=2).corr(ps2).fillna(0.0).to_numpy()
        elif self.op == "greater":
            res = np.where(s1 > s2, 1.0, 0.0)
        elif self.op == "less":
            res = np.where(s1 < s2, 1.0, 0.0)
        else:
            res = s1

        return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)

    def eval(self, data: Dict[str, Any]) -> List[float]:
        arr_data = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}
        return self.eval_array(arr_data).tolist()

    def to_formula(self) -> str:
        if self.op == "add":
            return f"({self.left.to_formula()} + {self.right.to_formula()})"
        elif self.op == "sub":
            return f"({self.left.to_formula()} - {self.right.to_formula()})"
        elif self.op == "mul":
            return f"({self.left.to_formula()} * {self.right.to_formula()})"
        elif self.op == "div":
            return f"({self.left.to_formula()} / {self.right.to_formula()})"
        elif self.op in ("sub_ratio", "ratio_diff"):
            return f"ratio_diff({self.left.to_formula()}, {self.right.to_formula()})"
        elif self.op == "ts_corr":
            return f"ts_corr({self.left.to_formula()}, {self.right.to_formula()}, {self.param})"
        return f"{self.op}({self.left.to_formula()}, {self.right.to_formula()})"

    def clone(self) -> BinaryOpNode:
        return BinaryOpNode(op=self.op, left=self.left.clone(), right=self.right.clone(), param=self.param)


# ============================================================================
# Formula Parser (String -> SymbolicNode AST)
# ============================================================================

def parse_formula(formula_str: str) -> SymbolicNode:
    """
    Parse a string formula expression into a SymbolicNode AST.
    Supports expressions like:
      - ts_mean(close, 5)
      - (close / volume)
      - ratio_diff(cvd, volume)
      - ts_corr(close, volume, 10)
      - ts_decay_linear((high - low), 10)
    """
    cleaned = formula_str.strip()
    try:
        parsed = ast.parse(cleaned, mode="eval")
        return _ast_to_symbolic_node(parsed.body)
    except Exception as e:
        logger.warning("Failed to parse formula '%s': %s", cleaned, e)
        return FieldNode(field_name=cleaned if cleaned in DEFAULT_FIELDS else "close")


def _ast_to_symbolic_node(node: ast.AST) -> SymbolicNode:
    if isinstance(node, ast.Name):
        return FieldNode(field_name=node.id)
    elif isinstance(node, ast.Constant):
        return ConstNode(value=float(node.value))
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return UnaryOpNode(op="neg", child=_ast_to_symbolic_node(node.operand))
        return _ast_to_symbolic_node(node.operand)
    elif isinstance(node, ast.BinOp):
        left = _ast_to_symbolic_node(node.left)
        right = _ast_to_symbolic_node(node.right)
        if isinstance(node.op, ast.Add):
            return BinaryOpNode(op="add", left=left, right=right)
        elif isinstance(node.op, ast.Sub):
            return BinaryOpNode(op="sub", left=left, right=right)
        elif isinstance(node.op, ast.Mult):
            return BinaryOpNode(op="mul", left=left, right=right)
        elif isinstance(node.op, ast.Div):
            return BinaryOpNode(op="div", left=left, right=right)
        else:
            return BinaryOpNode(op="add", left=left, right=right)
    elif isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else "ts_mean"
        args = node.args

        unary_ops = {"ts_mean", "ts_std", "ts_delta", "ts_max", "ts_min", "ts_rank",
                     "ts_decay_linear", "ts_zscore", "abs", "neg", "sign", "log_abs"}
        binary_ops = {"ts_corr", "sub_ratio", "ratio_diff", "add", "sub", "mul", "div", "greater", "less"}

        if func_name in unary_ops:
            child = _ast_to_symbolic_node(args[0]) if len(args) > 0 else FieldNode("close")
            param = int(args[1].value) if len(args) > 1 and isinstance(args[1], ast.Constant) else 5
            return UnaryOpNode(op=func_name, child=child, param=param)
        elif func_name in binary_ops:
            left = _ast_to_symbolic_node(args[0]) if len(args) > 0 else FieldNode("close")
            right = _ast_to_symbolic_node(args[1]) if len(args) > 1 else FieldNode("volume")
            param = int(args[2].value) if len(args) > 2 and isinstance(args[2], ast.Constant) else 5
            return BinaryOpNode(op=func_name, left=left, right=right, param=param)
        else:
            child = _ast_to_symbolic_node(args[0]) if len(args) > 0 else FieldNode("close")
            return UnaryOpNode(op="ts_mean", child=child, param=5)

    return FieldNode("close")


# ============================================================================
# Genetic Crossover & Mutation Operators
# ============================================================================

def crossover_trees(tree1: SymbolicNode, tree2: SymbolicNode) -> Tuple[SymbolicNode, SymbolicNode]:
    """
    Subtree Crossover operator: randomly swap subtrees between two parent ASTs.
    Returns two offspring ASTs.
    """
    t1 = tree1.clone()
    t2 = tree2.clone()

    nodes1 = t1.get_all_nodes()
    nodes2 = t2.get_all_nodes()

    target1 = random.choice(nodes1)
    target2 = random.choice(nodes2)

    def _replace_subnode(root: SymbolicNode, old_node: SymbolicNode, new_node: SymbolicNode) -> SymbolicNode:
        if root is old_node:
            return new_node.clone()
        if isinstance(root, UnaryOpNode):
            return UnaryOpNode(op=root.op, child=_replace_subnode(root.child, old_node, new_node), param=root.param)
        elif isinstance(root, BinaryOpNode):
            return BinaryOpNode(
                op=root.op,
                left=_replace_subnode(root.left, old_node, new_node),
                right=_replace_subnode(root.right, old_node, new_node),
                param=root.param,
            )
        return root

    offspring1 = _replace_subnode(t1, target1, target2)
    offspring2 = _replace_subnode(t2, target2, target1)
    return offspring1, offspring2


class SymbolicFactorGenerator:
    """Genetic Symbolic Alpha Formula Mining Engine with Vectorized Acceleration."""

    def __init__(
        self,
        fields: Optional[List[str]] = None,
        windows: Optional[List[int]] = None,
        random_seed: Optional[int] = None,
    ):
        self.fields = fields or DEFAULT_FIELDS
        self.windows = windows or DEFAULT_WINDOWS
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def generate_random_tree(self, depth: int = 2) -> SymbolicNode:
        """Generate a random AST expression tree up to maximum depth."""
        if depth <= 0:
            return FieldNode(random.choice(self.fields))

        choice = random.random()
        if choice < 0.20:
            return FieldNode(random.choice(self.fields))
        elif choice < 0.55:
            op = random.choice([
                "ts_mean", "ts_std", "ts_delta", "ts_rank", "ts_max", "ts_min",
                "ts_decay_linear", "ts_zscore", "abs", "neg", "sign"
            ])
            param = random.choice(self.windows)
            child = self.generate_random_tree(depth - 1)
            return UnaryOpNode(op=op, child=child, param=param)
        else:
            op = random.choice(["add", "sub", "mul", "div", "sub_ratio", "ts_corr"])
            param = random.choice(self.windows)
            left = self.generate_random_tree(depth - 1)
            right = self.generate_random_tree(depth - 1)
            return BinaryOpNode(op=op, left=left, right=right, param=param)

    def mutate_tree(self, node: SymbolicNode, mutation_prob: float = 0.3) -> SymbolicNode:
        """Randomly mutate nodes within an AST tree."""
        if random.random() < mutation_prob:
            return self.generate_random_tree(depth=2)

        if isinstance(node, UnaryOpNode):
            new_child = self.mutate_tree(node.child, mutation_prob)
            new_param = random.choice(self.windows) if random.random() < 0.3 else node.param
            new_op = random.choice(["ts_mean", "ts_std", "ts_delta", "ts_rank", "ts_decay_linear", "ts_zscore"]) if random.random() < 0.2 else node.op
            return UnaryOpNode(op=new_op, child=new_child, param=new_param)
        elif isinstance(node, BinaryOpNode):
            new_left = self.mutate_tree(node.left, mutation_prob)
            new_right = self.mutate_tree(node.right, mutation_prob)
            new_param = random.choice(self.windows) if random.random() < 0.3 else node.param
            new_op = random.choice(["add", "sub", "mul", "div", "sub_ratio", "ts_corr"]) if random.random() < 0.2 else node.op
            return BinaryOpNode(op=new_op, left=new_left, right=new_right, param=new_param)

        return node

    def mine_factors(
        self,
        data: Dict[str, Union[List[float], np.ndarray]],
        forward_returns: Union[List[float], np.ndarray],
        base_factors: Optional[List[Union[List[float], np.ndarray]]] = None,
        population_size: int = 30,
        generations: int = 3,
        min_rank_ic: float = 0.02,
        initial_seeds: Optional[List[Union[str, SymbolicNode]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run genetic alpha mining loop and return top-performing discovered factor expressions.
        Supports initial seed injection (e.g. from LLM / Experience Memory) and crossover.
        """
        arr_data = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}
        arr_returns = np.asarray(forward_returns, dtype=np.float64).tolist()
        arr_base = [np.asarray(bf, dtype=np.float64).tolist() for bf in base_factors] if base_factors else None

        population: List[SymbolicNode] = []

        if initial_seeds:
            for s in initial_seeds:
                if isinstance(s, str):
                    population.append(parse_formula(s))
                elif isinstance(s, SymbolicNode):
                    population.append(s.clone())

        while len(population) < population_size:
            population.append(self.generate_random_tree(depth=2))

        discovered: List[Dict[str, Any]] = []

        for gen in range(generations):
            evaluated: List[Tuple[SymbolicNode, float, Any]] = []

            for tree in population:
                try:
                    factor_vals = tree.eval_array(arr_data).tolist()
                    if len(set(factor_vals[:50])) <= 1:
                        continue

                    if arr_base:
                        inc_rep = FactorEvaluator.evaluate_incremental_alpha(
                            candidate_factor=factor_vals,
                            base_factors=arr_base,
                            forward_returns=arr_returns,
                            rolling_window=20,
                            candidate_name=tree.to_formula(),
                        )
                        score = abs(inc_rep.residual_ic.mean_rank_ic)
                        evaluated.append((tree, score, inc_rep))
                    else:
                        rep = FactorEvaluator.evaluate_time_series_ic(factor_vals, arr_returns, rolling_window=20)
                        score = abs(rep.mean_rank_ic)
                        evaluated.append((tree, score, rep))
                except Exception as e:
                    logger.debug("Tree evaluation failed: %s", e)

            evaluated.sort(key=lambda x: x[1], reverse=True)

            for tree, score, report_obj in evaluated:
                if score >= min_rank_ic:
                    formula = tree.to_formula()
                    if not any(d["formula"] == formula for d in discovered):
                        if arr_base:
                            discovered.append({
                                "formula": formula,
                                "rank_ic": report_obj.raw_ic.mean_rank_ic,
                                "residual_rank_ic": report_obj.residual_ic.mean_rank_ic,
                                "r_squared_overlap": report_obj.r_squared_overlap,
                                "is_true_incremental_alpha": report_obj.is_true_incremental_alpha,
                                "ir": report_obj.residual_ic.ir,
                                "t_stat": report_obj.residual_ic.t_stat,
                                "is_significant": report_obj.residual_ic.is_significant,
                            })
                        else:
                            discovered.append({
                                "formula": formula,
                                "rank_ic": report_obj.mean_rank_ic,
                                "ic": report_obj.mean_ic,
                                "ir": report_obj.ir,
                                "t_stat": report_obj.t_stat,
                                "is_significant": report_obj.is_significant,
                            })

            survivors = [t[0] for t in evaluated[: max(2, population_size // 4)]]
            if not survivors:
                survivors = [self.generate_random_tree(depth=2) for _ in range(5)]

            next_pop = [t.clone() for t in survivors]
            while len(next_pop) < population_size:
                if len(survivors) >= 2 and random.random() < 0.4:
                    p1, p2 = random.sample(survivors, 2)
                    c1, _ = crossover_trees(p1, p2)
                    next_pop.append(c1)
                else:
                    parent = random.choice(survivors)
                    mutated = self.mutate_tree(parent)
                    next_pop.append(mutated)

            population = next_pop

        sort_key = "residual_rank_ic" if base_factors else "rank_ic"
        discovered.sort(key=lambda x: abs(x.get(sort_key, 0.0)), reverse=True)
        return discovered
