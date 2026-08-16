"""
Self-Evolving Multi-Objective Alpha Mining Engine (AlphaGen & QuantaAlpha Architecture).

Features:
1. End-to-End Orchestration: Unifies LLM Researcher, Experience Memory, AST Evolution & Vectorized Backtesting.
2. Portfolio Synergy Reward: Evaluates candidate factor contribution to overall portfolio Sharpe ratio.
3. Multi-Objective Fitness: Balances Rank IC, Information Ratio (IR), Synergy Sharpe Delta, and Turnover.
4. Trajectory-Level Evolution: Continuous generation tracking, memory updates, and executable factor export.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .factor_analyzer import (
    FactorEvaluator,
    ICReport,
    _mean,
    _pearson_corr,
    _rank,
    _spearman_rank_corr,
    _std,
    gram_schmidt_orthogonalize,
)
from .factor_memory import ExperienceMemory, FactorPattern
from .llm_alpha_agent import LLMAlphaResearcher, QuantitativeHypothesis
from .symbolic_mining import (
    DEFAULT_FIELDS,
    DEFAULT_WINDOWS,
    BinaryOpNode,
    ConstNode,
    FieldNode,
    SymbolicFactorGenerator,
    SymbolicNode,
    UnaryOpNode,
    crossover_trees,
    parse_formula,
)

logger = logging.getLogger(__name__)


@dataclass
class EvolutionaryMiningConfig:
    """Hyperparameters for Self-Evolving Alpha Mining."""
    population_size: int = 30
    generations: int = 3
    min_rank_ic: float = 0.02
    min_ir: float = 0.20
    max_factor_overlap_r2: float = 0.60
    crossover_prob: float = 0.40
    mutation_prob: float = 0.35
    llm_seed_count: int = 3
    synergy_weight: float = 1.0
    turnover_penalty_weight: float = 0.05
    rolling_window: int = 20
    max_active_pool_size: int = 10


def _calculate_series_turnover(series: List[float]) -> float:
    """Estimate factor rank turnover: mean absolute change in normalized rank per bar."""
    if len(series) < 2:
        return 0.0
    ranks = _rank(series)
    n = len(ranks)
    normalized = [r / n for r in ranks]
    diffs = [abs(normalized[i] - normalized[i - 1]) for i in range(1, n)]
    return float(_mean(diffs))


def _calculate_sharpe(returns: List[float], annualization_factor: float = math.sqrt(252 * 24)) -> float:
    """Compute annualized Sharpe Ratio of a returns series."""
    if len(returns) < 5:
        return 0.0
    m = _mean(returns)
    s = _std(returns)
    if s < 1e-9:
        return 0.0
    return float((m / s) * annualization_factor)


class EvolvingAlphaEngine:
    """
    Unified Self-Evolving Alpha Factor & Strategy Mining Engine.
    """

    def __init__(
        self,
        memory: Optional[ExperienceMemory] = None,
        researcher: Optional[LLMAlphaResearcher] = None,
        config: Optional[EvolutionaryMiningConfig] = None,
    ):
        self.memory = memory or ExperienceMemory()
        self.researcher = researcher or LLMAlphaResearcher(memory=self.memory)
        self.config = config or EvolutionaryMiningConfig()
        self.generator = SymbolicFactorGenerator()

    def evaluate_synergy_sharpe(
        self,
        candidate_values: List[float],
        active_factors: Dict[str, List[float]],
        forward_returns: List[float],
    ) -> Tuple[float, float]:
        """
        Compute portfolio synergy Sharpe and incremental Sharpe delta (AlphaGen Synergy Reward).
        Returns: (composite_sharpe, delta_sharpe)
        """
        n = len(forward_returns)
        if n < 20:
            return 0.0, 0.0

        arr_returns = np.asarray(forward_returns, dtype=np.float64)

        if not active_factors:
            # Baseline is candidate alone
            cand_arr = np.asarray(candidate_values, dtype=np.float64)
            strat_ret = np.sign(cand_arr) * arr_returns
            cand_sharpe = _calculate_sharpe(strat_ret.tolist())
            return cand_sharpe, cand_sharpe

        # Build design matrix of active factors
        active_keys = list(active_factors.keys())
        X_active = np.column_stack([np.asarray(active_factors[k], dtype=np.float64)[:n] for k in active_keys])

        # Normalize features
        X_mean = np.mean(X_active, axis=0)
        X_std = np.std(X_active, axis=0) + 1e-8
        X_norm = (X_active - X_mean) / X_std

        # Baseline active composite (equal weight or linear combination)
        baseline_signal = np.mean(X_norm, axis=1)
        base_strat_ret = np.sign(baseline_signal) * arr_returns
        base_sharpe = _calculate_sharpe(base_strat_ret.tolist())

        # Combine with candidate factor
        c_arr = np.asarray(candidate_values, dtype=np.float64)[:n]
        c_norm = (c_arr - np.mean(c_arr)) / (np.std(c_arr) + 1e-8)

        X_new = np.column_stack([X_norm, c_norm])
        new_signal = np.mean(X_new, axis=1)
        new_strat_ret = np.sign(new_signal) * arr_returns
        new_sharpe = _calculate_sharpe(new_strat_ret.tolist())

        delta_sharpe = new_sharpe - base_sharpe
        return new_sharpe, delta_sharpe

    def compute_multi_objective_fitness(
        self,
        rank_ic: float,
        ir: float,
        delta_sharpe: float,
        turnover: float,
    ) -> float:
        """
        Multi-objective fitness function:
        Fitness = |Rank IC| + 0.5 * max(0, IR) + w_synergy * max(0, Delta Sharpe) - w_turnover * turnover
        """
        score = (
            abs(rank_ic) * 1.0
            + max(0.0, ir) * 0.5
            + max(0.0, delta_sharpe) * self.config.synergy_weight
            - turnover * self.config.turnover_penalty_weight
        )
        return float(score)

    def evolve_factors(
        self,
        data: Dict[str, Union[List[float], np.ndarray]],
        forward_returns: Union[List[float], np.ndarray],
        market_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute full self-evolving alpha discovery cycle across generations.
        """
        start_time = time.time()
        arr_data = {k: np.asarray(v, dtype=np.float64) for k, v in data.items()}
        arr_returns = np.asarray(forward_returns, dtype=np.float64).tolist()

        # Step 1: LLM Alpha Researcher generates domain hypotheses & formula seeds
        hypotheses = self.researcher.generate_hypotheses(market_summary, count=self.config.llm_seed_count)
        llm_seeds = [h.formula_candidate for h in hypotheses]

        # Step 2: Seed experience memory top templates into population
        mem_templates = self.memory.get_success_templates(top_n=3)

        initial_seeds = list(set(llm_seeds + mem_templates))

        # Build initial population
        population: List[SymbolicNode] = []
        for s in initial_seeds:
            is_blocked, _ = self.memory.is_redundant_or_constrained(s)
            if not is_blocked:
                population.append(parse_formula(s))

        while len(population) < self.config.population_size:
            population.append(self.generator.generate_random_tree(depth=2))

        # Tracking across generations
        generation_history: List[Dict[str, Any]] = []
        discovered_pool: Dict[str, Dict[str, Any]] = {}
        active_factor_series: Dict[str, List[float]] = {}

        # Prepopulate active series from existing memory active pool
        for k, pat in self.memory.active_factor_pool.items():
            try:
                node = parse_formula(k)
                active_factor_series[k] = node.eval_array(arr_data).tolist()
            except Exception:
                pass

        for gen in range(self.config.generations):
            evaluated: List[Tuple[SymbolicNode, float, Dict[str, Any]]] = []

            for tree in population:
                formula = tree.to_formula()
                # Pre-evaluation memory pruning
                is_blocked, block_reason = self.memory.is_redundant_or_constrained(formula, tree)
                if is_blocked:
                    logger.debug("Pruning blocked candidate '%s': %s", formula, block_reason)
                    continue

                try:
                    vals = tree.eval_array(arr_data).tolist()
                    if len(set(vals[:50])) <= 1:
                        continue

                    # Single factor IC / IR evaluation
                    ic_rep = FactorEvaluator.evaluate_time_series_ic(
                        vals, arr_returns, rolling_window=self.config.rolling_window
                    )

                    # Incremental orthogonalization against active pool
                    base_list = list(active_factor_series.values())
                    if base_list:
                        inc_rep = FactorEvaluator.evaluate_incremental_alpha(
                            candidate_factor=vals,
                            base_factors=base_list,
                            forward_returns=arr_returns,
                            rolling_window=self.config.rolling_window,
                            candidate_name=formula,
                        )
                        r2_overlap = inc_rep.r_squared_overlap
                        residual_rank_ic = inc_rep.residual_ic.mean_rank_ic
                        is_inc = inc_rep.is_true_incremental_alpha
                    else:
                        r2_overlap = 0.0
                        residual_rank_ic = ic_rep.mean_rank_ic
                        is_inc = True

                    # Portfolio synergy Sharpe & Turnover
                    comp_sharpe, delta_sharpe = self.evaluate_synergy_sharpe(
                        candidate_values=vals,
                        active_factors=active_factor_series,
                        forward_returns=arr_returns,
                    )
                    turnover = _calculate_series_turnover(vals)

                    # Multi-objective fitness
                    fitness = self.compute_multi_objective_fitness(
                        rank_ic=residual_rank_ic,
                        ir=ic_rep.ir,
                        delta_sharpe=delta_sharpe,
                        turnover=turnover,
                    )

                    metrics = {
                        "formula": formula,
                        "rank_ic": ic_rep.mean_rank_ic,
                        "residual_rank_ic": residual_rank_ic,
                        "ir": ic_rep.ir,
                        "t_stat": ic_rep.t_stat,
                        "is_significant": ic_rep.is_significant,
                        "r_squared_overlap": r2_overlap,
                        "is_true_incremental_alpha": is_inc,
                        "synergy_sharpe": comp_sharpe,
                        "delta_sharpe": delta_sharpe,
                        "turnover": turnover,
                        "fitness": fitness,
                        "generation": gen,
                    }

                    evaluated.append((tree, fitness, metrics))

                except Exception as e:
                    logger.debug("Evaluation error for tree '%s': %s", formula, e)

            # Sort by multi-objective fitness
            evaluated.sort(key=lambda x: x[1], reverse=True)

            gen_best_fitness = evaluated[0][1] if evaluated else 0.0
            gen_discovered_count = 0

            # Store high-performing discoveries and register to memory
            for tree, fit_score, m in evaluated:
                if (
                    abs(m["residual_rank_ic"]) >= self.config.min_rank_ic
                    and m["ir"] >= self.config.min_ir
                    and m["r_squared_overlap"] <= self.config.max_factor_overlap_r2
                ):
                    f_name = m["formula"]
                    if f_name not in discovered_pool:
                        discovered_pool[f_name] = m
                        gen_discovered_count += 1
                        # Register in Experience Memory
                        self.memory.record_success(m, is_active_pool=len(active_factor_series) < self.config.max_active_pool_size)
                        if len(active_factor_series) < self.config.max_active_pool_size:
                            active_factor_series[f_name] = tree.eval_array(arr_data).tolist()
                elif abs(m["rank_ic"]) < 0.01 or m["turnover"] > 0.8:
                    # Register failure constraint
                    self.memory.record_failure(m["formula"], reason="low_ic_or_high_turnover", metrics=m)

            generation_history.append({
                "generation": gen,
                "population_evaluated": len(evaluated),
                "best_fitness": gen_best_fitness,
                "new_factors_discovered": gen_discovered_count,
            })

            # Next generation offspring creation (Elitism + Crossover + Semantic Mutation)
            survivors = [t[0] for t in evaluated[: max(2, self.config.population_size // 4)]]
            if not survivors:
                survivors = [self.generator.generate_random_tree(depth=2) for _ in range(5)]

            next_pop = [t.clone() for t in survivors]

            while len(next_pop) < self.config.population_size:
                if len(survivors) >= 2 and random.random() < self.config.crossover_prob:
                    p1, p2 = random.sample(survivors, 2)
                    c1, _ = crossover_trees(p1, p2)
                    next_pop.append(c1)
                else:
                    parent = random.choice(survivors)
                    mutated = self.generator.mutate_tree(parent, mutation_prob=self.config.mutation_prob)
                    next_pop.append(mutated)

            population = next_pop

        elapsed = time.time() - start_time
        sorted_discoveries = sorted(discovered_pool.values(), key=lambda x: x["fitness"], reverse=True)

        return {
            "elapsed_seconds": elapsed,
            "generations_completed": self.config.generations,
            "total_factors_discovered": len(sorted_discoveries),
            "top_factors": sorted_discoveries[: self.config.max_active_pool_size],
            "generation_history": generation_history,
            "experience_memory_summary": self.memory.export_summary(),
        }
