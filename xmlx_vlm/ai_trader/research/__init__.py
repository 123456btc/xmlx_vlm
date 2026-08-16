"""AI Trader Quantitative Research, Backtesting & Factor Mining Module."""

from .backtest import BacktestConfig, BacktestEngine, BacktestResult, Position, TearSheet, TradeRecord
from .evolutionary_miner import (
    EvolvingAlphaEngine,
    EvolutionaryMiningConfig,
)
from .factor_analyzer import (
    FactorEvaluator,
    ICReport,
    IncrementalAlphaReport,
    QuantileReport,
    gram_schmidt_orthogonalize,
    symmetric_orthogonalize,
)
from .factor_memory import (
    ExperienceMemory,
    FactorPattern,
    FailureConstraint,
    compute_ast_similarity,
    extract_ast_skeleton,
)
from .llm_alpha_agent import LLMAlphaResearcher, QuantitativeHypothesis
from .symbolic_mining import (
    BinaryOpNode,
    ConstNode,
    FieldNode,
    SymbolicFactorGenerator,
    SymbolicNode,
    UnaryOpNode,
    crossover_trees,
    parse_formula,
)

__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    "TearSheet",
    "Position",
    "TradeRecord",
    "FactorEvaluator",
    "ICReport",
    "QuantileReport",
    "IncrementalAlphaReport",
    "gram_schmidt_orthogonalize",
    "symmetric_orthogonalize",
    "SymbolicFactorGenerator",
    "SymbolicNode",
    "UnaryOpNode",
    "BinaryOpNode",
    "FieldNode",
    "ConstNode",
    "parse_formula",
    "crossover_trees",
    "ExperienceMemory",
    "FactorPattern",
    "FailureConstraint",
    "extract_ast_skeleton",
    "compute_ast_similarity",
    "LLMAlphaResearcher",
    "QuantitativeHypothesis",
    "EvolvingAlphaEngine",
    "EvolutionaryMiningConfig",
]
