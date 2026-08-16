"""
Factor Effectiveness Analysis Engine (IC / Rank IC / IR & Quantile Layering Analysis).

Features:
1. Pearson IC, Spearman Rank IC, Information Ratio (IR), and Student's t-statistic test.
2. Cross-Sectional Multi-Asset Rolling IC time series.
3. Quantile (e.g. 5-Quantile / Decile) layering monotonicity returns & Long-Short spread.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(max(0.0, var))


def _pearson_corr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        return 0.0
    mx, my = _mean(x), _mean(y)
    sx, sy = _std(x), _std(y)
    if sx <= 1e-9 or sy <= 1e-9:
        return 0.0
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(len(x))) / (len(x) - 1)
    corr = cov / (sx * sy)
    return max(-1.0, min(1.0, corr))


def _rank(values: List[float]) -> List[float]:
    """Compute fractional ranks for a list of values."""
    n = len(values)
    if n == 0:
        return []
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
            j += 1
        # average rank for ties
        avg_r = (i + j + 2) / 2.0  # 1-based rank
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_r
        i = j + 1
    return ranks


def _spearman_rank_corr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    return _pearson_corr(rx, ry)


def _standardize(values: List[float]) -> List[float]:
    """Standardize series to mean=0, std=1."""
    if not values:
        return []
    m = _mean(values)
    s = _std(values)
    if s < 1e-9:
        return [0.0] * len(values)
    return [(x - m) / s for x in values]


def _dot_product(u: List[float], v: List[float]) -> float:
    return sum(a * b for a, b in zip(u, v))


def _jacobi_eigenvalues(A: List[List[float]], max_iter: int = 50, tol: float = 1e-8) -> Tuple[List[float], List[List[float]]]:
    """Jacobi eigenvalue algorithm for real symmetric matrices. Returns (eigenvalues, eigenvectors)."""
    n = len(A)
    a = [[A[i][j] for j in range(n)] for i in range(n)]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_iter):
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > max_val:
                    max_val = abs(a[i][j])
                    p, q = i, j

        if max_val < tol:
            break

        diff = a[q][q] - a[p][p]
        if abs(a[p][q]) < 1e-12:
            t = 0.0
        else:
            phi = diff / (2.0 * a[p][q])
            t = 1.0 / (abs(phi) + math.sqrt(phi * phi + 1.0))
            if phi < 0:
                t = -t
        c = 1.0 / math.sqrt(t * t + 1.0)
        s = t * c
        tau = s / (1.0 + c)

        apq = a[p][q]
        a[p][q] = 0.0
        a[q][p] = 0.0
        a[p][p] -= t * apq
        a[q][q] += t * apq

        for i in range(n):
            if i != p and i != q:
                aip = a[i][p]
                aiq = a[i][q]
                a[i][p] = aip - s * (aiq + tau * aip)
                a[p][i] = a[i][p]
                a[i][q] = aiq + s * (aip - tau * aiq)
                a[q][i] = a[i][q]

        for i in range(n):
            vip = v[i][p]
            viq = v[i][q]
            v[i][p] = vip - s * (viq + tau * vip)
            v[i][q] = viq + s * (vip - tau * viq)

    eigenvalues = [a[i][i] for i in range(n)]
    return eigenvalues, v


def gram_schmidt_orthogonalize(
    candidate: List[float],
    base_factors: List[List[float]],
) -> List[float]:
    """
    Modified Gram-Schmidt Orthogonalization (MGS).
    Projects candidate factor onto the orthogonal complement of the subspace spanned by base_factors.
    Guarantees that the output has ~0.0 Pearson correlation with all base_factors.
    """
    if not candidate:
        return []
    if not base_factors:
        return _standardize(candidate)

    v = _standardize(candidate)
    n = len(v)

    basis: List[List[float]] = []
    for raw_base in base_factors:
        if len(raw_base) != n:
            continue
        u = _standardize(raw_base)
        for b in basis:
            proj = _dot_product(u, b)
            u = [u[i] - proj * b[i] for i in range(n)]
        
        norm_sq = _dot_product(u, u)
        if norm_sq > 1e-9:
            norm = math.sqrt(norm_sq)
            b_norm = [u[i] / norm for i in range(n)]
            basis.append(b_norm)

    res = list(v)
    for b in basis:
        proj = _dot_product(res, b)
        res = [res[i] - proj * b[i] for i in range(n)]

    return _standardize(res)


def symmetric_orthogonalize(factors: List[List[float]]) -> List[List[float]]:
    """
    Symmetric (Löwdin) Orthogonalization for multiple factors: F_ortho = F * (F^T * F)^(-1/2).
    Keeps factors as close as possible to original factors while achieving mutual orthogonality.
    """
    k = len(factors)
    if k == 0:
        return []
    n = len(factors[0])
    if k == 1:
        return [_standardize(factors[0])]

    norm_factors = [_standardize(f) for f in factors]

    C = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            if i == j:
                C[i][j] = 1.0
            else:
                C[i][j] = _pearson_corr(norm_factors[i], norm_factors[j])

    eigenvalues, eigenvectors = _jacobi_eigenvalues(C)

    inv_sqrt_diag = [1.0 / math.sqrt(max(1e-6, val)) for val in eigenvalues]
    
    W = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            W[i][j] = sum(eigenvectors[i][m] * inv_sqrt_diag[m] * eigenvectors[j][m] for m in range(k))

    ortho_factors = []
    for j in range(k):
        ortho_j = [0.0] * n
        for t in range(n):
            ortho_j[t] = sum(norm_factors[i][t] * W[i][j] for i in range(k))
        ortho_factors.append(_standardize(ortho_j))

    return ortho_factors


@dataclass
class IncrementalAlphaReport:
    """Incremental Alpha & Orthogonality Evaluation Report."""

    candidate_name: str
    raw_ic: ICReport
    residual_ic: ICReport
    r_squared_overlap: float  # Fraction of variance explained by existing base factors (0.0 to 1.0)
    max_correlation_with_base: float
    is_true_incremental_alpha: bool  # True if residual_ic is significant and r_squared_overlap < 0.85

    def summary(self) -> str:
        status_str = "💎 真正的纯增量 Alpha (独立有效)" if self.is_true_incremental_alpha else "⚠️ 伪增量因子 (与已有因子共线性过高/剥离后失效)"
        return (
            f"=== 🧬 因子正交化与增量 Alpha 检验报告 ===\n"
            f"• 评估标的: {self.candidate_name} | 判定: {status_str}\n"
            f"• 原始 Rank IC: {self.raw_ic.mean_rank_ic:+.4f} (t={self.raw_ic.t_stat:.2f})\n"
            f"• 正交残差 Rank IC: {self.residual_ic.mean_rank_ic:+.4f} (t={self.residual_ic.t_stat:.2f})\n"
            f"• 既有因子重合度 (R²): {self.r_squared_overlap * 100:.2f}% | 最大相关系数: {self.max_correlation_with_base:+.2f}"
        )


@dataclass
class ICReport:
    """Information Coefficient (IC) & Predictive Power Report."""

    mean_ic: float
    std_ic: float
    ir: float  # Information Ratio = mean_ic / std_ic
    mean_rank_ic: float
    std_rank_ic: float
    rank_ir: float
    t_stat: float
    p_value_approx: float
    sample_size: int
    is_significant: bool  # |t_stat| > 2.0 and |mean_rank_ic| > 0.02
    ic_series: List[float] = field(default_factory=list)
    rank_ic_series: List[float] = field(default_factory=list)

    def summary(self) -> str:
        sig_str = "✅ 统计显著 (有效 Alpha)" if self.is_significant else "❌ 统计不显著 (伪因子/噪音)"
        return (
            f"=== 🔬 因子 IC / Rank IC 检验报告 ===\n"
            f"• 样本期数: {self.sample_size} | 状态: {sig_str}\n"
            f"• Mean IC: {self.mean_ic:+.4f} | IC 波动率: {self.std_ic:.4f} | IR (信息比率): {self.ir:.2f}\n"
            f"• Rank IC (秩相关): {self.mean_rank_ic:+.4f} | Rank IR: {self.rank_ir:.2f}\n"
            f"• t-统计量: {self.t_stat:.2f} (p-value ≈ {self.p_value_approx:.4f})"
        )


@dataclass
class QuantileReport:
    """Quantile Layering Monotonicity & Long-Short Return Report."""

    num_quantiles: int
    quantile_returns: Dict[str, float]  # Q1 to Q5 cumulative returns
    long_short_spread_pct: float
    monotonicity_score: float  # Rank correlation between Quantile index and Return (-1.0 to 1.0)
    is_monotonic: bool         # |monotonicity_score| >= 0.80
    quantile_curves: Dict[str, List[float]] = field(default_factory=dict)

    def summary(self) -> str:
        q_lines = " | ".join(f"{k}: {v:+.2f}%" for k, v in self.quantile_returns.items())
        mono_str = "✅ 单调性优秀 (强选股能力)" if self.is_monotonic else "⚠️ 单调性较弱"
        return (
            f"=== 📊 因子分层收益与单调性分析 ({self.num_quantiles} 分位数) ===\n"
            f"• 各组收益: {q_lines}\n"
            f"• 多空对冲收益 (Long-Short Spread): {self.long_short_spread_pct:+.2f}%\n"
            f"• 单调性得分 (Monotonicity Score): {self.monotonicity_score:+.2f} | 评估: {mono_str}"
        )


class FactorEvaluator:
    """Quantitative Factor Research & Evaluation Suite."""

    @staticmethod
    def evaluate_time_series_ic(
        factor_values: List[float],
        forward_returns: List[float],
        rolling_window: int = 30,
    ) -> ICReport:
        """
        Evaluate predictive power for a single asset's factor series against forward returns.
        """
        n = min(len(factor_values), len(forward_returns))
        if n < rolling_window + 5:
            return ICReport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, n, False)

        ic_series = []
        rank_ic_series = []

        for i in range(rolling_window, n):
            f_sub = factor_values[i - rolling_window : i]
            r_sub = forward_returns[i - rolling_window : i]
            ic = _pearson_corr(f_sub, r_sub)
            rank_ic = _spearman_rank_corr(f_sub, r_sub)
            ic_series.append(ic)
            rank_ic_series.append(rank_ic)

        return FactorEvaluator._build_ic_report(ic_series, rank_ic_series)

    @staticmethod
    def evaluate_cross_sectional_ic(
        factor_matrix: Dict[str, List[float]],
        returns_matrix: Dict[str, List[float]],
    ) -> ICReport:
        """
        Cross-sectional IC evaluation across multiple assets at each timestamp T.
        factor_matrix: symbol -> list of factor values of length T
        returns_matrix: symbol -> list of forward return values of length T
        """
        symbols = [s for s in factor_matrix.keys() if s in returns_matrix]
        if not symbols or len(symbols) < 3:
            return ICReport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, False)

        time_len = min(len(factor_matrix[s]) for s in symbols)
        ic_series = []
        rank_ic_series = []

        for t in range(time_len):
            f_cross = [factor_matrix[s][t] for s in symbols]
            r_cross = [returns_matrix[s][t] for s in symbols]

            # Filter valid numbers
            valid = [(f, r) for f, r in zip(f_cross, r_cross) if not (math.isnan(f) or math.isnan(r))]
            if len(valid) >= 3:
                fv = [v[0] for v in valid]
                rv = [v[1] for v in valid]
                ic = _pearson_corr(fv, rv)
                rank_ic = _spearman_rank_corr(fv, rv)
                ic_series.append(ic)
                rank_ic_series.append(rank_ic)

        return FactorEvaluator._build_ic_report(ic_series, rank_ic_series)

    @staticmethod
    def _build_ic_report(ic_series: List[float], rank_ic_series: List[float]) -> ICReport:
        if not ic_series:
            return ICReport(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, False)

        m_ic = _mean(ic_series)
        s_ic = _std(ic_series)
        ir = (m_ic / s_ic) if s_ic > 1e-8 else 0.0

        m_rank_ic = _mean(rank_ic_series)
        s_rank_ic = _std(rank_ic_series)
        rank_ir = (m_rank_ic / s_rank_ic) if s_rank_ic > 1e-8 else 0.0

        n = len(ic_series)
        t_stat = (m_rank_ic / (s_rank_ic / math.sqrt(n))) if s_rank_ic > 1e-8 and n > 1 else 0.0
        
        # Approximate two-tailed p-value using normal distribution approximation
        p_val = max(0.0001, min(1.0, 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))))
        is_sig = abs(t_stat) >= 2.0 and abs(m_rank_ic) >= 0.02

        return ICReport(
            mean_ic=round(m_ic, 4),
            std_ic=round(s_ic, 4),
            ir=round(ir, 2),
            mean_rank_ic=round(m_rank_ic, 4),
            std_rank_ic=round(s_rank_ic, 4),
            rank_ir=round(rank_ir, 2),
            t_stat=round(t_stat, 2),
            p_value_approx=round(p_val, 4),
            sample_size=n,
            is_significant=is_sig,
            ic_series=ic_series,
            rank_ic_series=rank_ic_series,
        )

    @staticmethod
    def evaluate_quantiles(
        factor_matrix: Dict[str, List[float]],
        returns_matrix: Dict[str, List[float]],
        num_quantiles: int = 5,
    ) -> QuantileReport:
        """
        Sort assets into N quantiles at each timestep and compute cumulative bucket returns.
        """
        symbols = [s for s in factor_matrix.keys() if s in returns_matrix]
        if not symbols or len(symbols) < num_quantiles:
            empty_q = {f"Q{q+1}": 0.0 for q in range(num_quantiles)}
            return QuantileReport(num_quantiles, empty_q, 0.0, 0.0, False)

        time_len = min(len(factor_matrix[s]) for s in symbols)
        # Quantile cumulative return tracking
        q_returns: Dict[int, List[float]] = {q: [] for q in range(num_quantiles)}

        for t in range(time_len):
            pairs = []
            for s in symbols:
                fv = factor_matrix[s][t]
                rv = returns_matrix[s][t]
                if not (math.isnan(fv) or math.isnan(rv)):
                    pairs.append((fv, rv))

            if len(pairs) < num_quantiles:
                continue

            # Sort ascending by factor value
            pairs.sort(key=lambda x: x[0])
            bucket_size = len(pairs) / num_quantiles

            for q in range(num_quantiles):
                start_idx = int(q * bucket_size)
                end_idx = int((q + 1) * bucket_size) if q < num_quantiles - 1 else len(pairs)
                bucket_pairs = pairs[start_idx:end_idx]
                if bucket_pairs:
                    avg_ret = sum(p[1] for p in bucket_pairs) / len(bucket_pairs)
                    q_returns[q].append(avg_ret)

        # Compute cumulative compounding return for each quantile
        q_results: Dict[str, float] = {}
        q_cum_values: List[float] = []
        for q in range(num_quantiles):
            rets = q_returns[q]
            cum = 1.0
            for r in rets:
                cum *= (1.0 + r)
            final_pct = (cum - 1.0) * 100.0
            q_results[f"Q{q+1}"] = round(final_pct, 2)
            q_cum_values.append(final_pct)

        # Long-Short spread (Q_highest - Q_lowest)
        spread = q_cum_values[-1] - q_cum_values[0]

        # Monotonicity score: Rank correlation between quantile indices [1, 2, 3, 4, 5] and cumulative returns
        q_indices = list(range(1, num_quantiles + 1))
        mono_score = _spearman_rank_corr(q_indices, q_cum_values)
        is_mono = abs(mono_score) >= 0.80

        return QuantileReport(
            num_quantiles=num_quantiles,
            quantile_returns=q_results,
            long_short_spread_pct=round(spread, 2),
            monotonicity_score=round(mono_score, 2),
            is_monotonic=is_mono,
        )

    @staticmethod
    def evaluate_incremental_alpha(
        candidate_factor: List[float],
        base_factors: List[List[float]],
        forward_returns: List[float],
        rolling_window: int = 30,
        candidate_name: str = "candidate_factor",
    ) -> IncrementalAlphaReport:
        """
        Evaluate whether a candidate factor provides genuine incremental alpha over existing base factors.
        Applies Modified Gram-Schmidt Orthogonalization and evaluates the residual factor.
        """
        raw_ic = FactorEvaluator.evaluate_time_series_ic(candidate_factor, forward_returns, rolling_window)

        if not base_factors:
            return IncrementalAlphaReport(
                candidate_name=candidate_name,
                raw_ic=raw_ic,
                residual_ic=raw_ic,
                r_squared_overlap=0.0,
                max_correlation_with_base=0.0,
                is_true_incremental_alpha=raw_ic.is_significant,
            )

        # 1. Compute max correlation with existing base factors
        max_corr = max(abs(_pearson_corr(candidate_factor, b)) for b in base_factors) if base_factors else 0.0

        # 2. Perform Gram-Schmidt orthogonalization
        residual_factor = gram_schmidt_orthogonalize(candidate_factor, base_factors)

        # 3. Compute R^2 overlap
        norm_orig = _standardize(candidate_factor)
        norm_res = _standardize(residual_factor)
        
        # Pearson correlation between original and residual: corr(cand, res)
        # R^2 with base factors is 1 - corr(cand, res)^2
        corr_orig_res = _pearson_corr(norm_orig, norm_res)
        r_sq = max(0.0, min(1.0, 1.0 - (corr_orig_res ** 2)))

        # 4. Evaluate residual IC
        residual_ic = FactorEvaluator.evaluate_time_series_ic(residual_factor, forward_returns, rolling_window)

        is_incremental = residual_ic.is_significant and r_sq < 0.85

        return IncrementalAlphaReport(
            candidate_name=candidate_name,
            raw_ic=raw_ic,
            residual_ic=residual_ic,
            r_squared_overlap=round(r_sq, 4),
            max_correlation_with_base=round(max_corr, 4),
            is_true_incremental_alpha=is_incremental,
        )

