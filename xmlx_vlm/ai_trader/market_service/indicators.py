"""纯函数技术指标计算.

与数据源、状态管理解耦，只接受序列并返回结果。
"""

from __future__ import annotations

from typing import List, Tuple

from .models import OHLCV


def _safe_mean(values: List[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def ema(values: List[float], period: int) -> List[float]:
    if period <= 0 or len(values) < period:
        return []
    alpha = 2.0 / (period + 1)
    out = [0.0] * len(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def rsi(values: List[float], period: int = 14) -> List[float]:
    if len(values) < period + 1:
        return [50.0] * len(values)
    gains, losses = [], []
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [50.0] * period
    for i in range(period, len(values)):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsis.append(100.0 - 100.0 / (1.0 + rs))
    return rsis


def atr(ohlcv: List[OHLCV], period: int = 14) -> List[float]:
    if len(ohlcv) < period + 1:
        return []
    trs = [0.0]
    for i in range(1, len(ohlcv)):
        c = ohlcv[i]
        p = ohlcv[i - 1]
        tr = max(
            c.high - c.low,
            abs(c.high - p.close),
            abs(c.low - p.close),
        )
        trs.append(tr)
    return ema(trs, period)


def adx(
    ohlcv: List[OHLCV], period: int = 14
) -> Tuple[List[float], List[float], List[float]]:
    """返回 (adx, +DI, -DI)."""
    n = len(ohlcv)
    plus_dm = [0.0]
    minus_dm = [0.0]
    trs = [0.0]
    for i in range(1, n):
        up = ohlcv[i].high - ohlcv[i - 1].high
        down = ohlcv[i - 1].low - ohlcv[i].low
        plus_dm.append(max(up, 0.0) if up > down else 0.0)
        minus_dm.append(max(down, 0.0) if down > up else 0.0)
        c = ohlcv[i]
        p = ohlcv[i - 1]
        trs.append(
            max(
                c.high - c.low,
                abs(c.high - p.close),
                abs(c.low - p.close),
            )
        )

    def _smooth(vals: List[float]) -> List[float]:
        if len(vals) < period:
            return [0.0] * len(vals)
        out = [0.0] * period
        s = sum(vals[1 : period + 1])
        out.append(s)
        for i in range(period + 1, len(vals)):
            s = s - s / period + vals[i]
            out.append(s)
        return out

    tr_smooth = _smooth(trs)
    plus_smooth = _smooth(plus_dm)
    minus_smooth = _smooth(minus_dm)

    plus_di = [0.0] * len(tr_smooth)
    minus_di = [0.0] * len(tr_smooth)
    dx = [0.0] * len(tr_smooth)
    for i in range(period, len(tr_smooth)):
        if tr_smooth[i] == 0:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
            dx[i] = 0.0
        else:
            plus_di[i] = 100.0 * plus_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_smooth[i] / tr_smooth[i]
            denom = plus_di[i] + minus_di[i]
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / denom if denom else 0.0

    adx_values = ema(dx[period:], period) if len(dx) > period else []
    # 补齐长度，使 adx 与输入对齐
    pad = len(ohlcv) - len(adx_values)
    adx_full = [0.0] * pad + list(adx_values)
    return adx_full, plus_di, minus_di


def volume_profile(
    ohlcv: List[OHLCV], bins: int = 24
) -> dict[str, float | None]:
    """基于收盘价分布计算 POC、VAH、VAL（70% 成交量价值区）."""
    if not ohlcv:
        return {"poc": None, "vah": None, "val": None}
    closes = [c.close for c in ohlcv]
    volumes = [c.volume for c in ohlcv]
    min_p, max_p = min(closes), max(closes)
    if min_p == max_p or bins <= 0:
        return {"poc": closes[-1], "vah": closes[-1], "val": closes[-1]}

    bin_edges = [min_p + (max_p - min_p) * i / bins for i in range(bins + 1)]
    bin_volumes = [0.0] * bins
    bin_prices = []
    for i in range(bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        bin_prices.append((lo + hi) / 2)
        for c, v in zip(closes, volumes):
            if (i < bins - 1 and lo <= c < hi) or (i == bins - 1 and lo <= c <= hi):
                bin_volumes[i] += v

    max_vol = max(bin_volumes)
    poc_idx = bin_volumes.index(max_vol)
    poc = bin_prices[poc_idx]

    total_vol = sum(bin_volumes)
    target = total_vol * 0.70
    cum = max_vol
    low_idx = high_idx = poc_idx
    while cum < target and (low_idx > 0 or high_idx < bins - 1):
        left_vol = bin_volumes[low_idx - 1] if low_idx > 0 else 0.0
        right_vol = bin_volumes[high_idx + 1] if high_idx < bins - 1 else 0.0
        if left_vol >= right_vol and low_idx > 0:
            low_idx -= 1
            cum += left_vol
        elif high_idx < bins - 1:
            high_idx += 1
            cum += right_vol
        else:
            break

    return {
        "poc": poc,
        "vah": bin_prices[high_idx],
        "val": bin_prices[low_idx],
        "coverage_pct": cum / total_vol * 100 if total_vol else 0.0,
    }


def vwap(trades: List[Tuple[float, float, int]]) -> float | None:
    """基于 (price, size, _) 序列计算 VWAP."""
    if not trades:
        return None
    pv = sum(p * s for p, s, _ in trades)
    v = sum(s for _, s, _ in trades)
    return pv / v if v else None


def cvd(trades: List[Tuple[str, float, float, int]]) -> float:
    """计算累积成交量差 (CVD).

    trades: List[(side, price, size, ts_ms)]
    """
    total = 0.0
    for side, price, size, _ in trades:
        if side.lower() == "buy":
            total += price * size
        else:
            total -= price * size
    return total
