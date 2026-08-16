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


# ── 数学与统计辅助函数 ──

def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return var ** 0.5


def _linear_slope(values: List[float]) -> float:
    """计算一维序列的线性回归斜率."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = _mean(values)
    numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    return numerator / denominator if denominator != 0 else 0.0


def _correlation(x: List[float], y: List[float]) -> float:
    """计算两个序列的皮尔逊相关系数."""
    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x_slice = x[-n:]
    y_slice = y[-n:]
    mx, my = _mean(x_slice), _mean(y_slice)
    num = sum((a - mx) * (b - my) for a, b in zip(x_slice, y_slice))
    denom = (sum((a - mx) ** 2 for a in x_slice) * sum((b - my) ** 2 for b in y_slice)) ** 0.5
    return num / denom if denom != 0 else 0.0


# ── 6 大币圈实战量化策略因子 ──

def pinbar_liquidity_sweep(
    ohlcv: List[OHLCV],
    min_wick_ratio: float = 0.60,
    volume_multiplier: float = 1.8,
    lookback_ma: int = 20,
) -> dict:
    """因子 1: 爆仓清算针与插针吸收因子 (Pin-Bar / Liquidity Sweep).

    识别长影线与放量结合的流动性掠夺与爆仓踩踏反转信号。
    """
    if not ohlcv:
        return {
            "sweep_type": "none",
            "is_sweep": False,
            "wick_ratio": 0.0,
            "volume_ratio": 1.0,
            "lower_wick_ratio": 0.0,
            "upper_wick_ratio": 0.0,
        }

    curr = ohlcv[-1]
    hl_range = curr.high - curr.low
    if hl_range <= 1e-8:
        return {
            "sweep_type": "none",
            "is_sweep": False,
            "wick_ratio": 0.0,
            "volume_ratio": 1.0,
            "lower_wick_ratio": 0.0,
            "upper_wick_ratio": 0.0,
        }

    lower_wick = (min(curr.open, curr.close) - curr.low) / hl_range
    upper_wick = (curr.high - max(curr.open, curr.close)) / hl_range

    # 历史成交量均值
    hist = ohlcv[-lookback_ma - 1 : -1] if len(ohlcv) > 1 else []
    avg_vol = _mean([c.volume for c in hist]) if hist else curr.volume
    vol_ratio = (curr.volume / avg_vol) if avg_vol > 0 else 1.0

    sweep_type = "none"
    is_sweep = False
    wick_ratio = 0.0

    if lower_wick >= min_wick_ratio and vol_ratio >= volume_multiplier:
        sweep_type = "bullish_sweep"  # 清算多头止损后被动吸筹收出长下影线
        is_sweep = True
        wick_ratio = lower_wick
    elif upper_wick >= min_wick_ratio and vol_ratio >= volume_multiplier:
        sweep_type = "bearish_sweep"  # 清算空头止损后被动出货收出长上影线
        is_sweep = True
        wick_ratio = upper_wick

    return {
        "sweep_type": sweep_type,
        "is_sweep": is_sweep,
        "wick_ratio": round(wick_ratio, 4),
        "volume_ratio": round(vol_ratio, 2),
        "lower_wick_ratio": round(lower_wick, 4),
        "upper_wick_ratio": round(upper_wick, 4),
    }


def cvd_price_divergence(
    prices: List[float],
    cvds: List[float],
    lookback: int = 15,
) -> dict:
    """因子 2: 价格与 CVD 净买卖背离因子 (Price-CVD Divergence / Passive Absorption).

    量化散户市价追单与巨鲸限价被动挂单的吸收背离。
    """
    if len(prices) < 4 or len(cvds) < 4:
        return {
            "divergence_type": "neutral",
            "correlation": 0.0,
            "price_slope": 0.0,
            "cvd_slope": 0.0,
            "is_divergence": False,
        }

    p_window = prices[-lookback:]
    c_window = cvds[-lookback:]

    p_slope = _linear_slope(p_window)
    c_slope = _linear_slope(c_window)
    corr = _correlation(p_window, c_window)

    p_norm_slope = p_slope / (p_window[-1] if p_window[-1] else 1.0) * 100.0
    c_range = max(c_window) - min(c_window)
    c_norm_slope = c_slope / (c_range if c_range > 0 else 1.0) * 100.0

    divergence_type = "neutral"
    is_divergence = False

    # 顶背离：价格斜率为正，但 CVD 斜率为负（或相关性显著为负）
    if p_norm_slope > 0.05 and (c_norm_slope < -0.05 or corr < -0.35):
        divergence_type = "bearish_divergence"
        is_divergence = True
    # 底背离：价格斜率为负，但 CVD 斜率为正（或相关性显著为负）
    elif p_norm_slope < -0.05 and (c_norm_slope > 0.05 or corr < -0.35):
        divergence_type = "bullish_divergence"
        is_divergence = True
    elif corr > 0.4 and p_norm_slope * c_norm_slope > 0:
        divergence_type = "confirmed_trend"

    return {
        "divergence_type": divergence_type,
        "correlation": round(corr, 3),
        "price_slope": round(p_slope, 4),
        "cvd_slope": round(c_slope, 4),
        "is_divergence": is_divergence,
    }


def oi_price_regime(
    prices: List[float],
    ois: List[float],
    lookback: int = 15,
) -> dict:
    """因子 3: 持仓量 (OI) - 价格共振 4 象限因子 (OI-Price Regime Matrix).

    区分增仓真突破、轧空衰竭（Short Squeeze）与踩踏爆仓底（Long Squeeze）。
    """
    if len(prices) < 2 or len(ois) < 2:
        return {
            "regime": "neutral",
            "regime_desc": "数据不足",
            "price_change_pct": 0.0,
            "oi_change_pct": 0.0,
        }

    p_start, p_end = prices[-lookback], prices[-1] if len(prices) >= lookback else (prices[0], prices[-1])
    oi_start, oi_end = ois[-lookback], ois[-1] if len(ois) >= lookback else (ois[0], ois[-1])

    p_pct = (p_end - p_start) / p_start * 100.0 if p_start > 0 else 0.0
    oi_pct = (oi_end - oi_start) / oi_start * 100.0 if oi_start > 0 else 0.0

    threshold_p = 0.3  # 0.3% 价格阈值
    threshold_oi = 0.8 # 0.8% OI 阈值

    if p_pct > threshold_p and oi_pct > threshold_oi:
        regime = "long_buildup"
        desc = "增仓拉升 (多头主动建仓/真趋势)"
    elif p_pct > threshold_p and oi_pct < -threshold_oi:
        regime = "short_squeeze"
        desc = "轧空反弹 (空头爆仓平仓推动/谨防力竭)"
    elif p_pct < -threshold_p and oi_pct > threshold_oi:
        regime = "short_buildup"
        desc = "增仓下砸 (空头主动建仓/单边空头)"
    elif p_pct < -threshold_p and oi_pct < -threshold_oi:
        regime = "long_liquidation"
        desc = "多头踩踏爆仓 (连环止损/接近流动性底)"
    else:
        regime = "neutral"
        desc = "中性平衡 (无明显主力仓位偏好)"

    return {
        "regime": regime,
        "regime_desc": desc,
        "price_change_pct": round(p_pct, 2),
        "oi_change_pct": round(oi_pct, 2),
    }


def bollinger_bands(
    values: List[float],
    period: int = 20,
    std_dev_mult: float = 2.0,
) -> dict:
    """因子 4: 布林带基础指标 (Bollinger Bands)."""
    if len(values) < period:
        latest = values[-1] if values else 0.0
        return {
            "middle": latest,
            "upper": latest,
            "lower": latest,
            "bandwidth": 0.0,
            "percent_b": 0.5,
        }

    window = values[-period:]
    mid = _mean(window)
    sd = _std(window)
    upper = mid + std_dev_mult * sd
    lower = mid - std_dev_mult * sd
    bw = (upper - lower) / mid if mid > 0 else 0.0
    latest = values[-1]
    denom = upper - lower
    percent_b = (latest - lower) / denom if denom > 1e-8 else 0.5

    return {
        "middle": round(mid, 4),
        "upper": round(upper, 4),
        "lower": round(lower, 4),
        "bandwidth": round(bw, 6),
        "percent_b": round(percent_b, 4),
    }


def bollinger_squeeze(
    bandwidth_history: List[float],
    lookback: int = 100,
) -> dict:
    """因子 4 (进阶): 布林带挤压突变因子 (Bollinger Squeeze Factor).

    计算 BandWidth 处于历史 lookback 周期的分位数。分数 < 0.15 代表极度紧缩，即将大爆发。
    """
    if not bandwidth_history:
        return {"squeeze_score": 0.5, "is_squeezed": False, "current_bandwidth": 0.0}

    current_bw = bandwidth_history[-1]
    window = bandwidth_history[-lookback:]
    if len(window) < 10:
        return {"squeeze_score": 0.5, "is_squeezed": False, "current_bandwidth": round(current_bw, 6)}

    # 分位数计算
    smaller_count = sum(1 for x in window if x <= current_bw)
    squeeze_score = smaller_count / len(window)
    is_squeezed = squeeze_score < 0.15

    return {
        "squeeze_score": round(squeeze_score, 4),
        "is_squeezed": is_squeezed,
        "current_bandwidth": round(current_bw, 6),
    }


def funding_rate_zscore(
    funding_history: List[float],
    lookback: int = 72,
) -> dict:
    """因子 5: 资金费率 Z-Score 与持仓拥挤度因子 (Funding Rate Crowding).

    识别多头/空头极端拥挤与费率磨损踩踏风险。
    """
    if not funding_history:
        return {
            "current_rate": 0.0,
            "zscore": 0.0,
            "mean_rate": 0.0,
            "crowding_status": "normal",
            "is_crowded": False,
        }

    curr = funding_history[-1]
    window = funding_history[-lookback:]
    mean_f = _mean(window)
    std_f = _std(window)

    z = (curr - mean_f) / std_f if std_f > 1e-8 else 0.0

    crowding = "normal"
    is_crowded = False
    if z >= 2.5 or curr >= 0.0005:  # 8h 费率 0.05% 以上
        crowding = "long_overcrowded"
        is_crowded = True
    elif z <= -2.5 or curr <= -0.0005:
        crowding = "short_overcrowded"
        is_crowded = True

    return {
        "current_rate": round(curr, 6),
        "zscore": round(z, 2),
        "mean_rate": round(mean_f, 6),
        "crowding_status": crowding,
        "is_crowded": is_crowded,
    }


def candle_efficiency(ohlcv: List[OHLCV]) -> dict:
    """因子 6: K 线实体推进效率比 (Candle Body Efficiency Ratio).

    实体波幅占全波幅比例。> 0.75 为真突破，< 0.35 为长影线假突破/诱多诱空。
    """
    if not ohlcv:
        return {"efficiency": 0.5, "is_high_efficiency": False, "is_fakeout_risk": False}

    curr = ohlcv[-1]
    hl_range = curr.high - curr.low
    if hl_range <= 1e-8:
        return {"efficiency": 1.0, "is_high_efficiency": True, "is_fakeout_risk": False}

    body = abs(curr.close - curr.open)
    eff = body / hl_range

    return {
        "efficiency": round(eff, 4),
        "is_high_efficiency": eff >= 0.75,
        "is_fakeout_risk": eff < 0.35,
    }

