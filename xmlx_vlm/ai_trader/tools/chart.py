"""图表工具 —— 使用 Pillow 绘制带技术指标的 K 线图，无需 matplotlib."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont

from xmlx_vlm.ai_trader.config import DATA_DIR
from xmlx_vlm.ai_trader.tools.market import MarketDataTool, OHLCV

logger = logging.getLogger(__name__)


try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    np = None  # type: ignore
    logger.debug("numpy not installed: %s", exc)


@dataclass
class ChartResult:
    image: Image.Image
    image_path: str
    ohlcv_count: int


class ChartTool:
    """K 线图生成工具（Pillow 实现，兼容 Python 3.14）."""

    name = "render_chart"
    description = "根据历史 K 线数据生成带技术指标（EMA、成交量、RSI）的图表，用于视觉分析."
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "交易对，例如 BTC/USDC、ETH/USDC",
            },
            "exchange": {
                "type": "string",
                "description": "已固定为 hyperliquid，可省略",
                "default": "hyperliquid",
            },
            "timeframe": {
                "type": "string",
                "description": "K 线周期，例如 1h、4h、1d",
                "default": "1h",
            },
            "limit": {
                "type": "integer",
                "description": "K 线数量",
                "default": 100,
            },
            "indicators": {
                "type": "array",
                "items": {"type": "string"},
                "description": "要绘制的指标，例如 [\"ema20\", \"ema50\", \"rsi\"]",
                "default": ["ema20", "ema50", "rsi"],
            },
        },
        "required": ["symbol"],
    }

    def __init__(self):
        self.market = MarketDataTool()

    @staticmethod
    def _ema(values: List[float], period: int) -> List[float]:
        if np is None:
            raise RuntimeError("numpy 未安装")
        arr = np.array(values, dtype=float)
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(arr)
        ema[0] = arr[0]
        for i in range(1, len(arr)):
            ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
        return ema.tolist()

    @staticmethod
    def _rsi(values: List[float], period: int = 14) -> List[float]:
        if np is None:
            raise RuntimeError("numpy 未安装")
        arr = np.array(values, dtype=float)
        deltas = np.diff(arr)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        rsis = [float("nan")] * (period + 1)

        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rsis.append(100.0 - (100.0 / (1 + avg_gain / avg_loss)))

        for i in range(period + 1, len(arr)):
            gain = gains[i - 1]
            loss = losses[i - 1]
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            if avg_loss == 0:
                rsis.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsis.append(100.0 - (100.0 / (1 + rs)))
        return rsis

    def render(
        self,
        symbol: str,
        exchange: str = "hyperliquid",
        timeframe: str = "1h",
        limit: int = 100,
        indicators: Optional[List[str]] = None,
    ) -> ChartResult:
        """生成 K 线图并返回 PIL Image."""
        indicators = indicators or ["ema20", "ema50", "rsi"]
        indicators = [i.lower() for i in indicators]

        ohlcv = self.market.get_ohlcv(symbol, exchange, timeframe, limit)
        if not ohlcv:
            raise ValueError("未获取到 K 线数据")

        closes = [c.close for c in ohlcv]
        volumes = [c.volume for c in ohlcv]
        dates = [datetime.fromtimestamp(c.timestamp / 1000) for c in ohlcv]

        ema20 = self._ema(closes, 20) if "ema20" in indicators else None
        ema50 = self._ema(closes, 50) if "ema50" in indicators else None
        rsi_values = self._rsi(closes, 14) if "rsi" in indicators else None

        show_rsi = "rsi" in indicators and rsi_values
        width, height = 1200, 800 if show_rsi else 650
        margin_left, margin_right = 70, 40
        margin_top, margin_bottom = 60, 80

        img = Image.new("RGB", (width, height), "#0d1117")
        draw = ImageDraw.Draw(img)

        # 尝试加载字体，失败则用默认
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except Exception:
            font = ImageFont.load_default()
            title_font = font

        # 标题
        title = f"{symbol} {timeframe} ({len(ohlcv)} bars)"
        draw.text((margin_left, 20), title, fill="white", font=title_font)

        # 布局
        if show_rsi:
            price_height = int((height - margin_top - margin_bottom) * 0.55)
            volume_height = int((height - margin_top - margin_bottom) * 0.20)
            rsi_height = int((height - margin_top - margin_bottom) * 0.25)
        else:
            price_height = int((height - margin_top - margin_bottom) * 0.70)
            volume_height = int((height - margin_top - margin_bottom) * 0.30)
            rsi_height = 0

        price_top = margin_top
        price_bottom = price_top + price_height
        volume_top = price_bottom + 10
        volume_bottom = volume_top + volume_height
        rsi_top = volume_bottom + 10
        rsi_bottom = rsi_top + rsi_height

        chart_left = margin_left
        chart_right = width - margin_right
        chart_width = chart_right - chart_left

        n = len(ohlcv)
        candle_width = max(2, chart_width // (n * 2))
        gap = max(1, candle_width // 2)
        step = (chart_width - candle_width) / max(1, n - 1)

        def x_of(i: int) -> float:
            return chart_left + i * step + candle_width / 2

        # 价格范围
        min_price = min(c.low for c in ohlcv)
        max_price = max(c.high for c in ohlcv)
        price_pad = (max_price - min_price) * 0.05
        min_price -= price_pad
        max_price += price_pad

        def y_price(p: float) -> float:
            return price_bottom - (p - min_price) / (max_price - min_price) * price_height

        def y_volume(v: float) -> float:
            max_vol = max(volumes) * 1.1
            return volume_bottom - v / max_vol * volume_height

        # 画价格网格和标签
        for i in range(6):
            price = min_price + (max_price - min_price) * i / 5
            y = y_price(price)
            draw.line([(chart_left, y), (chart_right, y)], fill="#21262d", width=1)
            draw.text((10, int(y) - 6), f"{price:,.2f}", fill="#8b949e", font=font)

        # 画蜡烛
        for i, c in enumerate(ohlcv):
            x = x_of(i)
            color = "#26a69a" if c.close >= c.open else "#ef5350"
            y_high = y_price(c.high)
            y_low = y_price(c.low)
            y_open = y_price(c.open)
            y_close = y_price(c.close)
            # 影线
            draw.line([(x, y_high), (x, y_low)], fill=color, width=1)
            # 实体
            top = min(y_open, y_close)
            bottom = max(y_open, y_close)
            body_height = max(1, bottom - top)
            draw.rectangle(
                [(x - candle_width / 2, top), (x + candle_width / 2, bottom)],
                fill=color,
                outline=color,
            )

        # 画 EMA
        def draw_line(values, color):
            points = [(x_of(i), y_price(values[i])) for i in range(len(values))]
            draw.line(points, fill=color, width=2)

        if ema20:
            draw_line(ema20, "#ffa500")
        if ema50:
            draw_line(ema50, "#58a6ff")

        # 图例
        legend_items = []
        if ema20:
            legend_items.append(("EMA20", "#ffa500"))
        if ema50:
            legend_items.append(("EMA50", "#58a6ff"))
        lx = chart_right - 150
        ly = price_top + 10
        for label, color in legend_items:
            draw.line([(lx, ly + 6), (lx + 20, ly + 6)], fill=color, width=2)
            draw.text((lx + 25, ly), label, fill="white", font=font)
            ly += 18

        # 成交量
        for i, v in enumerate(volumes):
            x = x_of(i)
            c = ohlcv[i]
            color = "#26a69a" if c.close >= c.open else "#ef5350"
            y = y_volume(v)
            draw.rectangle(
                [(x - candle_width / 2, y), (x + candle_width / 2, volume_bottom)],
                fill=color,
            )
        draw.text((10, volume_top + 5), "Volume", fill="#8b949e", font=font)

        # RSI
        if show_rsi:
            valid_idx = [i for i in range(len(rsi_values)) if not (rsi_values[i] != rsi_values[i])]  # not NaN
            if valid_idx:
                min_rsi, max_rsi = 0, 100

                def y_rsi(v: float) -> float:
                    return rsi_bottom - (v - min_rsi) / (max_rsi - min_rsi) * rsi_height

                points = [(x_of(i), y_rsi(rsi_values[i])) for i in valid_idx]
                draw.line(points, fill="#a371f7", width=2)
                draw.line([(chart_left, y_rsi(70)), (chart_right, y_rsi(70))], fill="#ef5350", width=1)
                draw.line([(chart_left, y_rsi(30)), (chart_right, y_rsi(30))], fill="#26a69a", width=1)
                draw.text((10, rsi_top + 5), "RSI(14)", fill="#8b949e", font=font)
                draw.text((chart_right - 30, y_rsi(70) - 12), "70", fill="#ef5350", font=font)
                draw.text((chart_right - 30, y_rsi(30) + 2), "30", fill="#26a69a", font=font)

        # 时间轴标签
        step_label = max(1, n // 6)
        for i in range(0, n, step_label):
            x = x_of(i)
            label = dates[i].strftime("%m-%d %H:%M")
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            draw.text((int(x - text_w / 2), height - margin_bottom + 10), label, fill="#8b949e", font=font)

        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol.replace('/', '_')}_{timeframe}_{timestamp}.png"
        path = str(DATA_DIR / filename)
        img.save(path)

        return ChartResult(image=img, image_path=path, ohlcv_count=len(ohlcv))

    def run(self, **kwargs) -> str:
        """工具统一入口."""
        symbol = kwargs.get("symbol")
        if not symbol:
            return "错误：必须提供 symbol 参数"
        exchange = kwargs.get("exchange", "hyperliquid")
        timeframe = kwargs.get("timeframe", "1h")
        limit = int(kwargs.get("limit", 100))
        indicators = kwargs.get("indicators")

        try:
            result = self.render(symbol, exchange, timeframe, limit, indicators)
            return f"已生成 {result.ohlcv_count} 根 K 线的图表，保存于 {result.image_path}"
        except Exception as exc:
            logger.exception("render_chart tool failed")
            return f"图表生成失败: {exc}"
