"""AI Trader 默认配置.

本文件是 service.sh 的 Python 端镜像。所有默认值应与 service.sh 保持一致，
实现“改一处，到处生效”。环境变量优先于本文件默认值。
"""

import os
from pathlib import Path

# 项目内默认存储路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

for d in (DATA_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── 本地推理服务配置（与 service.sh 保持一致）──
DEFAULT_MODEL = os.getenv(
    "XMLX_VLM_MODEL", "mlx-community/diffusiongemma-26B-A4B-it-4bit"
)
DEFAULT_PORT = int(os.getenv("XMLX_VLM_PORT", "5118"))
DEFAULT_CHAT_PORT = int(os.getenv("XMLX_VLM_CHAT_PORT", "5119"))
DEFAULT_API_KEY = os.getenv("XMLX_VLM_API_KEY", "x123456")
DEFAULT_SERVER_URL = f"http://localhost:{DEFAULT_PORT}"

# 默认交易所（统一使用 Hyperliquid）
DEFAULT_EXCHANGE = "hyperliquid"

# 默认交易对
DEFAULT_SYMBOL = "BTC/USDC"

# 默认 K 线周期
DEFAULT_TIMEFRAME = "1h"

# 默认历史 K 线根数
DEFAULT_LIMIT = 100

# 风控默认值（仅用于演示，实盘请按自己资金调整）
DEFAULT_RISK = {
    "max_daily_loss_pct": 5.0,
    "max_position_pct": 50.0,
    "max_single_loss_pct": 2.0,
}
