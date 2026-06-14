"""AI Trader 默认配置."""

from pathlib import Path

# 项目内默认存储路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

for d in (DATA_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

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
