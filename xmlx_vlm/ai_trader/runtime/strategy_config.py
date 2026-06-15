"""策略配置模型."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from xmlx_vlm.ai_trader.config import DEFAULT_API_KEY, DEFAULT_MODEL, DEFAULT_SERVER_URL


class AgentStrategyConfig(BaseModel):
    """Agent 自主交易专用配置."""

    mode: Literal["observe", "advise", "semi_auto", "full_auto"] = "observe"
    daily_volatility_target_pct: float = Field(default=2.0, gt=0)
    max_drawdown_pct: float = Field(default=5.0, gt=0)
    sharpe_target: float = Field(default=1.0)
    max_open_positions: int = Field(default=5, ge=1)
    preferred_timeframe: str = "1h"
    max_risk_pct_per_trade: float = Field(default=1.0, gt=0)
    max_risk_usd_per_trade: Optional[float] = None
    max_position_size_usd: float = Field(default=10000.0, gt=0)
    max_leverage: int = Field(default=5, ge=1, le=50)
    min_confidence: int = Field(default=60, ge=0, le=100)
    min_risk_reward_ratio: float = Field(default=1.5, gt=0)


class GridStrategyConfig(BaseModel):
    """网格策略专用配置."""

    symbol: str
    upper_price: Optional[float] = None
    lower_price: Optional[float] = None
    grid_count: int = 5
    total_investment: float = Field(default=1000.0, gt=0)
    max_drawdown_pct: float = Field(default=5.0, gt=0)
    daily_loss_limit_pct: float = Field(default=2.0, gt=0)


class StrategyConfig(BaseModel):
    """单个策略实例配置."""

    id: str = Field(..., min_length=1)
    name: str = ""
    exchange: Literal["paper", "hyperliquid"] = "paper"
    strategy_type: Literal["trend", "grid", "agent"] = "trend"
    symbols: List[str] = Field(default_factory=lambda: ["BTC/USDC", "ETH/USDC"])
    scan_interval_seconds: int = Field(default=300, ge=10)
    prompt_variant: Literal["default", "conservative", "aggressive", "grid"] = "default"
    max_positions: int = Field(default=3, ge=0)
    min_confidence: int = Field(default=60, ge=0, le=100)
    default_leverage: int = Field(default=3, ge=1, le=50)
    enabled: bool = True
    live_enabled: bool = False
    dry_run: bool = True
    risk_profile: Literal["conservative", "moderate", "aggressive", "custom"] = "conservative"
    # 本地推理配置（与 service.sh 共享环境变量，默认值来自 config.py）
    server_url: str = DEFAULT_SERVER_URL
    api_key: Optional[str] = DEFAULT_API_KEY
    model_path: Optional[str] = DEFAULT_MODEL
    temperature: float = 0.3
    max_tokens: int = 2048
    allow_mlx_fallback: bool = True
    order_sync_enabled: bool = False
    order_sync_interval_seconds: int = 5
    grid: Optional[GridStrategyConfig] = None
    agent: Optional[AgentStrategyConfig] = None

    wallet_address: Optional[str] = None
    private_key: Optional[str] = None
    testnet: bool = False

    def to_oms_settings_kwargs(self) -> Dict[str, Any]:
        """生成 OMSSettings 构造参数."""
        return {
            "exchange": self.exchange,
            "live_enabled": self.live_enabled,
            "risk_profile": self.risk_profile,
            "dry_run": self.dry_run,
            "wallet_address": self.wallet_address,
            "private_key": self.private_key,
            "testnet": self.testnet,
        }
