"""OMS 配置."""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from xmlx_vlm.ai_trader.config import LOGS_DIR
from xmlx_vlm.ai_trader.oms.config.profiles import RISK_PROFILES
from xmlx_vlm.ai_trader.oms.exceptions import ConfigurationError
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal


class OMSSettings(BaseModel):
    """OMS 配置模型.

    配置优先级（高到低）：
    1. 构造参数
    2. 环境变量
    3. 默认值
    """

    # 模式
    # local = 本地仿真机构盘（与 hyperliquid 实盘地位相同，只是不连接交易所）
    live_enabled: bool = Field(default=False)
    exchange: Literal["local", "paper", "local_sim", "hyperliquid"] = Field(default="local")
    risk_profile: Literal["conservative", "moderate", "aggressive", "custom"] = Field(
        default="conservative"
    )

    # 路径
    audit_log_dir: Path = Field(default=LOGS_DIR / "oms_audit")
    audit_db_path: Path = Field(default=LOGS_DIR / "oms_audit.db")
    state_path: Path = Field(default=LOGS_DIR / "oms_state.json")

    # Hyperliquid 凭证（仅环境变量）
    wallet_address: Optional[str] = Field(default=None)
    private_key: Optional[str] = Field(default=None)
    signer_endpoint: Optional[str] = Field(default=None)
    testnet: bool = Field(default=False)
    request_timeout: int = Field(default=20)

    # 风控阈值（自定义时生效）
    max_daily_loss_pct: Decimal = Field(default=Decimal("3.0"))
    max_single_position_pct: Decimal = Field(default=Decimal("20.0"))
    max_total_position_pct: Decimal = Field(default=Decimal("50.0"))
    max_single_order_notional: Decimal = Field(default=Decimal("5000"))
    min_order_notional: Decimal = Field(default=Decimal("10"))
    max_price_deviation_pct: Decimal = Field(default=Decimal("1.0"))
    max_orders_per_minute: int = Field(default=12)
    max_orders_per_second: int = Field(default=3)
    min_available_margin_pct: Decimal = Field(default=Decimal("20.0"))
    max_slippage_pct: Decimal = Field(default=Decimal("0.5"))

    # 熔断
    daily_loss_circuit_enabled: bool = Field(default=True)
    api_error_circuit_enabled: bool = Field(default=True)
    consecutive_loss_circuit_enabled: bool = Field(default=True)
    max_api_errors: int = Field(default=5)
    max_consecutive_losses: int = Field(default=3)

    # 本地仿真机构盘（paper / local_sim）
    paper_fill_slippage_pct: Decimal = Field(default=Decimal("0.0"))
    paper_initial_equity: Decimal = Field(default=Decimal("100000"))
    paper_default_price: Decimal = Field(default=Decimal("50000"))
    paper_order_book_depth: Decimal = Field(default=Decimal("50000"))
    paper_latency_ms: int = Field(default=0)

    # 执行算法
    algo_enabled: bool = Field(default=True)
    default_algo: str = Field(default="twap")

    # 冲击模型
    impact_model_enabled: bool = Field(default=True)
    impact_adv_window_days: int = Field(default=20)
    impact_volatility_window: int = Field(default=30)

    # 智能路由
    smart_router_enabled: bool = Field(default=True)

    # 杂项
    dry_run: bool = Field(default=False)
    auto_flatten_on_kill: bool = Field(default=True)

    @field_validator("exchange", mode="before")
    @classmethod
    def _coerce_exchange(cls, v):
        if isinstance(v, str):
            val = v.lower()
            if val == "paper" or val == "local_sim":
                return "local"
            return val
        return v

    @field_validator(
        "max_daily_loss_pct",
        "max_single_position_pct",
        "max_total_position_pct",
        "max_single_order_notional",
        "min_order_notional",
        "max_price_deviation_pct",
        "min_available_margin_pct",
        "max_slippage_pct",
        "paper_fill_slippage_pct",
        "paper_initial_equity",
        "paper_default_price",
        "paper_order_book_depth",
        mode="before",
    )
    @classmethod
    def _coerce_decimal(cls, v):
        return to_decimal(v)

    def model_post_init(self, __context: Any) -> None:
        # 应用风控模板
        if self.risk_profile != "custom":
            profile = RISK_PROFILES.get(self.risk_profile, RISK_PROFILES["conservative"])
            for key, value in profile.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def risk_profile_dict(self) -> Dict[str, Any]:
        """返回风控规则可用的配置字典."""
        return {
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_single_position_pct": self.max_single_position_pct,
            "max_total_position_pct": self.max_total_position_pct,
            "max_single_order_notional": self.max_single_order_notional,
            "min_order_notional": self.min_order_notional,
            "max_price_deviation_pct": self.max_price_deviation_pct,
            "max_orders_per_minute": self.max_orders_per_minute,
            "max_orders_per_second": self.max_orders_per_second,
            "min_available_margin_pct": self.min_available_margin_pct,
            "max_slippage_pct": self.max_slippage_pct,
        }

    def validate_live(self) -> None:
        """校验实盘配置是否完整."""
        if not self.live_enabled:
            return
        if self.exchange != "hyperliquid":
            raise ConfigurationError("live trading only supported with hyperliquid exchange")
        if not self.wallet_address:
            raise ConfigurationError("wallet_address is required for live trading")
        if not self.private_key and not self.signer_endpoint:
            raise ConfigurationError(
                "private_key or signer_endpoint is required for live trading"
            )


_SETTINGS: Optional[OMSSettings] = None


def get_settings(
    live: Optional[bool] = None,
    exchange: Optional[str] = None,
    risk_profile: Optional[str] = None,
) -> OMSSettings:
    """从环境变量构建 OMS 配置."""
    global _SETTINGS
    if _SETTINGS is not None and live is None and exchange is None and risk_profile is None:
        return _SETTINGS

    def _env_bool(key: str, default: bool = False) -> bool:
        val = os.getenv(key)
        if val is None:
            return default
        return val.lower() in ("1", "true", "yes", "on")

    def _env_decimal(key: str, default: Decimal) -> Decimal:
        val = os.getenv(key)
        if val is None:
            return default
        return to_decimal(val)

    kwargs: Dict[str, Any] = {
        "live_enabled": _env_bool("AI_TRADER_LIVE", False),
        "exchange": os.getenv("AI_TRADER_EXCHANGE", "paper"),
        "risk_profile": os.getenv("AI_TRADER_RISK_PROFILE", "conservative"),
        "wallet_address": os.getenv("HL_API_WALLET_ADDRESS"),
        "private_key": os.getenv("HL_API_PRIVATE_KEY"),
        "signer_endpoint": os.getenv("HL_SIGNER_ENDPOINT"),
        "testnet": _env_bool("HL_TESTNET", False),
        "dry_run": _env_bool("AI_TRADER_DRY_RUN", False),
    }

    # 风控阈值环境变量
    risk_env_keys = [
        "max_daily_loss_pct",
        "max_single_position_pct",
        "max_total_position_pct",
        "max_single_order_notional",
        "min_order_notional",
        "max_price_deviation_pct",
        "min_available_margin_pct",
        "max_slippage_pct",
        "paper_fill_slippage_pct",
        "paper_initial_equity",
        "paper_order_book_depth",
    ]
    for key in risk_env_keys:
        env_key = f"AI_TRADER_{key.upper()}"
        if os.getenv(env_key):
            kwargs[key] = _env_decimal(env_key, Decimal("0"))

    int_env_keys = [
        "max_orders_per_minute",
        "max_orders_per_second",
        "paper_latency_ms",
        "impact_adv_window_days",
        "impact_volatility_window",
    ]
    for key in int_env_keys:
        env_key = f"AI_TRADER_{key.upper()}"
        if os.getenv(env_key):
            kwargs[key] = int(os.getenv(env_key))

    bool_env_keys = [
        "algo_enabled",
        "impact_model_enabled",
        "smart_router_enabled",
    ]
    for key in bool_env_keys:
        env_key = f"AI_TRADER_{key.upper()}"
        if os.getenv(env_key):
            kwargs[key] = _env_bool(env_key, False)

    if os.getenv("AI_TRADER_DEFAULT_ALGO"):
        kwargs["default_algo"] = os.getenv("AI_TRADER_DEFAULT_ALGO")

    # 显式参数覆盖环境变量
    if live is not None:
        kwargs["live_enabled"] = live
    if exchange is not None:
        kwargs["exchange"] = exchange
    if risk_profile is not None:
        kwargs["risk_profile"] = risk_profile

    settings = OMSSettings(**kwargs)
    if settings.live_enabled:
        settings.validate_live()

    if live is None and exchange is None and risk_profile is None:
        _SETTINGS = settings
    return settings


def reset_settings() -> None:
    global _SETTINGS
    _SETTINGS = None
