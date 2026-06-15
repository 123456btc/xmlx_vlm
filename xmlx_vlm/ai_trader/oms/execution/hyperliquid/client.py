"""Hyperliquid REST 客户端."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests

from xmlx_vlm.ai_trader.oms.constants import (
    HL_MAINNET_EXCHANGE_URL,
    HL_MAINNET_INFO_URL,
    HL_TESTNET_EXCHANGE_URL,
    HL_TESTNET_INFO_URL,
)
from xmlx_vlm.ai_trader.oms.exceptions import AdapterError

logger = logging.getLogger(__name__)


class HyperliquidClient:
    """Hyperliquid 行情与交易 REST 客户端."""

    def __init__(
        self,
        wallet_address: Optional[str] = None,
        testnet: bool = False,
        timeout: int = 20,
    ):
        self.wallet_address = wallet_address
        self.testnet = testnet
        self.timeout = timeout
        self.info_url = HL_TESTNET_INFO_URL if testnet else HL_MAINNET_INFO_URL
        self.exchange_url = (
            HL_TESTNET_EXCHANGE_URL if testnet else HL_MAINNET_EXCHANGE_URL
        )
        self._session = requests.Session()

    def info(self, payload: Dict[str, Any]) -> Any:
        """调用 info 端点（公开数据）."""
        try:
            resp = self._session.post(
                self.info_url, json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise AdapterError(f"HL info request failed: {exc}") from exc

    def exchange(self, payload: Dict[str, Any]) -> Any:
        """调用 exchange 端点（需签名）."""
        try:
            resp = self._session.post(
                self.exchange_url, json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            raise AdapterError(f"HL exchange request failed: {exc}") from exc

    def get_user_abstraction(self, wallet_address: str) -> str:
        """查询账户抽象模式.

        返回 unifiedAccount / portfolioMargin / disabled / default / dexAbstraction。
        """
        return self.info({"type": "userAbstraction", "user": wallet_address})

    def get_spot_clearinghouse_state(self, wallet_address: str) -> Any:
        """查询 spot 清算状态（unified / portfolio margin 的余额源）."""
        return self.info({"type": "spotClearinghouseState", "user": wallet_address})

    def get_meta_and_asset_ctxs(self) -> Any:
        return self.info({"type": "metaAndAssetCtxs"})

    def get_candles(
        self,
        coin: str,
        interval: str,
        start_time_ms: int,
        end_time_ms: int,
    ) -> Any:
        return self.info(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": coin,
                    "interval": interval,
                    "startTime": start_time_ms,
                    "endTime": end_time_ms,
                },
            }
        )

    def close(self):
        self._session.close()
