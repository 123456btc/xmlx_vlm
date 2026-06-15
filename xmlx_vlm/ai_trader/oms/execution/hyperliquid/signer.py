"""Hyperliquid 签名模块.

支持两种方式：
1. 本地私钥签名（仅用于开发/测试，私钥通过环境变量传入，不写入代码）
2. 外部签名器（生产推荐）：将待签名 payload POST 到 HL_SIGNER_ENDPOINT，私钥不落地
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import requests

from xmlx_vlm.ai_trader.oms.exceptions import AdapterError, ConfigurationError

logger = logging.getLogger(__name__)


class Signer(ABC):
    """签名器抽象基类."""

    @abstractmethod
    def sign(self, action: Dict[str, Any], timestamp_ms: int) -> Dict[str, Any]:
        """返回包含 signature 的完整请求 payload."""
        ...


class LocalPrivateKeySigner(Signer):
    """本地私钥签名器.

    依赖 eth-account。未安装时抛出 ConfigurationError。
    """

    def __init__(self, wallet_address: str, private_key: str):
        self.wallet_address = wallet_address
        self.private_key = private_key
        self._eth_account = self._import_eth_account()

    def _import_eth_account(self):
        try:
            from eth_account import Account

            return Account
        except ImportError as exc:
            raise ConfigurationError(
                "eth-account is required for Hyperliquid local signing. "
                "Install with: pip install eth-account"
            ) from exc

    def sign(self, action: Dict[str, Any], timestamp_ms: int) -> Dict[str, Any]:
        try:
            from eth_account import Account
            from hyperliquid.utils.signing import sign_l1_action

            wallet = Account.from_key(self.private_key)
            is_mainnet = not os.getenv("HL_TESTNET") or os.getenv("HL_TESTNET") == "0"

            sig = sign_l1_action(
                wallet,
                action,
                active_pool=None,
                nonce=timestamp_ms,
                expires_after=None,
                is_mainnet=is_mainnet
            )
            return {
                "action": action,
                "nonce": timestamp_ms,
                "signature": sig,
                "vaultAddress": None,
            }
        except Exception as exc:
            raise AdapterError(f"failed to sign hyperliquid action: {exc}") from exc


class ExternalSigner(Signer):
    """外部签名器：把签名请求转发到外部服务."""

    def __init__(self, wallet_address: str, signer_endpoint: str, timeout: int = 10):
        self.wallet_address = wallet_address
        self.signer_endpoint = signer_endpoint
        self.timeout = timeout

    def sign(self, action: Dict[str, Any], timestamp_ms: int) -> Dict[str, Any]:
        try:
            resp = requests.post(
                self.signer_endpoint,
                json={
                    "wallet_address": self.wallet_address,
                    "action": action,
                    "nonce": timestamp_ms,
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "action": action,
                "nonce": timestamp_ms,
                "signature": data.get("signature", ""),
                "vaultAddress": data.get("vaultAddress"),
            }
        except requests.RequestException as exc:
            raise AdapterError(f"external signer request failed: {exc}") from exc


def create_signer(
    wallet_address: Optional[str] = None,
    private_key: Optional[str] = None,
    signer_endpoint: Optional[str] = None,
) -> Signer:
    """根据环境变量创建签名器."""
    wallet_address = wallet_address or os.getenv("HL_API_WALLET_ADDRESS")
    private_key = private_key or os.getenv("HL_API_PRIVATE_KEY")
    signer_endpoint = signer_endpoint or os.getenv("HL_SIGNER_ENDPOINT")

    if not wallet_address:
        raise ConfigurationError(
            "HL_API_WALLET_ADDRESS is required for hyperliquid live trading"
        )

    if signer_endpoint:
        logger.info("using external hyperliquid signer: %s", signer_endpoint)
        return ExternalSigner(wallet_address, signer_endpoint)

    if private_key:
        logger.warning(
            "using local private key signer; production should use HL_SIGNER_ENDPOINT"
        )
        return LocalPrivateKeySigner(wallet_address, private_key)

    raise ConfigurationError(
        "either HL_SIGNER_ENDPOINT or HL_API_PRIVATE_KEY must be set "
        "for hyperliquid live trading"
    )
