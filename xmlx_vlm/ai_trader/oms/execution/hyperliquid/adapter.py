"""Hyperliquid 执行适配器实现."""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from xmlx_vlm.ai_trader.oms.constants import OrderSide, OrderState, PositionSide
from xmlx_vlm.ai_trader.oms.core.account import AccountSnapshot
from xmlx_vlm.ai_trader.oms.core.order import Fill, Order
from xmlx_vlm.ai_trader.oms.core.position import Position
from xmlx_vlm.ai_trader.oms.core.trade import Trade
from xmlx_vlm.ai_trader.oms.exceptions import AdapterError, ConfigurationError

from xmlx_vlm.ai_trader.oms.execution.hyperliquid.mapper import (
    coin_from_symbol,
    hl_positions_to_positions,
    hl_response_to_order,
    order_to_hl_action,
)
from xmlx_vlm.ai_trader.oms.execution.hyperliquid.signer import Signer, create_signer
from xmlx_vlm.ai_trader.oms.interfaces.execution_adapter import (
    CancelAck,
    ExecutionAdapter,
    OrderAck,
)
from xmlx_vlm.ai_trader.oms.market_data.models import OrderBook, OrderBookLevel, Quote, VolumeProfile
from xmlx_vlm.ai_trader.oms.utils.decimal import to_decimal, ZERO

# 需要读取 spot 余额的统一/组合保证金模式
_UNIFIED_LIKE_MODES = {"unifiedAccount", "portfolioMargin"}

logger = logging.getLogger(__name__)


class HyperliquidExecutionAdapter(ExecutionAdapter):
    """Hyperliquid 实盘执行适配器.

    与 paper（本地仿真机构盘）地位相同，都通过统一 ExecutionAdapter
    接口接入 OMS；区别仅在于本适配器连接真实 Hyperliquid 交易所。
    """

    def __init__(
        self,
        wallet_address: Optional[str] = None,
        private_key: Optional[str] = None,
        signer_endpoint: Optional[str] = None,
        testnet: bool = False,
        timeout: int = 20,
        signer: Optional[Signer] = None,
    ):
        from xmlx_vlm.ai_trader.oms.execution.hyperliquid.client import HyperliquidClient
        try:
            self._client = HyperliquidClient(
                wallet_address=wallet_address, testnet=testnet, timeout=timeout
            )
        except TypeError as err:
            if "timeout" in str(err):
                self._client = HyperliquidClient(
                    wallet_address=wallet_address, testnet=testnet
                )
            else:
                raise
        self._signer = signer or create_signer(
            wallet_address=wallet_address,
            private_key=private_key,
            signer_endpoint=signer_endpoint,
        )
        self._wallet_address = self._signer.wallet_address
        self._account_mode = self._detect_account_abstraction()

        # Store in-memory symbol maps
        self._cloid_to_symbol = {}
        self._order_id_to_symbol = {}

        # Instantiate the SDK Exchange object if private_key is available
        self._exchange_sdk = None
        if private_key:
            try:
                from eth_account import Account
                from hyperliquid.exchange import Exchange
                from hyperliquid.utils import constants
                base_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
                wallet = Account.from_key(private_key)
                
                # Check if the recovered address is different from the DB wallet address
                recovered_addr = wallet.address.lower()
                master_addr = wallet_address.lower() if wallet_address else recovered_addr
                account_address = master_addr if recovered_addr != master_addr else None
                if account_address:
                    logger.info(f"Using Agent Wallet {recovered_addr} to trade on behalf of Master Wallet {master_addr}")
                
                self._exchange_sdk = Exchange(
                    wallet, 
                    base_url, 
                    account_address=account_address,
                    timeout=float(timeout)
                )
            except Exception as e:
                logger.error(f"Failed to initialize Hyperliquid Exchange SDK: {e}")

    @property
    def name(self) -> str:
        return "hyperliquid"

    @property
    def account_mode(self) -> str:
        """当前 Hyperliquid 账户抽象模式."""
        return self._account_mode

    @property
    def is_live(self) -> bool:
        return True

    async def _get_coin_for_order(self, order_id: str, client_order_id: Optional[str] = None) -> str:
        # Check in-memory maps
        if client_order_id and client_order_id in self._cloid_to_symbol:
            return self._cloid_to_symbol[client_order_id]
        if order_id and order_id in self._order_id_to_symbol:
            return self._order_id_to_symbol[order_id]

        # Try to query order from exchange to find the coin
        order = await self.query_order(client_order_id or order_id)
        if order:
            coin = coin_from_symbol(order.symbol)
            if client_order_id:
                self._cloid_to_symbol[client_order_id] = coin
            if order_id:
                self._order_id_to_symbol[order_id] = coin
            return coin

        # Default fallback
        return "BTC"

    async def submit(self, order: Order) -> OrderAck:
        order.exchange = self.name
        coin = coin_from_symbol(order.symbol)

        # Store symbol mapping
        if order.client_order_id:
            self._cloid_to_symbol[order.client_order_id] = coin

        # Use the SDK if available
        if self._exchange_sdk:
            from hyperliquid.utils.types import Cloid
            is_buy = order.side == OrderSide.BUY
            sz = float(order.qty)
            limit_px = float(order.price) if order.price else 0.0
            order_type = {"limit": {"tif": "Gtc"}}

            # Format cloid
            cloid = None
            if order.client_order_id:
                try:
                    cloid_hex = order.client_order_id.replace("-", "")
                    if not cloid_hex.startswith("0x"):
                        cloid_hex = "0x" + cloid_hex
                    if len(cloid_hex) == 34:
                        cloid = Cloid(cloid_hex)
                except Exception as e:
                    logger.warning(f"Failed to parse client_order_id as Cloid: {e}")

            try:
                # Place order synchronously using the SDK
                response = self._exchange_sdk.order(
                    name=coin,
                    is_buy=is_buy,
                    sz=sz,
                    limit_px=limit_px,
                    order_type=order_type,
                    reduce_only=False,
                    cloid=cloid
                )
                order.raw_response = response
                hl_response_to_order(response, order)

                # Store order ID to symbol mapping if received
                if order.order_id:
                    self._order_id_to_symbol[order.order_id] = coin

                return OrderAck(
                    success=not order.is_done() or order.state.value != "rejected",
                    order_id=order.order_id or order.client_order_id,
                    message=order.reject_reason or "submitted",
                    raw=response,
                )
            except Exception as exc:
                order.transition_to(OrderState.REJECTED, reason=str(exc))
                raise AdapterError(f"SDK order placement failed: {exc}") from exc

        # Fallback to manual signing
        action = order_to_hl_action(order)
        timestamp_ms = int(time.time() * 1000)
        payload = self._signer.sign(action, timestamp_ms)

        try:
            response = self._client.exchange(payload)
            order.raw_response = response
            hl_response_to_order(response, order)
            if order.order_id:
                self._order_id_to_symbol[order.order_id] = coin
            return OrderAck(
                success=not order.is_done() or order.state.value != "rejected",
                order_id=order.order_id or order.client_order_id,
                message=order.reject_reason or "submitted",
                raw=response,
            )
        except AdapterError as exc:
            order.transition_to(OrderState.REJECTED, reason=str(exc))
            raise

    async def cancel(self, order_id: str, client_order_id: Optional[str] = None) -> CancelAck:
        coin = await self._get_coin_for_order(order_id, client_order_id)

        # Use the SDK if available
        if self._exchange_sdk:
            try:
                use_cloid = False
                cloid_val = None
                if client_order_id:
                    cloid_hex = client_order_id.replace("-", "")
                    if not cloid_hex.startswith("0x"):
                        cloid_hex = "0x" + cloid_hex
                    if len(cloid_hex) == 34:
                        from hyperliquid.utils.types import Cloid
                        cloid_val = Cloid(cloid_hex)
                        use_cloid = True

                if use_cloid:
                    response = self._exchange_sdk.cancel_by_cloid(coin, cloid_val)
                else:
                    response = self._exchange_sdk.cancel(coin, int(order_id))

                return CancelAck(
                    success=True,
                    order_id=order_id,
                    raw=response,
                )
            except Exception as exc:
                return CancelAck(success=False, order_id=order_id, message=str(exc))

        # Fallback to manual signing
        asset_idx = 0
        try:
            meta = self._client.info({"type": "meta"})
            universe = meta.get("universe", [])
            idx = next((i for i, u in enumerate(universe) if u.get("name") == coin), None)
            if idx is not None:
                asset_idx = idx
        except Exception:
            pass

        use_cloid = False
        cloid_hex = ""
        if client_order_id:
            h = client_order_id.replace("-", "")
            if not h.startswith("0x"):
                h = "0x" + h
            if len(h) == 34:
                cloid_hex = h
                use_cloid = True

        if use_cloid:
            action = {
                "type": "cancelByCloid",
                "cancels": [{"asset": asset_idx, "cloid": cloid_hex}],
            }
        else:
            action = {
                "type": "cancel",
                "cancels": [{"a": asset_idx, "o": int(order_id)}],
            }

        timestamp_ms = int(time.time() * 1000)
        payload = self._signer.sign(action, timestamp_ms)
        try:
            response = self._client.exchange(payload)
            return CancelAck(
                success=True,
                order_id=order_id,
                raw=response,
            )
        except AdapterError as exc:
            return CancelAck(success=False, order_id=order_id, message=str(exc))

    async def query_order(self, order_id: str) -> Optional[Order]:
        """查询 HL 订单状态.

        支持 oid（数字）或 cloid（16 字节 hex）。
        """
        try:
            data = self._client.info(
                {
                    "type": "orderStatus",
                    "user": self._wallet_address,
                    "oid": int(order_id) if (isinstance(order_id, str) and order_id.isdigit()) else order_id,
                }
            )
            logger.debug("order status response: %s", data)
            if not isinstance(data, dict) or data.get("status") != "order":
                return None
            return self._hl_order_status_to_order(data.get("order", {}))
        except AdapterError:
            return None

    async def query_recent_fills(self, limit: int = 100) -> List[Fill]:
        """查询最近成交（兜底用）."""
        try:
            data = self._client.info(
                {
                    "type": "userFills",
                    "user": self._wallet_address,
                }
            )
            if not isinstance(data, list):
                return []
            fills: List[Fill] = []
            for item in data[:limit]:
                if not isinstance(item, dict):
                    continue
                fill = self._hl_fill_to_fill(item)
                if fill:
                    fills.append(fill)
            return fills
        except AdapterError as exc:
            logger.error("failed to query hyperliquid fills: %s", exc)
            return []

    def _hl_order_status_to_order(self, status_data: Dict[str, Any]) -> Optional[Order]:
        """把 HL orderStatus 响应转为内部 Order."""
        hl_order = status_data.get("order", {})
        if not hl_order:
            return None

        coin = hl_order.get("coin", "")
        symbol = f"{coin}/USDC" if not coin.startswith("@") else coin
        side = OrderSide.BUY if hl_order.get("side") == "B" else OrderSide.SELL
        qty = to_decimal(hl_order.get("origSz", "0"))
        remaining = to_decimal(hl_order.get("sz", "0"))
        filled_qty = qty - remaining if qty >= remaining else qty
        price = to_decimal(hl_order.get("limitPx", "0")) or None
        cloid = hl_order.get("cloid")
        oid = str(hl_order.get("oid", ""))

        order = Order(
            symbol=symbol,
            side=side,
            qty=qty,
            order_type="limit",
            price=price,
            client_order_id=cloid or oid,
        )
        order.order_id = oid
        order.filled_qty = filled_qty
        order.remaining_qty = remaining
        order.state = self._hl_status_to_state(status_data.get("status", ""), remaining)
        return order

    def _hl_status_to_state(self, status: str, remaining: Decimal) -> OrderState:
        status = status.lower()
        if status == "filled":
            return OrderState.FILLED
        if status == "open":
            return OrderState.PARTIAL_FILLED if remaining > ZERO else OrderState.ACKNOWLEDGED
        if status in {"canceled", "margincanceled", "selftradecanceled",
                      "reduceonlycanceled", "siblingfilledcanceled", "delistedcanceled",
                      "liquidatedcanceled", "scheduledcancel"}:
            return OrderState.CANCELLED
        if "rejected" in status:
            return OrderState.REJECTED
        return OrderState.ACKNOWLEDGED

    def _hl_fill_to_fill(self, item: Dict[str, Any]) -> Optional[Fill]:
        """把 HL userFills 单项转为内部 Fill."""
        coin = item.get("coin", "")
        symbol = f"{coin}/USDC" if not coin.startswith("@") else coin
        side = OrderSide.BUY if item.get("side") == "B" else OrderSide.SELL
        qty = to_decimal(item.get("sz", "0"))
        if qty <= ZERO:
            return None
        return Fill(
            fill_id=str(item.get("tid", "")),
            order_id=str(item.get("oid", "")),
            symbol=symbol,
            side=side,
            qty=qty,
            price=to_decimal(item.get("px", "0")),
            fee=to_decimal(item.get("fee", "0")),
            timestamp_ms=int(item.get("time", time.time() * 1000)),
            raw=item,
        )

    async def sync_positions(self) -> Dict[str, Position]:
        try:
            data = self._client.info(
                {"type": "clearinghouseState", "user": self._wallet_address}
            )
            positions = data.get("assetPositions", [])
            raw_positions = [p.get("position", {}) for p in positions]
            return hl_positions_to_positions(raw_positions)
        except AdapterError as exc:
            logger.error("failed to sync hyperliquid positions: %s", exc)
            return {}

    def _detect_account_abstraction(self) -> str:
        """自动检测账户抽象模式，失败时回退到 standard 行为."""
        if not hasattr(self._client, "get_user_abstraction"):
            return "disabled"
        try:
            mode = self._client.get_user_abstraction(self._wallet_address)
            if isinstance(mode, str):
                logger.info("hyperliquid account mode: %s", mode)
                return mode
        except AdapterError as exc:
            logger.warning("failed to detect hyperliquid account abstraction: %s", exc)
        return "disabled"

    async def sync_account(self) -> AccountSnapshot:
        try:
            raw_perp = self._client.info(
                {"type": "clearinghouseState", "user": self._wallet_address}
            )
            margin_summary = raw_perp.get("marginSummary", {})
            perp_account_value = to_decimal(margin_summary.get("accountValue", "0"))
            total_margin_used = to_decimal(margin_summary.get("totalMarginUsed", "0"))
            withdrawable = to_decimal(raw_perp.get("withdrawable", "0"))

            if self._account_mode in _UNIFIED_LIKE_MODES:
                # unified / portfolio margin：余额与可用保证金以 spot state 为准
                raw_spot = self._client.get_spot_clearinghouse_state(self._wallet_address)
                balances = raw_spot.get("balances", [])
                # 默认以 USDC 作为结算与保证金资产（token 0 或 coin "USDC"）
                usdc_balance = next(
                    (
                        b
                        for b in balances
                        if b.get("coin") == "USDC" or b.get("token") == 0
                    ),
                    {},
                )
                usdc_total = to_decimal(usdc_balance.get("total", "0"))
                usdc_hold = to_decimal(usdc_balance.get("hold", "0"))
                equity = usdc_total
                available = equity - usdc_hold
                raw = {
                    "mode": self._account_mode,
                    "perp": raw_perp,
                    "spot": raw_spot,
                }
            else:
                equity = perp_account_value
                available = withdrawable
                raw = {"mode": self._account_mode, "perp": raw_perp}

            return AccountSnapshot(
                equity=equity,
                available_margin=available,
                used_margin=total_margin_used,
                total_position_value=total_margin_used,
                cash=available,
                mode=self._account_mode,
                raw=raw,
            )
        except AdapterError as exc:
            logger.error("failed to sync hyperliquid account: %s", exc)
            return AccountSnapshot(mode=self._account_mode)

    def close(self):
        self._client.close()

    # ── 行情接口 ──
    async def get_quote(self, symbol: str) -> Optional[Quote]:
        """从 metaAndAssetCtxs 获取最新报价."""
        try:
            data = self._client.get_meta_and_asset_ctxs()
            if not isinstance(data, list) or len(data) < 2:
                return None
            coin = coin_from_symbol(symbol)
            universe = data[0].get("universe", [])
            ctxs = data[1]
            idx = next((i for i, u in enumerate(universe) if u.get("name") == coin), None)
            if idx is None or idx >= len(ctxs):
                return None
            ctx = ctxs[idx]
            impact = ctx.get("impactPxs", [])
            bid = to_decimal(impact[0]) if impact else None
            ask = to_decimal(impact[1]) if len(impact) > 1 else None
            return Quote(
                symbol=symbol.upper(),
                bid=bid,
                ask=ask,
                mark=to_decimal(ctx.get("markPx", "0")) or None,
                last=to_decimal(ctx.get("midPx", "0")) or None,
                timestamp_ms=int(time.time() * 1000),
                raw=ctx,
            )
        except AdapterError as exc:
            logger.warning("failed to get hyperliquid quote for %s: %s", symbol, exc)
            return None

    async def get_order_book(self, symbol: str, depth: int = 10) -> Optional[OrderBook]:
        """Hyperliquid 没有免费 L2，用 impactPxs 构造合成 order book."""
        quote = await self.get_quote(symbol)
        if quote is None or quote.bid is None or quote.ask is None:
            return None
        # 用日成交量估算合成深度
        try:
            data = self._client.get_meta_and_asset_ctxs()
            coin = coin_from_symbol(symbol)
            universe = data[0].get("universe", [])
            ctxs = data[1]
            idx = next((i for i, u in enumerate(universe) if u.get("name") == coin), None)
            day_base_vlm = to_decimal(ctxs[idx].get("dayBaseVlm", "0")) if idx is not None else ZERO
            depth_qty = day_base_vlm / Decimal("24") if day_base_vlm > ZERO else Decimal("1")
        except Exception:
            depth_qty = Decimal("1")
        return OrderBook(
            symbol=symbol.upper(),
            bids=[OrderBookLevel(price=quote.bid, qty=depth_qty)],
            asks=[OrderBookLevel(price=quote.ask, qty=depth_qty)],
            timestamp_ms=quote.timestamp_ms,
        )

    async def get_recent_volume(
        self, symbol: str, window_seconds: int = 300
    ) -> Optional[Decimal]:
        """按日成交量估算窗口内成交量."""
        try:
            data = self._client.get_meta_and_asset_ctxs()
            if not isinstance(data, list) or len(data) < 2:
                return None
            coin = coin_from_symbol(symbol)
            universe = data[0].get("universe", [])
            ctxs = data[1]
            idx = next((i for i, u in enumerate(universe) if u.get("name") == coin), None)
            if idx is None or idx >= len(ctxs):
                return None
            day_base_vlm = to_decimal(ctxs[idx].get("dayBaseVlm", "0"))
            if day_base_vlm <= ZERO:
                return None
            return day_base_vlm * Decimal(window_seconds) / Decimal("86400")
        except AdapterError as exc:
            logger.warning("failed to get hyperliquid volume for %s: %s", symbol, exc)
            return None

    async def get_volume_profile(
        self,
        symbol: str,
        duration_seconds: int = 86400,
        buckets: int = 24,
    ) -> Optional[VolumeProfile]:
        """通过 candleSnapshot 获取成交量分布."""
        try:
            coin = coin_from_symbol(symbol)
            end_ms = int(time.time() * 1000)
            start_ms = end_ms - duration_seconds * 1000
            # 选择合适的 interval
            if duration_seconds <= 3600:
                interval = "1m"
            elif duration_seconds <= 86400:
                interval = "15m"
            else:
                interval = "1h"
            candles = self._client.get_candles(coin, interval, start_ms, end_ms)
            if not isinstance(candles, list) or not candles:
                return None
            total_volume = sum(
                (to_decimal(c.get("v", "0")) for c in candles if isinstance(c, dict)),
                ZERO,
            )
            # 按桶数聚合成 weights
            per_bucket = max(1, len(candles) // buckets)
            bucket_volumes = []
            current = ZERO
            for i, c in enumerate(candles):
                if isinstance(c, dict):
                    current += to_decimal(c.get("v", "0"))
                if (i + 1) % per_bucket == 0:
                    bucket_volumes.append(current)
                    current = ZERO
            if current > ZERO:
                bucket_volumes.append(current)
            # 补齐桶数
            while len(bucket_volumes) < buckets:
                bucket_volumes.append(ZERO)
            return VolumeProfile(
                symbol=symbol.upper(),
                total_volume=total_volume,
                buckets=bucket_volumes[:buckets],
                bucket_labels=[f"bucket_{i}" for i in range(len(bucket_volumes[:buckets]))],
            )
        except AdapterError as exc:
            logger.warning("failed to get hyperliquid volume profile for %s: %s", symbol, exc)
            return None

    async def get_volatility(
        self,
        symbol: str,
        window_days: int = 30,
    ) -> Optional[Decimal]:
        """用日线收盘价标准差估算日波动率."""
        try:
            coin = coin_from_symbol(symbol)
            end_ms = int(time.time() * 1000)
            start_ms = end_ms - window_days * 86400 * 1000
            candles = self._client.get_candles(coin, "1d", start_ms, end_ms)
            if not isinstance(candles, list) or len(candles) < 2:
                return None
            closes = [to_decimal(c.get("c", "0")) for c in candles if isinstance(c, dict) and c.get("c")]
            if len(closes) < 2:
                return None
            # 计算对数收益率标准差
            import math
            returns = []
            for i in range(1, len(closes)):
                if closes[i - 1] > ZERO:
                    returns.append(float((closes[i] / closes[i - 1]).ln()))
            if not returns:
                return None
            mean = sum(returns) / len(returns)
            variance = sum((r - mean) ** 2 for r in returns) / len(returns)
            std = math.sqrt(variance)
            return Decimal(str(std))
        except AdapterError as exc:
            logger.warning("failed to get hyperliquid volatility for %s: %s", symbol, exc)
            return None
