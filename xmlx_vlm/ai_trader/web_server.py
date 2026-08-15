"""FastAPI Web Server for AI Trader platform - serving local web UI dashboard."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
import sys
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Add package root to sys.path if not present
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from xmlx_vlm.ai_trader.config import DATA_DIR, LOGS_DIR, DEFAULT_SERVER_URL, DEFAULT_API_KEY
from xmlx_vlm.ai_trader.store.session_db import QuantSessionDB
from xmlx_vlm.ai_trader.agent.agent_loop import AITraderAgent
from xmlx_vlm.ai_trader.store import vault
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine

logger = logging.getLogger("xmlx_vlm.ai_trader.web_server")

# Global instances
db: Optional[QuantSessionDB] = None
agent: Optional[AITraderAgent] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI to manage startup/shutdown tasks."""
    logger.info("Initializing SQLite Session DB...")
    global db, agent
    db = QuantSessionDB()
    
    # Auto setup secure vault
    _auto_init_and_unlock_vault()
    
    # Initialize AITraderAgent
    logger.info("Initializing AITraderAgent Loop...")
    agent = AITraderAgent(
        db=db,
        server_url=app.state.server_url,
        api_key=app.state.api_key,
        model_path=app.state.model,
        local_only=app.state.local,
        temperature=app.state.temperature,
        max_tokens=app.state.max_tokens,
        live=app.state.live,
        exchange=app.state.exchange,
        risk_profile=app.state.risk_profile,
        dry_run=app.state.dry_run,
    )
    
    # Load agent (connects to inference server or loads model locally)
    await agent.load_agent()
    
    # Start live market service in background
    logger.info("Initializing live market data service...")
    try:
        from xmlx_vlm.ai_trader.tools.market import _get_live_service
        _get_live_service()
    except Exception as e:
        logger.warning("Failed to start live market service: %s", e)

    # Mock decisions are disabled platform-wide

    yield
    
    # Shutdown tasks
    try:
        from xmlx_vlm.ai_trader.tools.market import _get_live_service
        svc = _get_live_service()
        if svc:
            logger.info("Stopping live market service...")
            svc.stop()
    except Exception:
        pass

    # Disconnect MCP servers
    if agent and agent.registry:
        try:
            logger.info("Disconnecting from MCP servers...")
            await agent.registry.disconnect_mcp_servers()
        except Exception as e:
            logger.warning("Failed to disconnect MCP servers: %s", e)

    logger.info("Web server shutdown completed.")


app = FastAPI(
    title="AI Trader Quant Agent Chat Platform",
    description="Dedicated Quant Trading Agent Chat and Dashboard Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST API Router ──

@app.get("/api/sessions")
def list_sessions():
    """List all available chat sessions."""
    try:
        return db.list_sessions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sessions")
def create_session(payload: Dict[str, str]):
    """Create a new chat session."""
    try:
        session_id = payload.get("session_id") or str(uuid.uuid4())
        title = payload.get("title") or "New Session"
        model_name = agent.server_model if agent.use_server else agent.model_path
        
        session = db.create_session(
            session_id=session_id,
            title=title,
            model=model_name or "default",
            mode="live" if agent.live else "paper",
        )
        return session
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a chat session and its history."""
    try:
        db.delete_session(session_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str):
    """Retrieve chat history for a session."""
    try:
        return db.get_messages(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sessions/{session_id}/trades")
def get_trades(session_id: str):
    """Retrieve trades executed during a session."""
    try:
        return db.get_trades(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── KMS Vault API Endpoints ──

@app.post("/api/kms/init")
def kms_init(payload: Dict[str, str]):
    """Initialize the KMS vault with a master password."""
    password = payload.get("password")
    if not password or len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")
    
    # Check if already initialized
    if db.get_kms_config("vault_initialized") == "true":
        raise HTTPException(status_code=400, detail="Vault is already initialized")
        
    import hashlib
    import os
    salt = os.urandom(16)
    salt_hex = salt.hex()
    verifier_hex = vault.derive_key(password, salt).hex()
    
    db.init_kms_vault(salt_hex, verifier_hex)
    db.log_kms_audit("VAULT_INIT", "Vault initialized with master password")
    return {"status": "success", "message": "Vault successfully initialized"}


@app.post("/api/kms/status")
def kms_status():
    """Get the status of the KMS vault."""
    initialized = db.get_kms_config("vault_initialized") == "true"
    unlocked = vault.is_unlocked()
    active_credential = vault.get_active_credential()
    
    active_key_info = None
    if active_credential:
        active_key_info = {
            "label": active_credential.get("label"),
            "wallet_address": active_credential.get("wallet_address"),
            "testnet": active_credential.get("testnet")
        }
        
    return {
        "initialized": initialized,
        "unlocked": unlocked,
        "active_key": active_key_info,
        "hsm_state": "ONLINE" if unlocked else "OFFLINE",
        "encryption_algorithm": "HMAC-SHA256-CTR (FIPS-compliant software enclave)"
    }


@app.post("/api/kms/unlock")
def kms_unlock(payload: Dict[str, str]):
    """Unlock the vault with the master password and decrypt all saved keys into memory."""
    password = payload.get("password")
    if not password:
        raise HTTPException(status_code=400, detail="Password is required")
        
    if db.get_kms_config("vault_initialized") != "true":
        raise HTTPException(status_code=400, detail="Vault is not initialized")
        
    salt_hex = db.get_kms_config("vault_salt")
    verifier_hex = db.get_kms_config("vault_verifier")
    
    # Verify password
    salt = bytes.fromhex(salt_hex)
    derived = vault.derive_key(password, salt)
    if derived.hex() != verifier_hex:
        db.log_kms_audit("VAULT_UNLOCK_FAILED", "Failed unlock attempt: incorrect password")
        raise HTTPException(status_code=401, detail="Incorrect master password")
        
    # Decrypt all keys and store in memory
    encrypted_keys = db.list_kms_keys()
    unlocked_creds = {}
    active_wallet = None
    
    for key_row in encrypted_keys:
        key_id = key_row["key_id"]
        full_row = db.get_encrypted_kms_key(key_id)
        enc_payload_str = full_row["encrypted_private_key"]
        
        try:
            enc_dict = json.loads(enc_payload_str)
            decrypted_private_key = vault.decrypt_data(enc_dict, password)
            
            unlocked_creds[full_row["wallet_address"]] = {
                "key_id": key_id,
                "label": full_row["label"],
                "wallet_address": full_row["wallet_address"],
                "private_key": decrypted_private_key,
                "testnet": bool(full_row["testnet"])
            }
            if full_row["status"] == "active":
                active_wallet = full_row["wallet_address"]
        except Exception as e:
            logger.error(f"Failed to decrypt key {key_id}: {e}")
            
    vault.set_unlocked_credentials(unlocked_creds)
    vault.set_active_wallet(active_wallet)
    
    # Dynamically update the OMS adapter if an active key exists
    if active_wallet:
        active_cred = unlocked_creds[active_wallet]
        _activate_key_in_oms(active_cred)
        
    db.log_kms_audit("VAULT_UNLOCKED", f"Vault successfully unlocked. Decrypted {len(unlocked_creds)} key(s).")
    return {"status": "success", "message": f"Vault unlocked. Loaded {len(unlocked_creds)} keys."}


@app.post("/api/kms/lock")
def kms_lock():
    """Lock the vault and clear all decrypted keys from memory."""
    vault.wipe_vault()
    _deactivate_key_in_oms()
    db.log_kms_audit("VAULT_LOCKED", "Vault locked, memory wiped.")
    return {"status": "success", "message": "Vault successfully locked"}


@app.get("/api/kms/keys")
def kms_list_keys():
    """List all stored wallet details (with private keys hidden)."""
    if not vault.is_unlocked():
        return []
        
    return db.list_kms_keys()


@app.post("/api/kms/keys")
def kms_add_key(payload: Dict[str, Any]):
    """Add a new exchange key to the vault."""
    if not vault.is_unlocked():
        raise HTTPException(status_code=403, detail="Vault is locked")
        
    label = payload.get("label")
    wallet_address = payload.get("wallet_address")
    private_key = payload.get("private_key")
    password = os.environ.get("XMLX_VLM_VAULT_PASSWORD", "xmlx_vlm_default_secure_vault_passphrase_123456!")
    testnet_val = payload.get("testnet", False)
    if isinstance(testnet_val, str):
        testnet = testnet_val.lower() in ("true", "1", "yes", "on")
    else:
        testnet = bool(testnet_val)
    
    if not label or not wallet_address or not private_key:
        raise HTTPException(status_code=400, detail="Missing required fields")
        
    salt_hex = db.get_kms_config("vault_salt")
    verifier_hex = db.get_kms_config("vault_verifier")
    salt = bytes.fromhex(salt_hex)
    derived = vault.derive_key(password, salt)
    if derived.hex() != verifier_hex:
        raise HTTPException(status_code=401, detail="Incorrect master password")
        
    try:
        enc_dict = vault.encrypt_data(private_key, password)
        encrypted_key_str = json.dumps(enc_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encryption failed: {e}")
        
    key_id = str(uuid.uuid4())
    db.add_kms_key(key_id, label, wallet_address, encrypted_key_str, testnet)
    
    unlocked = vault.get_unlocked_credentials()
    unlocked[wallet_address] = {
        "key_id": key_id,
        "label": label,
        "wallet_address": wallet_address,
        "private_key": private_key,
        "testnet": testnet
    }
    vault.set_unlocked_credentials(unlocked)
    vault.set_active_wallet(wallet_address)
    _activate_key_in_oms(unlocked[wallet_address])
    
    db.log_kms_audit("KEY_ADDED", f"Added wallet key: {label} ({wallet_address[:6]}...{wallet_address[-4:]}) and set as active.")
    return {"status": "success", "message": "Key securely encrypted and saved to database and set as active"}


@app.delete("/api/kms/keys/{key_id}")
def kms_delete_key(key_id: str):
    """Delete a key from the vault."""
    if not vault.is_unlocked():
        raise HTTPException(status_code=403, detail="Vault is locked")
        
    enc_key = db.get_encrypted_kms_key(key_id)
    if not enc_key:
        raise HTTPException(status_code=404, detail="Key not found")
        
    db.delete_kms_key(key_id)
    
    unlocked = vault.get_unlocked_credentials()
    addr = enc_key["wallet_address"]
    if addr in unlocked:
        del unlocked[addr]
    vault.set_unlocked_credentials(unlocked)
    
    if vault.get_active_wallet() == addr:
        vault.set_active_wallet(None)
        _deactivate_key_in_oms()
        
    db.log_kms_audit("KEY_DELETED", f"Deleted wallet key: {enc_key['label']} ({addr[:6]}...{addr[-4:]})")
    return {"status": "success"}


@app.post("/api/kms/keys/{key_id}/activate")
def kms_activate_key(key_id: str):
    """Set a key as active for trading and dynamic adapter configuration."""
    if not vault.is_unlocked():
        raise HTTPException(status_code=403, detail="Vault is locked")
        
    activated_key = db.activate_kms_key(key_id)
    if not activated_key:
        raise HTTPException(status_code=404, detail="Key not found")
        
    wallet_address = activated_key["wallet_address"]
    unlocked = vault.get_unlocked_credentials()
    
    if wallet_address not in unlocked:
        raise HTTPException(status_code=500, detail="Key decrypted state missing in memory. Please unlock vault again.")
        
    vault.set_active_wallet(wallet_address)
    active_cred = unlocked[wallet_address]
    
    _activate_key_in_oms(active_cred)
    
    db.log_kms_audit("KEY_ACTIVATED", f"Activated key for live execution: {active_cred['label']} ({wallet_address[:6]}...{wallet_address[-4:]})")
    return {"status": "success", "message": f"Active trading key switched to: {active_cred['label']}"}


@app.post("/api/kms/keys/deactivate")
def kms_deactivate_key():
    """Deactivate the active key and switch back to local exchange."""
    if not vault.is_unlocked():
        raise HTTPException(status_code=403, detail="Vault is locked")
        
    db.deactivate_all_kms_keys()
    vault.set_active_wallet(None)
    _deactivate_key_in_oms()
    
    db.log_kms_audit("KEY_DEACTIVATED", "Deactivated active key, switched to local exchange.")
    return {"status": "success", "message": "Switched to local exchange."}


@app.get("/api/kms/audit")
def kms_audit_logs():
    """Fetch vault security logs."""
    return db.get_kms_audit_logs(limit=50)


# ── Hyperliquid KMS Proxy API Endpoints ──

@app.get("/api/kms/exchange/assets")
async def get_kms_exchange_assets(wallet: Optional[str] = None):
    """Fetch wallet equity and spot balances from Hyperliquid using the active key, or return local simulator state."""
    cred = _ensure_oms_has_wallet(wallet)
    
    try:
        oms = _get_oms()
        await oms.sync()
        summary = oms.portfolio_summary()
        acc = summary.get("account", {})
        avail = float(acc.get("available_margin", 10000.0))
        equity = float(acc.get("equity", 10000.0))
        used = float(acc.get("used_margin", 0.0))
        
        if oms.is_live and cred:
            client = oms.adapter._client
            wallet_address = oms.adapter._wallet_address
            network = "Testnet" if client.testnet else "Mainnet"
            
            spot_balances = []
            try:
                spot_state = client.get_spot_clearinghouse_state(wallet_address)
                balances = spot_state.get("balances", [])
                for b in balances:
                    coin = b.get("coin", "USDC")
                    total = float(b.get("total", 0))
                    hold = float(b.get("hold", 0))
                    spot_balances.append({
                        "coin": coin,
                        "total": total,
                        "available": total - hold,
                        "hold": hold
                    })
            except Exception as spot_err:
                logger.warning("Failed to fetch spot state: %s", spot_err)
                
            return {
                "account_address": wallet_address,
                "label": cred["label"],
                "network": network,
                "perp_equity": equity,
                "available_margin": avail,
                "used_margin": used,
                "spot_balances": spot_balances
            }
        else:
            spot_balances = [{
                "coin": "USDC",
                "total": equity,
                "available": avail,
                "hold": used
            }]
            return {
                "account_address": "0xLocalExchangeAccountAddress",
                "label": "Local Exchange",
                "network": "Local Network",
                "perp_equity": equity,
                "available_margin": avail,
                "used_margin": used,
                "spot_balances": spot_balances
            }
    except Exception as exc:
        logger.error("Failed to fetch exchange assets via OMS: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/kms/exchange/positions")
async def get_kms_exchange_positions(wallet: Optional[str] = None):
    """Fetch active perpetual positions from Hyperliquid, or return local simulated positions."""
    _ensure_oms_has_wallet(wallet)
    
    try:
        oms = _get_oms()
        await oms.sync()
        summary = oms.portfolio_summary()
        raw_positions = summary.get("positions", [])
        
        positions = []
        for p in raw_positions:
            qty = float(p.get("qty", 0))
            if qty == 0:
                continue
            avg_px = float(p.get("avg_entry_price", 0))
            positions.append({
                "symbol": p.get("symbol", ""),
                "side": p.get("side", "long").upper(),
                "qty": qty,
                "avg_entry_price": avg_px,
                "mark_price": float(p.get("mark_price", avg_px)),
                "unrealized_pnl": float(p.get("unrealized_pnl", 0)),
                "leverage": int(p.get("leverage", 1)),
                "margin_type": p.get("margin_type", "cross"),
                "liq_price": float(p.get("liq_price", 0.0))
            })
        return positions
    except Exception as exc:
        logger.error("Failed to fetch exchange positions via OMS: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/kms/exchange/orders")
def get_kms_exchange_orders(wallet: Optional[str] = None):
    """Fetch open limit orders from Hyperliquid, or return simulated open orders."""
    _ensure_oms_has_wallet(wallet)
    
    try:
        oms = _get_oms()
        
        if oms.is_live:
            client = oms.adapter._client
            wallet_address = oms.adapter._wallet_address
            data = client.info({"type": "frontendOpenOrders", "user": wallet_address})
            
            orders = []
            if isinstance(data, list):
                for o in data:
                    coin = o.get("coin", "")
                    side = "BUY" if o.get("side") == "B" else "SELL"
                    price = float(o.get("limitPx", 0))
                    size = float(o.get("sz", 0))
                    oid = o.get("oid")
                    cloid = o.get("cloid", "")
                    timestamp = int(o.get("timestamp", 0)) / 1000.0
                    
                    orders.append({
                        "order_id": str(oid),
                        "client_order_id": cloid,
                        "symbol": f"{coin}/USDC",
                        "side": side,
                        "qty": size,
                        "price": price,
                        "type": "LIMIT",
                        "timestamp": timestamp
                    })
            return orders
        else:
            orders = []
            for o in oms._orders.values():
                if o.order_type.value == "limit" and not o.is_done():
                    orders.append({
                        "order_id": o.client_order_id,
                        "client_order_id": o.client_order_id,
                        "symbol": o.symbol,
                        "side": o.side.value.upper(),
                        "qty": float(o.qty),
                        "price": float(o.price) if o.price else 0.0,
                        "type": "LIMIT",
                        "timestamp": o.timestamp_ms / 1000.0
                    })
            return orders
    except Exception as exc:
        logger.error("Failed to fetch open orders via OMS: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/kms/exchange/history")
def get_kms_exchange_history(wallet: Optional[str] = None):
    """Fetch fills/trades history from Hyperliquid, or return simulated trade history."""
    _ensure_oms_has_wallet(wallet)
    
    try:
        oms = _get_oms()
        
        if oms.is_live:
            client = oms.adapter._client
            wallet_address = oms.adapter._wallet_address
            data = client.info({"type": "userFills", "user": wallet_address})
            
            history = []
            if isinstance(data, list):
                for item in data:
                    coin = item.get("coin", "")
                    side = "BUY" if item.get("side") == "B" else "SELL"
                    price = float(item.get("px", 0))
                    size = float(item.get("sz", 0))
                    fee = float(item.get("fee", 0))
                    timestamp = int(item.get("time", 0)) / 1000.0
                    tid = item.get("tid")
                    pnl = float(item.get("pnl", 0))
                    
                    history.append({
                        "trade_id": str(tid),
                        "symbol": f"{coin}/USDC",
                        "side": side,
                        "qty": size,
                        "price": price,
                        "fee": fee,
                        "pnl": pnl,
                        "timestamp": timestamp
                    })
            return history
        else:
            with db._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM trades ORDER BY timestamp DESC LIMIT 100"
                ).fetchall()
                history = []
                for row in rows:
                    history.append({
                        "trade_id": row["trade_id"],
                        "symbol": row["symbol"],
                        "side": row["side"].upper(),
                        "qty": float(row["qty"]),
                        "price": float(row["price"]),
                        "fee": 0.0,
                        "pnl": float(row["pnl"]) if row["pnl"] else 0.0,
                        "timestamp": row["timestamp"]
                    })
                return history
    except Exception as exc:
        logger.error("Failed to fetch trade history via OMS: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/kms/exchange/cancel_order")
async def kms_cancel_order(payload: Dict[str, str]):
    """Cancel an active order directly using KMS credentials, or cancel simulated order."""
    order_id = payload.get("order_id")
    if not order_id:
        raise HTTPException(status_code=400, detail="Missing order_id")
        
    wallet = payload.get("wallet")
    _ensure_oms_has_wallet(wallet)
    
    try:
        oms = _get_oms()
        res = await oms.cancel_order(order_id)
        
        mode_str = "KMS adapter" if oms.is_live else "local simulator"
        db.log_kms_audit("ORDER_CANCEL", f"Cancelled order {order_id} using {mode_str}.")
        return res
    except Exception as e:
        logger.error("Failed to cancel order via OMS: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/kms/exchange/close_position")
async def kms_close_position(payload: Dict[str, str]):
    """Close a perpetual position using KMS credentials, or close simulated position."""
    symbol = payload.get("symbol")
    if not symbol:
        raise HTTPException(status_code=400, detail="Missing symbol")
        
    wallet = payload.get("wallet")
    _ensure_oms_has_wallet(wallet)
    
    try:
        oms = _get_oms()
        order = await oms.close_position(symbol)
        if order is None:
            raise HTTPException(status_code=400, detail=f"No active position found for {symbol}")
            
        res = f"已平仓 {order.symbol}: {order.side.value} {order.qty}"
        
        mode_str = "KMS adapter" if oms.is_live else "local simulator"
        db.log_kms_audit("POSITION_CLOSE", f"Closed position for {symbol} using {mode_str}.")
        return {"status": "success", "message": res}
    except Exception as e:
        logger.error("Failed to close position via OMS: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# Helpers to dynamically link KMS key to running agent OMS

_global_oms_engine: Optional[OMSEngine] = None

def _get_oms() -> OMSEngine:
    """Get the current OMS Engine, prioritizing Agent's tool, falling back to a global instance."""
    global _global_oms_engine
    if agent and agent.registry:
        try:
            trading_tool = agent.registry.get_tool("trading")
            if trading_tool and trading_tool.oms:
                return trading_tool.oms
        except Exception:
            pass
            
    if _global_oms_engine is not None:
        return _global_oms_engine
        
    import xmlx_vlm.ai_trader.oms.config.settings as settings_mod
    cred = vault.get_active_credential()
    if cred:
        sett = settings_mod.get_settings(
            live=False,
            exchange="hyperliquid",
        )
        sett.wallet_address = cred["wallet_address"]
        sett.private_key = cred["private_key"]
        sett.testnet = cred["testnet"]
        sett.live_enabled = True
        _global_oms_engine = OMSEngine(settings=sett)
        return _global_oms_engine
    else:
        sett = settings_mod.get_settings()
        _global_oms_engine = OMSEngine(settings=sett)
        return _global_oms_engine


def _ensure_oms_has_wallet(wallet: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Ensure that the OMS engine is dynamically using the specified wallet credential."""
    cred = None
    if wallet:
        cred = vault.get_unlocked_credentials().get(wallet)
    if not cred:
        cred = vault.get_active_credential()
        
    if cred:
        try:
            oms = _get_oms()
            if not oms.is_live or oms.adapter._wallet_address != cred["wallet_address"]:
                _activate_key_in_oms(cred)
        except Exception as e:
            logger.error(f"Failed to ensure wallet in OMS: {e}")
    return cred


def _activate_key_in_oms(cred: Dict[str, Any]):
    """Inject decrypted KMS credential details into running OMS configuration."""
    try:
        import xmlx_vlm.ai_trader.oms.config.settings as settings_mod
        settings_mod.reset_settings()
        sett = settings_mod.get_settings(
            live=False,
            exchange="hyperliquid",
        )
        sett.wallet_address = cred["wallet_address"]
        sett.private_key = cred["private_key"]
        sett.testnet = cred["testnet"]
        sett.live_enabled = True
        sett.validate_live()
        
        global _global_oms_engine
        _global_oms_engine = OMSEngine(
            settings=sett,
        )
        
        if agent:
            agent.live = True
            agent.exchange = "hyperliquid"
            trading_tool = agent.registry.get_tool("trading")
            if trading_tool:
                trading_tool._oms = _global_oms_engine
                logger.info("OMS Engine dynamically reloaded with decrypted KMS credential.")
    except Exception as e:
        logger.error(f"Failed to inject KMS credential into agent OMS: {e}")


def _deactivate_key_in_oms():
    """Reset running agent OMS to default configurations."""
    try:
        import xmlx_vlm.ai_trader.oms.config.settings as settings_mod
        settings_mod.reset_settings()
        sett = settings_mod.get_settings()
        
        global _global_oms_engine
        _global_oms_engine = OMSEngine(
            settings=sett,
        )
        
        if agent:
            agent.live = False
            agent.exchange = "local"
            trading_tool = agent.registry.get_tool("trading")
            if trading_tool:
                trading_tool._oms = _global_oms_engine
                logger.info("OMS Engine reset to default configuration.")
    except Exception as e:
        logger.error(f"Failed to reset agent OMS: {e}")


def _auto_init_and_unlock_vault():
    """Auto-initialize and unlock KMS vault at web server startup using default password."""
    global db
    if db is None:
        logger.error("Database not initialized, cannot setup vault.")
        return
        
    password = os.environ.get("XMLX_VLM_VAULT_PASSWORD", "xmlx_vlm_default_secure_vault_passphrase_123456!")
    
    # Check if verifier matches the default password
    initialized = db.get_kms_config("vault_initialized") == "true"
    if initialized:
        salt_hex = db.get_kms_config("vault_salt")
        verifier_hex = db.get_kms_config("vault_verifier")
        try:
            salt = bytes.fromhex(salt_hex)
            derived = vault.derive_key(password, salt)
            if derived.hex() != verifier_hex:
                logger.warning("Vault verifier mismatch detected (possibly old vault configuration). Re-initializing vault automatically...")
                with db._get_conn() as conn:
                    conn.execute("DELETE FROM kms_config")
                    conn.execute("DELETE FROM kms_keys")
                    conn.commit()
                initialized = False
        except Exception as e:
            logger.error(f"Error validating existing vault verifier: {e}. Resetting...")
            try:
                with db._get_conn() as conn:
                    conn.execute("DELETE FROM kms_config")
                    conn.execute("DELETE FROM kms_keys")
                    conn.commit()
            except Exception:
                pass
            initialized = False

    # 1. Initialize secure vault if not already initialized in SQLite DB
    if not initialized:
        logger.info("Auto-initializing secure KMS vault...")
        try:
            salt = os.urandom(16)
            salt_hex = salt.hex()
            verifier_hex = vault.derive_key(password, salt).hex()
            db.init_kms_vault(salt_hex, verifier_hex)
            db.log_kms_audit("VAULT_INIT", "Vault auto-initialized by system")
        except Exception as e:
            logger.error(f"Failed to auto-initialize secure vault: {e}")
            return
            
    # 2. Unlock vault and load decrypted keys into secure volatile memory
    logger.info("Auto-unlocking secure KMS vault...")
    try:
        salt_hex = db.get_kms_config("vault_salt")
        verifier_hex = db.get_kms_config("vault_verifier")
        salt = bytes.fromhex(salt_hex)
        derived = vault.derive_key(password, salt)
        if derived.hex() != verifier_hex:
            logger.error("Auto-unlock failed: verifier mismatch")
            return
            
        encrypted_keys = db.list_kms_keys()
        unlocked_creds = {}
        active_wallet = None
        
        for key_row in encrypted_keys:
            key_id = key_row["key_id"]
            full_row = db.get_encrypted_kms_key(key_id)
            enc_payload_str = full_row["encrypted_private_key"]
            try:
                enc_dict = json.loads(enc_payload_str)
                decrypted_private_key = vault.decrypt_data(enc_dict, password)
                unlocked_creds[full_row["wallet_address"]] = {
                    "key_id": key_id,
                    "label": full_row["label"],
                    "wallet_address": full_row["wallet_address"],
                    "private_key": decrypted_private_key,
                    "testnet": bool(full_row["testnet"])
                }
                if full_row["status"] == "active":
                    active_wallet = full_row["wallet_address"]
            except Exception as e:
                logger.error(f"Failed to decrypt key {key_id}: {e}")
                
        vault.set_unlocked_credentials(unlocked_creds)
        vault.set_active_wallet(active_wallet)
        
        if active_wallet:
            active_cred = unlocked_creds[active_wallet]
            _activate_key_in_oms(active_cred)
        logger.info(f"KMS vault auto-unlocked successfully. Loaded {len(unlocked_creds)} key(s).")
    except Exception as e:
        logger.error(f"Failed to auto-unlock KMS vault: {e}")


# ── Market Watches ──

@app.get("/api/market/watchlist")
def get_watchlist():
    """Fetch watchlist tickers from live MarketDataService, fallback to REST."""
    from xmlx_vlm.ai_trader.tools.market import _get_live_service, MarketDataTool
    
    svc = _get_live_service()
    if not svc:
        # Fallback: query hyperliquid tickers via REST for BTC and ETH
        try:
            tool = MarketDataTool()
            tickers = []
            for sym in ["BTC", "ETH"]:
                text = tool.get_ticker(sym, "hyperliquid")
                # Parse output to extract last price
                import re
                match = re.search(r"mark=([\d,]+\.?\d*)", text)
                price = float(match.group(1).replace(",", "")) if match else 0.0
                match_pct = re.search(r"24h_change=([\+\-]*[\d\.]+)%", text)
                change = float(match_pct.group(1)) if match_pct else 0.0
                tickers.append({
                    "symbol": sym,
                    "price": price,
                    "change_24h_pct": change,
                    "volume_24h": 0.0
                })
            return tickers
        except Exception as e:
            logger.error("REST fallback watchlist query failed: %s", e)
            return []

    # Read from live WS memory state
    try:
        watched_symbols = svc.get_watched_coins()
        watchlist = []
        for sym in watched_symbols:
            summary = svc.get_summary(sym, light=True)
            if summary:
                watchlist.append({
                    "symbol": sym,
                    "price": summary.mark_price,
                    "change_24h_pct": summary.change_24h_pct,
                    "volume_24h": summary.volume_24h,
                    "cvd_15m": summary.cvd_15m,
                    "funding_rate": summary.funding_rate
                })
        return watchlist
    except Exception as e:
        logger.error("Failed to read from live market service: %s", e)
        return []


@app.websocket("/api/market/watchlist/ws")
async def watchlist_websocket(websocket: WebSocket):
    """WebSocket endpoint to stream watchlist tickers in real-time."""
    await websocket.accept()
    
    from xmlx_vlm.ai_trader.tools.market import _get_live_service
    from xmlx_vlm.ai_trader.market_service.events import PriceUpdateEvent, FundingUpdateEvent
    
    svc = _get_live_service()
    if not svc:
        # Fallback loop: pull via REST/DB and push periodically
        try:
            while True:
                watchlist_data = get_watchlist()
                await websocket.send_json(watchlist_data)
                await asyncio.sleep(5.0)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logger.error("Error in fallback watchlist websocket: %s", e)
            try:
                await websocket.close()
            except Exception:
                pass
        return

    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def event_handler(event: Any) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, True)
        except Exception:
            pass

    # Subscribe to events
    svc.event_bus.subscribe(PriceUpdateEvent, event_handler)
    svc.event_bus.subscribe(FundingUpdateEvent, event_handler)

    async def send_loop():
        # Send initial state
        watchlist_data = get_watchlist()
        await websocket.send_json(watchlist_data)

        while True:
            try:
                await asyncio.wait_for(queue.get(), timeout=5.0)
                # Throttle to batch multiple rapid updates (reduce server/UI load)
                await asyncio.sleep(1.5)
                while not queue.empty():
                    queue.get_nowait()
            except asyncio.TimeoutError:
                pass

            watchlist_data = get_watchlist()
            await websocket.send_json(watchlist_data)

    async def recv_loop():
        while True:
            await websocket.receive_text()

    send_task = asyncio.create_task(send_loop())
    recv_task = asyncio.create_task(recv_loop())

    try:
        done, pending = await asyncio.wait(
            [send_task, recv_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        # Propagate exceptions if any
        for task in done:
            task.result()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error("Watchlist WebSocket connection ended with error: %s", e)
    finally:
        send_task.cancel()
        recv_task.cancel()
        svc.event_bus.unsubscribe(PriceUpdateEvent, event_handler)
        svc.event_bus.unsubscribe(FundingUpdateEvent, event_handler)


# ── OMS Portfolios ──

@app.get("/api/oms/portfolio")
async def get_portfolio():
    """Query current positions, balance, and account metrics from the OMS."""
    try:
        trading_tool = agent.registry.get_tool("trading")
        oms = trading_tool.oms
        
        # Trigger portfolio/position sync with exchange (non-blocking thread execution)
        await oms.sync()
        summary = oms.portfolio_summary()
        return summary
    except Exception as e:
        logger.error("Failed to fetch OMS portfolio: %s", e)
        # Return empty mock format
        return {
            "account": {"available_margin": "0.0", "margin_utilization_pct": "0.0"},
            "positions": [],
            "unrealized_pnl": "0.0",
            "realized_pnl": "0.0",
            "margin_utilization_pct": "0"
        }


@app.post("/api/oms/emergency_stop")
async def trigger_emergency_stop():
    """Trigger emergency liquidation / circuit breaker."""
    try:
        oms = _get_oms()
        res = await oms.emergency_stop(flatten=True)
        flatten_count = len(res.get("flatten_results", []))
        msg = f"急停已触发，已执行全平操作（处理 {flatten_count} 个持仓标的）并锁定新开仓。"
        return {"status": "success", "message": msg, "details": res}
    except Exception as e:
        logger.exception("Emergency stop failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/config")
def get_current_config():
    """Get current configuration of the platform."""
    return {
        "model": agent.server_model if agent.use_server else agent.model_path,
        "mode": "live" if agent.live else "local",
        "exchange": "local" if agent.exchange == "paper" else agent.exchange,
        "risk_profile": agent.risk_profile,
        "server_connected": agent.use_server,
    }



@app.get("/api/strategy/list")
def get_strategy_list():
    """Get list of unique strategy IDs (trader_id) from the decision store."""
    from xmlx_vlm.ai_trader.store.sqlite_store import SQLiteDecisionStore
    store = SQLiteDecisionStore(LOGS_DIR / "ai_trader.db")
    try:
        cursor = store._conn.execute("SELECT DISTINCT trader_id FROM decision_records")
        return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logger.error("Failed to fetch strategy list: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        store.close()


@app.get("/api/strategy/decisions")
def get_strategy_decisions(trader_id: Optional[str] = None, limit: int = 20, offset: int = 0):
    """Fetch strategy decisions and COT audits from the decision store."""
    from xmlx_vlm.ai_trader.store.sqlite_store import SQLiteDecisionStore
    store = SQLiteDecisionStore(LOGS_DIR / "ai_trader.db")
    try:
        if not trader_id:
            trader_id = "trend_follow_btc_paper"
        records = store.list_decisions(trader_id=trader_id, limit=limit, offset=offset)
        return [r.to_dict() for r in records]
    except Exception as e:
        logger.error("Failed to fetch strategy decisions: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        store.close()


def _init_mock_decisions():
    """Disabled mock decisions completely."""
    pass
# ── Chat WebSocket Server ──

@app.websocket("/api/chat/{session_id}/ws")
async def chat_websocket(websocket: WebSocket, session_id: str):
    """Establishes chat session and handles full streaming conversation loop."""
    await websocket.accept()
    logger.info("WS connection accepted for session: %s", session_id)
    
    prompt_queue = asyncio.Queue()
    
    async def read_from_client():
        try:
            while True:
                data = await websocket.receive_text()
                try:
                    payload = json.loads(data)
                except Exception:
                    payload = {"type": "prompt", "prompt": data, "attachments": []}
                    
                msg_type = payload.get("type", "prompt")
                
                if msg_type == "prompt":
                    prompt = payload.get("prompt") or ""
                    attachments = payload.get("attachments") or []
                    if prompt.strip() or attachments:
                        await prompt_queue.put((prompt, attachments))
                elif msg_type == "approval_response":
                    tool_call_id = payload.get("tool_call_id")
                    approved = payload.get("approved", False)
                    logger.info("Received approval response: ID=%s, approved=%s", tool_call_id, approved)
                    
                    fut = agent.pending_approvals.get(tool_call_id)
                    if fut and not fut.done():
                        fut.set_result(approved)
        except WebSocketDisconnect:
            # Main loop will catch disconnect
            pass
        except Exception as err:
            logger.error("Error in read_from_client: %s", err)

    # Start the client reader task in the background
    reader_task = asyncio.create_task(read_from_client())
    
    try:
        while True:
            try:
                prompt_item = await prompt_queue.get()
            except asyncio.CancelledError:
                break
                
            prompt, attachments = prompt_item
            logger.info("Processing prompt for session %s: %s (attachments: %d)", session_id, prompt, len(attachments))
            
            try:
                async for chunk in agent.generate_stream(session_id, prompt, attachments=attachments):
                    await websocket.send_json(chunk)
                await websocket.send_json({"type": "done"})
            except (WebSocketDisconnect, RuntimeError):
                logger.info("WS connection lost during streaming for session: %s", session_id)
                break
            except Exception as stream_err:
                logger.exception("Error in agent generator stream")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Execution error: {str(stream_err)}"
                    })
                except Exception:
                    pass
    except WebSocketDisconnect:
        logger.info("WS disconnected for session: %s", session_id)
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except Exception:
            pass



# ── Static File Routing ──

# Ensure uploads directory exists
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file to serve as media or text analysis attachment."""
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        file_id = str(uuid.uuid4())
        original_name = file.filename or "file"
        ext = Path(original_name).suffix
        safe_name = f"{file_id}{ext}"
        dest_path = UPLOAD_DIR / safe_name
        
        content = await file.read()
        with open(dest_path, "wb") as f:
            f.write(content)
            
        mime_type = file.content_type or "application/octet-stream"
        major_type = "document"
        if mime_type.startswith("image/"):
            major_type = "image"
        elif mime_type.startswith("video/"):
            major_type = "video"
        elif mime_type.startswith("text/") or ext.lower() in [".txt", ".csv", ".log", ".json", ".py", ".md"]:
            major_type = "text"
            
        return {
            "status": "success",
            "file_id": file_id,
            "name": original_name,
            "type": major_type,
            "mime_type": mime_type,
            "size": len(content),
            "url": f"/api/static/uploads/{safe_name}",
            "path": str(dest_path.resolve())
        }
    except Exception as e:
        logger.exception("Failed to upload file")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/static/uploads/{filename}")
def serve_upload(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Upload file not found")
    return FileResponse(file_path)

# Serves Pillow-generated PNG charts from the /Users/hongjianjia/xmlx_vlm/xmlx_vlm/ai_trader/data directory
@app.get("/api/static/charts/{filename}")
def serve_chart(filename: str):
    chart_path = DATA_DIR / filename
    if not chart_path.exists():
        raise HTTPException(status_code=404, detail="Chart not found")
    return FileResponse(chart_path)


# Setup static directories and base template serving
STATIC_DIR = Path(__file__).parent / "web" / "static"

# Fallback serving of SPA index.html for unknown routes
@app.get("/")
def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return {"status": "error", "message": "Static assets folder 'web/static' is missing. Please create index.html"}
    return FileResponse(index_file, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

# Mount remaining files (CSS, JS) statically
try:
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
except Exception as e:
    logger.warning("Could not mount static directory: %s", e)


def main():
    parser = argparse.ArgumentParser(description="AI Trader Web Interface Platform Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5119, help="Port (default: 5119)")
    parser.add_argument("--server-url", type=str, default=DEFAULT_SERVER_URL, help="xmlx_vlm server URL")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="API Key for inference server")
    parser.add_argument("--local", action="store_true", help="Load MLX model directly in-process")
    parser.add_argument("--model", type=str, default=None, help="Local model path or HF repo ID")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens")
    parser.add_argument("--live", action="store_true", help="Enable live trading execution (requires credentials)")
    parser.add_argument("--exchange", type=str, default="paper", choices=["paper", "hyperliquid"], help="Trading exchange mode")
    parser.add_argument("--risk-profile", type=str, default="conservative", help="OMS Risk profile config")
    parser.add_argument("--dry-run", action="store_true", help="Wind control dry run")
    args = parser.parse_args()

    # Pass command line options to FastAPI state
    app.state.server_url = args.server_url
    app.state.api_key = args.api_key
    app.state.local = args.local
    app.state.model = args.model
    app.state.temperature = args.temperature
    app.state.max_tokens = args.max_tokens
    app.state.live = args.live or (os.environ.get("AI_TRADER_LIVE") == "1")
    app.state.exchange = args.exchange
    app.state.risk_profile = args.risk_profile
    app.state.dry_run = args.dry_run

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Starting AI Trader Web Platform on http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
