"""OMS Live Integration Test.

Place a limit order far from the market price, query it, and then cancel it.
"""

import sys
from pathlib import Path
import os
import asyncio
from decimal import Decimal

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from xmlx_vlm.ai_trader.store.session_db import QuantSessionDB
from xmlx_vlm.ai_trader.store import vault
from xmlx_vlm.ai_trader.oms.core.oms_engine import OMSEngine
import xmlx_vlm.ai_trader.oms.config.settings as settings_mod
from xmlx_vlm.ai_trader.oms.core.order import Order
from xmlx_vlm.ai_trader.oms.constants import OrderSide

async def run_test():
    print("==================================================")
    print("Initializing Database...")
    db = QuantSessionDB()
    
    password = os.environ.get("XMLX_VLM_VAULT_PASSWORD", "xmlx_vlm_default_secure_vault_passphrase_123456!")
    print("Unlocking KMS Vault...")
    salt_hex = db.get_kms_config("vault_salt")
    verifier_hex = db.get_kms_config("vault_verifier")
    if not salt_hex or not verifier_hex:
        print("Error: KMS Vault is not initialized in DB!")
        return
        
    salt = bytes.fromhex(salt_hex)
    derived = vault.derive_key(password, salt)
    if derived.hex() != verifier_hex:
        print("Error: Master password verifier mismatch!")
        return

    encrypted_keys = db.list_kms_keys()
    unlocked_creds = {}
    active_cred = None
    for key_row in encrypted_keys:
        key_id = key_row["key_id"]
        full_row = db.get_encrypted_kms_key(key_id)
        import json
        enc_payload = json.loads(full_row["encrypted_private_key"])
        decrypted_pk = vault.decrypt_data(enc_payload, password)
        cred = {
            "key_id": key_id,
            "label": full_row["label"],
            "wallet_address": full_row["wallet_address"],
            "private_key": decrypted_pk,
            "testnet": bool(full_row["testnet"])
        }
        unlocked_creds[full_row["wallet_address"]] = cred
        if full_row["status"] == "active":
            active_cred = cred

    if not active_cred:
        print("Error: No active KMS key found in the database!")
        return
        
    print(f"Active Key Found: {active_cred['label']}")
    print(f"Wallet Address: {active_cred['wallet_address']}")
    print(f"Testnet: {active_cred['testnet']}")
    print("==================================================")

    # Configure settings
    os.environ["HL_API_WALLET_ADDRESS"] = active_cred["wallet_address"]
    os.environ["HL_API_PRIVATE_KEY"] = active_cred["private_key"]
    os.environ["HL_TESTNET"] = "1" if active_cred["testnet"] else "0"
    os.environ["AI_TRADER_LIVE"] = "1"
    os.environ["AI_TRADER_EXCHANGE"] = "hyperliquid"
    
    settings_mod.reset_settings()
    sett = settings_mod.get_settings(
        live=True,
        exchange="hyperliquid",
    )
    
    # Customize risk profile to allow testing order
    sett.risk_profile = "custom"
    sett.max_slippage_pct = Decimal("50.0")  # Allow larger slippage/deviations for testing
    sett.max_price_deviation_pct = Decimal("95.0")  # Allow placing order far from market price
    sett.min_order_notional = Decimal("1.0")  # Allow small order (minimum limit)
    sett.max_single_order_notional = Decimal("1000.0")


    print("Initializing OMSEngine...")
    oms = OMSEngine(settings=sett)
    
    print("Syncing Account Info...")
    account_snapshot = await oms.adapter.sync_account()
    print(f"Account snapshot: {account_snapshot}")
    print(f"Available margin: {account_snapshot.available_margin} USDC")
    print(f"Account equity: {account_snapshot.equity} USDC")
    print("==================================================")
    
    # Query current quote for BTC/USDC
    symbol = "BTC/USDC"
    print(f"Fetching quote for {symbol}...")
    quote = await oms.adapter.get_quote(symbol)
    if not quote or not quote.last:
        print("Error: Failed to fetch market price!")
        return
    print(f"Current price for {symbol}: {quote.last}")
    
    # Let's get the tick size rules
    # Hyperliquid requires limit price to be rounded to 5 significant figures.
    # We can write a helper function to round to Hyperliquid's rules.
    def round_price(price: float) -> float:
        # Round to 5 significant figures
        from math import log10, floor
        if price == 0:
            return 0.0
        decimals = 5 - int(floor(log10(abs(price)))) - 1
        # Hyperliquid max decimals is 5
        decimals = max(0, min(5, decimals))
        return round(price, decimals)

    raw_target_price = float(quote.last) * 0.5
    target_price = round_price(raw_target_price)
    qty = 0.0004 # Minimum size for BTC perp is 0.0001, but we need notional > $10
    
    print(f"Target Price: {target_price} (raw: {raw_target_price})")
    print(f"Qty: {qty} BTC")
    print("==================================================")
    
    print("Placing BUY limit order...")
    order = oms.create_order(
        symbol=symbol,
        side="buy",
        qty=qty,
        order_type="limit",
        price=target_price,
    )
    
    print("Submitting order to OMS...")
    # Satisfying pre-trade risk check context by passing mark/oracle prices
    res = await oms.submit_order(order, mark_price=quote.last, oracle_price=quote.last)
    print("Submit result:", res)
    
    if res.get("status") != "submitted" or not res.get("ack", {}).get("success"):
        print("Error: Order submission failed or rejected by exchange!")
        return
        
    client_order_id = order.client_order_id
    order_id = order.order_id or res.get("ack", {}).get("order_id")
    print(f"Order successfully submitted! client_order_id: {client_order_id}, order_id: {order_id}")
    print("==================================================")
    
    # Query open orders to verify
    print("Querying open orders from exchange...")
    await asyncio.sleep(2)  # Wait for L1 state updates
    
    open_orders = oms.adapter._client.info({"type": "frontendOpenOrders", "user": active_cred["wallet_address"]})
    print("Active open orders:")
    found = False
    if isinstance(open_orders, list):
        for o in open_orders:
            print(f"  - ID: {o.get('oid')}, Coin: {o.get('coin')}, Price: {o.get('limitPx')}, Sz: {o.get('sz')}, Cloid: {o.get('cloid')}")
            if str(o.get('oid')) == str(order_id) or o.get('cloid') == client_order_id:
                found = True
    else:
        print("  None (or error fetching open orders)")
        
    if found:
        print("Success: Placed order verified in open orders on exchange!")
    else:
        print("Warning: Placed order not found in open orders listing.")
    print("==================================================")
        
    # Cancel the order
    print(f"Cancelling order (cloid: {client_order_id})...")
    cancel_res = await oms.cancel_order(client_order_id)
    print("Cancel result:", cancel_res)
    
    if cancel_res.get("status") == "cancelled" or cancel_res.get("ack", {}).get("success"):
        print("Success: Cancel command executed successfully.")
    else:
        print("Error: Cancel command failed!")
        
    print("==================================================")
    # Query open orders again to verify removal
    print("Querying open orders after cancellation...")
    await asyncio.sleep(2)
    open_orders_after = oms.adapter._client.info({"type": "frontendOpenOrders", "user": active_cred["wallet_address"]})
    still_present = False
    if isinstance(open_orders_after, list):
        for o in open_orders_after:
            if str(o.get('oid')) == str(order_id) or o.get('cloid') == client_order_id:
                still_present = True
                
    if not still_present:
        print("Success: Order confirmed cancelled and removed from exchange open orders!")
    else:
        print("Error: Order is still present in exchange open orders after cancellation!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_test())
