import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from xmlx_vlm.ai_trader.store import vault
import xmlx_vlm.ai_trader.web_server as web_server

# Mock HyperliquidClient
class MockHyperliquidClient:
    def __init__(self, wallet_address, testnet=True):
        self.wallet_address = wallet_address
        self.testnet = testnet
        
    def info(self, query):
        if query["type"] == "clearinghouseState":
            return {
                "marginSummary": {
                    "accountValue": "12345.67",
                    "totalMarginUsed": "500.0"
                },
                "withdrawable": "11845.67",
                "assetPositions": [
                    {
                        "position": {
                            "coin": "ETH",
                            "szi": "2.5",
                            "entryPx": "2000.0",
                            "unrealizedPnl": "250.0",
                            "leverage": {"value": 5, "type": "cross"},
                            "liquidationPx": "1600.0"
                        }
                    }
                ]
            }
        elif query["type"] == "frontendOpenOrders":
            return [
                {
                    "coin": "ETH",
                    "side": "B",
                    "limitPx": "1900.0",
                    "sz": "1.0",
                    "oid": 12345,
                    "cloid": "abc",
                    "timestamp": 1600000000000
                }
            ]
        elif query["type"] == "userFills":
            return [
                {
                    "coin": "ETH",
                    "side": "B",
                    "px": "1950.0",
                    "sz": "0.5",
                    "fee": "0.1",
                    "time": 1600000100000,
                    "tid": 9999,
                    "pnl": "0.0"
                }
            ]
        return {}
        
    def get_spot_clearinghouse_state(self, address):
        return {
            "balances": [
                {
                    "coin": "USDC",
                    "total": "1000.0",
                    "hold": "50.0"
                }
            ]
        }
        
    def close(self):
        pass


@pytest.fixture
def mock_vault_credentials(monkeypatch):
    # Setup test credentials
    cred_a = {
        "wallet_address": "0xAAAAAA1234567890abcdef1234567890abcdef",
        "private_key": "0x" + "a" * 64,
        "testnet": True,
        "label": "Wallet A"
    }
    cred_b = {
        "wallet_address": "0xBBBBBB1234567890abcdef1234567890abcdef",
        "private_key": "0x" + "b" * 64,
        "testnet": False,
        "label": "Wallet B"
    }
    
    # Store decrypted keys in vault secure memory
    vault.wipe_vault()
    vault.set_unlocked_credentials({
        cred_a["wallet_address"]: cred_a,
        cred_b["wallet_address"]: cred_b
    })
    
    # Set Wallet A as default active
    vault.set_active_wallet(cred_a["wallet_address"])
    
    # Mock HyperliquidClient inside the web_server module import path
    import xmlx_vlm.ai_trader.oms.execution.hyperliquid.client as hl_client_mod
    monkeypatch.setattr(hl_client_mod, "HyperliquidClient", MockHyperliquidClient)
    
    yield cred_a, cred_b
    vault.wipe_vault()


def test_get_exchange_assets_with_wallet(mock_vault_credentials):
    cred_a, cred_b = mock_vault_credentials
    client = TestClient(web_server.app)
    
    # 1. Fetching default active wallet (Wallet A)
    resp = client.get("/api/kms/exchange/assets")
    assert resp.status_code == 200
    data = resp.json()
    assert data["account_address"] == cred_a["wallet_address"]
    assert data["label"] == "Wallet A"
    assert data["network"] == "Testnet"
    assert data["perp_equity"] == 12345.67
    
    # 2. Fetching specific wallet (Wallet B)
    resp = client.get(f"/api/kms/exchange/assets?wallet={cred_b['wallet_address']}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["account_address"] == cred_b["wallet_address"]
    assert data["label"] == "Wallet B"
    assert data["network"] == "Mainnet"


def test_get_exchange_positions_with_wallet(mock_vault_credentials):
    cred_a, cred_b = mock_vault_credentials
    client = TestClient(web_server.app)
    
    # 1. Fetch positions for Wallet B
    resp = client.get(f"/api/kms/exchange/positions?wallet={cred_b['wallet_address']}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["symbol"] == "ETH/USDC"
    assert data[0]["qty"] == 2.5
    # Verify mark_price field exists and is calculated
    assert "mark_price" in data[0]
    assert data[0]["mark_price"] == 2100.0  # entryPx (2000.0) + unrealizedPnl (250.0) / szi (2.5) = 2100.0


def test_get_exchange_orders_with_wallet(mock_vault_credentials):
    cred_a, cred_b = mock_vault_credentials
    client = TestClient(web_server.app)
    
    # 1. Fetch orders for Wallet B
    resp = client.get(f"/api/kms/exchange/orders?wallet={cred_b['wallet_address']}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["symbol"] == "ETH/USDC"
    assert data[0]["qty"] == 1.0


def test_get_exchange_history_with_wallet(mock_vault_credentials):
    cred_a, cred_b = mock_vault_credentials
    client = TestClient(web_server.app)
    
    # 1. Fetch history for Wallet B
    resp = client.get(f"/api/kms/exchange/history?wallet={cred_b['wallet_address']}")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["symbol"] == "ETH/USDC"
    assert data[0]["qty"] == 0.5
