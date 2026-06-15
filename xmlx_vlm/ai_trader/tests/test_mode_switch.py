import pytest
from xmlx_vlm.ai_trader.store.session_db import QuantSessionDB
from xmlx_vlm.ai_trader.agent.agent_loop import AITraderAgent
import xmlx_vlm.ai_trader.web_server as web_server
from xmlx_vlm.ai_trader.oms.execution.factory import ExecutionAdapterFactory

def test_dynamic_oms_mode_switching(tmp_path, monkeypatch):
    # Setup temporary database
    db_file = tmp_path / "test_sessions.db"
    db = QuantSessionDB(db_path=db_file)
    
    # Mock ExecutionAdapterFactory.create to avoid eth-account dependency
    class MockHyperliquidAdapter:
        @property
        def name(self) -> str:
            return "hyperliquid"

        @property
        def is_live(self) -> bool:
            return True

        def __init__(self, **kwargs):
            self._wallet_address = kwargs.get("wallet_address")
            self.testnet = kwargs.get("testnet", True)
            
        async def submit(self, order):
            return None
            
        async def cancel(self, order_id, client_order_id=None):
            return None
            
        async def query_order(self, order_id):
            return None
            
        async def sync_positions(self):
            return {}
            
        async def sync_account(self):
            return None
            
        def sync(self):
            return {}
            
        def portfolio_summary(self):
            return {"positions": []}
            
    original_create = ExecutionAdapterFactory.create
    def mock_create(exchange="local", market_data_tool=None, **kwargs):
        if exchange.lower() == "hyperliquid":
            return MockHyperliquidAdapter(**kwargs)
        return original_create(exchange, market_data_tool, **kwargs)
        
    monkeypatch.setattr(ExecutionAdapterFactory, "create", mock_create)
    
    # Initialize AITraderAgent
    agent = AITraderAgent(db=db, live=False, exchange="local")
    
    # Mock global agent in web_server module
    web_server.db = db
    web_server.agent = agent
    
    # Assert initial local state
    assert not agent.live
    assert agent.exchange == "local"
    
    # Perform mock activation
    mock_cred = {
        "wallet_address": "0x1234567890abcdef1234567890abcdef12345678",
        "private_key": "0x" + "a" * 64,
        "testnet": True
    }
    
    # Activate key in OMS
    web_server._activate_key_in_oms(mock_cred)
    
    # Verify agent properties updated dynamically
    assert agent.live
    assert agent.exchange == "hyperliquid"
    
    # Check trading tool OMS settings
    trading_tool = agent.registry.get_tool("trading")
    assert trading_tool.oms.is_live
    assert trading_tool.oms.settings.live_enabled
    assert trading_tool.oms.settings.exchange == "hyperliquid"
    assert trading_tool.oms.settings.wallet_address == mock_cred["wallet_address"]
    
    # Deactivate keys in OMS
    web_server._deactivate_key_in_oms()
    
    # Verify agent properties reset dynamically
    assert not agent.live
    assert agent.exchange == "local"
    assert not trading_tool.oms.is_live
    assert not trading_tool.oms.settings.live_enabled
    assert trading_tool.oms.settings.exchange == "local"
