"""Unit tests for FactorMiningTool, ToolRegistry integration, TraderSDK.research, and Web Server APIs."""

import json
import pytest
from fastapi.testclient import TestClient

from xmlx_vlm.ai_trader.tools.research import FactorMiningTool
from xmlx_vlm.ai_trader.tools.registry import ToolRegistry
from xmlx_vlm.ai_trader.sdk.client import TraderSDK, ResearchSDK
from xmlx_vlm.ai_trader.web_server import app


class TestFactorMiningToolIntegration:
    """Test suite for AI Trading OS integration of the self-evolving research engine."""

    def test_factor_mining_tool_evolve(self):
        tool = FactorMiningTool()
        output = tool.execute(
            action="evolve_factors",
            symbol="BTC/USDT",
            timeframe="1h",
            generations=2,
            population_size=15,
        )
        assert "自进化 Alpha 因子挖掘报告" in output
        assert "Rank IC" in output
        assert "经验记忆池统计" in output

    def test_factor_mining_tool_memory_and_diagnosis(self):
        tool = FactorMiningTool()

        # 1. Memory summary
        summary_raw = tool.execute(action="get_memory_summary")
        summary_json = json.loads(summary_raw)
        assert summary_json["status"] == "success"
        assert "total_success_factors" in summary_json["summary"]

        # 2. Top factors
        top_raw = tool.execute(action="get_top_factors", top_n=3)
        top_json = json.loads(top_raw)
        assert isinstance(top_json, list)

        # 3. Diagnose factor
        diag_raw = tool.execute(
            action="diagnose_factor",
            formula="ts_delta(close, 5)",
            rank_ic=0.06,
            ir=0.20,
        )
        diag_json = json.loads(diag_raw)
        assert "refined_formula" in diag_json
        assert "diagnosis" in diag_json

    def test_tool_registry_contains_factor_mining(self):
        registry = ToolRegistry()
        tool = registry.get_tool("factor_mining")
        assert tool is not None
        assert tool.name == "factor_mining"
        schema_list = registry.list_tools()
        names = [t["function"]["name"] for t in schema_list]
        assert "factor_mining" in names

    def test_trader_sdk_research(self):
        sdk = TraderSDK()
        assert hasattr(sdk, "research")
        assert isinstance(sdk.research, ResearchSDK)

        # Diagnose via SDK
        diag_res = sdk.research.diagnose_factor("ratio_diff(high, low)")
        assert "refined_formula" in diag_res

        # Memory summary via SDK
        mem_summary = sdk.research.get_memory_summary()
        assert "summary" in mem_summary

    def test_web_server_research_endpoints(self):
        client = TestClient(app)

        # 1. GET /api/research/memory
        res_mem = client.get("/api/research/memory")
        assert res_mem.status_code == 200
        data_mem = res_mem.json()
        assert data_mem["status"] == "success"

        # 2. GET /api/research/top_factors
        res_top = client.get("/api/research/top_factors?top_n=3")
        assert res_top.status_code == 200
        assert isinstance(res_top.json(), list)

        # 3. POST /api/research/diagnose
        res_diag = client.post("/api/research/diagnose", json={
            "formula": "ts_mean(close, 10)",
            "rank_ic": 0.05,
            "ir": 0.22,
        })
        assert res_diag.status_code == 200
        data_diag = res_diag.json()
        assert "refined_formula" in data_diag

        # 4. POST /api/research/evolve
        res_evolve = client.post("/api/research/evolve", json={
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "generations": 1,
            "population_size": 10,
        })
        assert res_evolve.status_code == 200
        data_evolve = res_evolve.json()
        assert data_evolve["status"] == "success"
        assert "report_markdown" in data_evolve
