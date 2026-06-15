import unittest
from decimal import Decimal
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from xmlx_vlm.ai_trader.agent.evaluator import LLMSignalEvaluator
from xmlx_vlm.ai_trader.agent.config import AgentObjective
from xmlx_vlm.ai_trader.market_service.events import IndicatorAlertEvent
from xmlx_vlm.ai_trader.store.session_db import QuantSessionDB

class TestLLMSignalEvaluator(unittest.TestCase):

    def setUp(self):
        # Create a temp directory for DB testing
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = self.temp_dir / "test_trader_sessions.db"
        self.db = QuantSessionDB(self.db_path)
        self.objective = AgentObjective()
        self.evaluator = LLMSignalEvaluator(
            objective=self.objective,
            db=self.db,
            server_url="http://localhost:5118",
            api_key="test-key",
            model_name="test-model",
            use_fallback=True
        )

    def tearDown(self):
        # Clean up temp files
        shutil.rmtree(self.temp_dir)

    def test_db_reflections_add_and_get(self):
        # 1. Test database log reflection
        details = {"entry": 60000.0, "exit": 59500.0}
        reflection = self.db.add_reflection(
            symbol="BTC/USDC",
            pnl=-50.0,
            trade_details=details,
            lesson="Avoid long positions in high volatility book imbalance."
        )
        self.assertEqual(reflection["symbol"], "BTC/USDC")
        self.assertEqual(reflection["pnl"], -50.0)
        self.assertEqual(reflection["lesson"], "Avoid long positions in high volatility book imbalance.")

        # 2. Test database retrieve recent reflections
        reflections = self.db.get_recent_reflections(limit=5)
        self.assertEqual(len(reflections), 1)
        self.assertEqual(reflections[0]["symbol"], "BTC/USDC")
        self.assertEqual(reflections[0]["pnl"], -50.0)
        self.assertEqual(reflections[0]["lesson"], "Avoid long positions in high volatility book imbalance.")
        self.assertEqual(reflections[0]["trade_details"], details)

    def test_parse_json_response(self):
        # 1. Test parsing pure JSON
        raw_json = '{"debate": {"bull_case": "b1", "bear_case": "b2", "rebuttal": "r1"}, "decision": {"direction": "long", "confidence": 80}}'
        data = self.evaluator._parse_json_response(raw_json)
        self.assertIsNotNone(data)
        self.assertEqual(data["decision"]["direction"], "long")
        self.assertEqual(data["decision"]["confidence"], 80)

        # 2. Test parsing markdown code block
        markdown_json = 'Some text here\n```json\n{"debate": {"bull_case": "b1", "bear_case": "b2", "rebuttal": "r1"}, "decision": {"direction": "short", "confidence": 70}}\n```\nOther text.'
        data = self.evaluator._parse_json_response(markdown_json)
        self.assertIsNotNone(data)
        self.assertEqual(data["decision"]["direction"], "short")
        self.assertEqual(data["decision"]["confidence"], 70)

        # 3. Test parsing generic code block
        generic_code = 'Text\n```\n{"debate": {"bull_case": "b1", "bear_case": "b2", "rebuttal": "r1"}, "decision": {"direction": "neutral", "confidence": 50}}\n```'
        data = self.evaluator._parse_json_response(generic_code)
        self.assertIsNotNone(data)
        self.assertEqual(data["decision"]["direction"], "neutral")

        # 4. Test fallback to first/last braces
        braces_text = 'Check output: {"debate": {"bull_case": "b1", "bear_case": "b2", "rebuttal": "r1"}, "decision": {"direction": "long", "confidence": 90}}.'
        data = self.evaluator._parse_json_response(braces_text)
        self.assertIsNotNone(data)
        self.assertEqual(data["decision"]["confidence"], 90)

    @patch("requests.post")
    def test_evaluate_success(self, mock_post):
        # Mock successful response from local model server
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"debate": {"bull_case": "bull text", "bear_case": "bear text", "rebuttal": "rebuttal text"}, "decision": {"direction": "long", "confidence": 85, "stop_loss": 60000.0, "take_profit": 63000.0, "rationale": "Strong bull pattern"}}\n```'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        # Add a past reflection to DB to ensure prompt formatting works
        self.db.add_reflection("BTC/USDC", 10.0, {}, "Trend is friend.")

        event = IndicatorAlertEvent(symbol="BTC/USDC", timestamp_ms=1, alert_type="breakout", payload={"volume_confirmed": True})
        eval_res = self.evaluator.evaluate(
            event=event,
            mark_price=Decimal("61000.0"),
            atr=Decimal("500.0"),
            portfolio_summary={"account": {"equity": "10000.0"}}
        )

        self.assertEqual(eval_res.symbol, "BTC/USDC")
        self.assertEqual(eval_res.confidence, 85)
        self.assertEqual(eval_res.stop_loss, Decimal("60000.0"))
        self.assertEqual(eval_res.take_profit, Decimal("63000.0"))
        self.assertEqual(eval_res.metadata["direction"], "long")
        self.assertEqual(eval_res.metadata["rationale"], "Strong bull pattern")
        self.assertIn("LLM Consensus: LONG (Confidence: 85)", eval_res.notes[0])

    @patch("requests.post")
    def test_evaluate_failure_fallback(self, mock_post):
        # Mock connection failure (raises exception)
        mock_post.side_effect = Exception("Connection refused")

        # Verify evaluate falls back to rule-based SignalEvaluator
        event = IndicatorAlertEvent(symbol="BTC/USDC", timestamp_ms=1, alert_type="breakout", payload={"direction": "long", "volume_confirmed": True})
        eval_res = self.evaluator.evaluate(
            event=event,
            mark_price=Decimal("61000.0"),
            atr=Decimal("500.0"),
            portfolio_summary={"account": {"equity": "10000.0"}}
        )

        # SignalEvaluator returns evaluation based on rules
        self.assertEqual(eval_res.symbol, "BTC/USDC")
        self.assertEqual(eval_res.metadata["direction"], "long")
        # Ensure fallback indicators are present (RSI/EMA/ATR defaults)
        self.assertTrue(eval_res.confidence > 50)

    @patch("requests.post")
    @patch("asyncio.sleep", return_value=None)
    def test_run_post_trade_reflection(self, mock_sleep, mock_post):
        from xmlx_vlm.ai_trader.agent.loop import AutonomousAgentLoop
        from xmlx_vlm.ai_trader.agent.decision import TradeProposal, ActionType
        import asyncio

        # Mock OMS and portfolio
        mock_oms = MagicMock()
        mock_position = MagicMock()
        mock_position.realized_pnl = Decimal("150.0")
        mock_oms.portfolio.get_position.return_value = mock_position

        # Mock response from LLM server for reflection
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"lesson": "Be careful when booking imbalance is skewed."}\n```'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        # Instantiate AutonomousAgentLoop
        loop = AutonomousAgentLoop(
            oms=mock_oms,
            objective=self.objective,
            evaluator=self.evaluator
        )

        proposal = TradeProposal(
            action=ActionType.CLOSE_LONG,
            symbol="BTC/USDC",
            size_usd=Decimal("1000.0"),
            reason="closing out position"
        )

        # Run reflection synchronously
        asyncio.run(loop._run_post_trade_reflection(proposal, {}))

        # Check that reflection was added to the DB
        reflections = self.db.get_recent_reflections(limit=1)
        self.assertEqual(len(reflections), 1)
        self.assertEqual(reflections[0]["symbol"], "BTC/USDC")
        self.assertEqual(reflections[0]["pnl"], 150.0)
        self.assertEqual(reflections[0]["lesson"], "Be careful when booking imbalance is skewed.")
