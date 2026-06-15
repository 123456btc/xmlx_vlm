"""Test script for AITraderAgent and QuantSessionDB."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from xmlx_vlm.ai_trader.store.session_db import QuantSessionDB
from xmlx_vlm.ai_trader.agent.agent_loop import AITraderAgent


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_sessions.db"
        self.db = QuantSessionDB(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_database_creation(self):
        # Verify tables exist and can insert sessions
        session = self.db.create_session("test_sess_1", "Test Chat", "mock-model", "paper")
        self.assertEqual(session["session_id"], "test_sess_1")
        self.assertEqual(session["mode"], "paper")

        loaded = self.db.get_session("test_sess_1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["title"], "Test Chat")

    def test_message_history(self):
        self.db.create_session("sess_2", "Sess 2", "mock-model", "paper")
        self.db.add_message("msg_1", "sess_2", "user", "Hello")
        self.db.add_message("msg_2", "sess_2", "assistant", "Hi there")

        msgs = self.db.get_messages("sess_2")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["content"], "Hello")
        self.assertEqual(msgs[1]["role"], "assistant")

    def test_trade_logs(self):
        self.db.create_session("sess_3", "Sess 3", "mock-model", "paper")
        self.db.log_trade(
            trade_id="t1",
            session_id="sess_3",
            symbol="BTC/USDC",
            side="buy",
            qty=0.5,
            price=65000.0,
            pnl=0.0,
            status="simulated",
        )

        trades = self.db.get_trades("sess_3")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["symbol"], "BTC/USDC")
        self.assertEqual(trades[0]["side"], "buy")

    def test_title_summarization_trigger(self):
        # We want to test that AITraderAgent.generate_stream yields title_update on first turn
        agent = AITraderAgent(db=self.db, server_url="http://localhost:5118", local_only=True)
        
        # Mock _summarize_session_title to return a mocked summary title
        async def mock_summarize(user_input):
            return "Mocked Summary Title"
        agent._summarize_session_title = mock_summarize
        
        # Mock _stream_from_local to yield a simple response
        async def mock_stream_from_local(db_messages):
            yield "content", "Hello user!"
        agent._stream_from_local = mock_stream_from_local
        agent.model_path = "mock-model"
        
        # Run generate_stream and collect yielded events
        events = []
        async def run_stream():
            async for chunk in agent.generate_stream("sess_4", "What is BTC price?"):
                events.append(chunk)
        
        asyncio.run(run_stream())
        
        # Verify that the title was updated in the DB
        session = self.db.get_session("sess_4")
        self.assertEqual(session["title"], "Mocked Summary Title")
        
        # Verify title_update was yielded
        title_events = [e for e in events if e.get("type") == "title_update"]
        self.assertEqual(len(title_events), 1)
        self.assertEqual(title_events[0]["title"], "Mocked Summary Title")
        self.assertEqual(title_events[0]["session_id"], "sess_4")

    def test_multimodal_message_handling(self):
        # 1. Test _build_history_for_openai with image and video paths
        agent = AITraderAgent(db=self.db, server_url="http://localhost:5118", local_only=True)
        
        mock_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Check this out"},
                    {"type": "image_url", "image_url": {"url": "/api/static/uploads/1.png"}, "path": "nonexistent_img.png"},
                    {"type": "video_url", "video_url": {"url": "/api/static/uploads/2.mp4"}, "path": "nonexistent_vid.mp4"}
                ]
            }
        ]
        
        history = agent._build_history_for_openai(mock_messages)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["role"], "user")
        self.assertTrue(isinstance(history[0]["content"], list))
        
        # 2. Test text attachment parsing in generate_stream
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Bitcoin is decentralized digital currency")
            temp_txt_path = f.name
            
        try:
            # Mock _stream_from_local to prevent actual execution
            async def mock_stream_from_local(db_messages):
                yield "content", "Understood data"
            agent._stream_from_local = mock_stream_from_local
            
            async def mock_summarize(user_input):
                return "Text Analysis"
            agent._summarize_session_title = mock_summarize
            agent.model_path = "mock-model"
            
            attachments = [
                {
                    "type": "text",
                    "name": "data.txt",
                    "path": temp_txt_path,
                    "url": "/api/static/uploads/data.txt",
                    "mime_type": "text/plain",
                    "size": 41
                }
            ]
            
            events = []
            async def run_stream():
                async for chunk in agent.generate_stream("sess_multimodal", "Read this text file", attachments=attachments):
                    events.append(chunk)
            
            asyncio.run(run_stream())
            
            # Verify the stored message content contains the text file's contents inline
            db_msgs = self.db.get_messages("sess_multimodal")
            user_msg = next(m for m in db_msgs if m["role"] == "user")
            content_list = user_msg["content"]
            
            self.assertTrue(any("Bitcoin is decentralized digital currency" in part.get("text", "") 
                                for part in content_list if part.get("type") == "text"))
            
        finally:
            import os
            try:
                os.remove(temp_txt_path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
