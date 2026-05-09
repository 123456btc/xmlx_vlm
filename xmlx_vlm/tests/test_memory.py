# SPDX-License-Identifier: Apache-2.0
"""Tests for the lightweight memory store."""

import os
import tempfile
import unittest

from xmlx_vlm.memory import MemoryStore


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.environ["XMLX_VLM_MEMORY_ENABLED"] = "false"  # disable global singleton
        self.store = MemoryStore(
            db_path=self.db_path,
            embed_model="mlx-embeddings/all-MiniLM-L6-v2",
            rerank_model="mlx-community/jina-reranker-v2-base-multilingual",
            top_k=3,
        )

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_add_and_stats(self):
        self.store.add("sess1", "I love machine learning")
        self.store.add("sess1", "Python is great")
        self.store.add("sess2", "Go is fast")

        stats = self.store.stats()
        self.assertEqual(stats["total_memories"], 3)
        self.assertEqual(stats["distinct_sessions"], 2)

    def test_clear_session(self):
        self.store.add("sess1", "A")
        self.store.add("sess2", "B")
        deleted = self.store.clear(session_id="sess1")
        self.assertEqual(deleted, 1)
        self.assertEqual(self.store.stats()["total_memories"], 1)

    def test_clear_all(self):
        self.store.add("sess1", "A")
        self.store.add("sess2", "B")
        deleted = self.store.clear()
        self.assertEqual(deleted, 2)
        self.assertEqual(self.store.stats()["total_memories"], 0)

    def test_empty_search(self):
        results = self.store.search("")
        self.assertEqual(results, [])

    def test_cosine_similarity(self):
        from xmlx_vlm.memory import _cosine_similarity

        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(_cosine_similarity(a, b), 1.0)

        c = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(_cosine_similarity(a, c), 0.0)

    def test_skip_empty_content(self):
        self.store.add("sess1", "")
        self.store.add("sess1", "   ")
        self.assertEqual(self.store.stats()["total_memories"], 0)


if __name__ == "__main__":
    unittest.main()
