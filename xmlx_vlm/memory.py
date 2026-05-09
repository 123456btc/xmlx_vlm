# SPDX-License-Identifier: Apache-2.0
"""
Lightweight persistent memory store for conversational context.

Uses SQLite for persistence and numpy for in-memory vector similarity.
Leverages the existing EmbeddingEngine and RerankEngine for encoding
and re-ranking.
"""

import json
import logging
import math
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("xmlx_vlm.memory")

_DEFAULT_EMBED_MODEL = "mlx-embeddings/all-MiniLM-L6-v2"
_DEFAULT_RERANK_MODEL = "mlx-community/jina-reranker-v2-base-multilingual"


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class MemoryStore:
    """
    Persistent memory store backed by SQLite.

    Stores conversation snippets with embeddings. Retrieval uses
    cosine-similarity pre-filtering followed by optional cross-encoder
    re-ranking via RerankEngine.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        embed_model: Optional[str] = None,
        rerank_model: Optional[str] = None,
        top_k: int = 5,
    ):
        self.db_path = db_path or os.path.expanduser("~/.cache/xmlx_vlm/memory.db")
        self.embed_model = embed_model or os.environ.get(
            "XMLX_VLM_MEMORY_MODEL", _DEFAULT_EMBED_MODEL
        )
        self.rerank_model = rerank_model or os.environ.get(
            "XMLX_VLM_MEMORY_RERANK_MODEL", _DEFAULT_RERANK_MODEL
        )
        self.top_k = top_k

        self._embed_engine = None
        self._rerank_engine = None
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT,
                    metadata TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id)"
            )
            conn.commit()

    def _get_embed_engine(self):
        if self._embed_engine is None:
            from .embedding_engine import EmbeddingEngine

            self._embed_engine = EmbeddingEngine(self.embed_model)
        return self._embed_engine

    def _get_rerank_engine(self):
        if self._rerank_engine is None:
            from .rerank_engine import RerankEngine

            self._rerank_engine = RerankEngine(self.rerank_model)
        return self._rerank_engine

    def add(
        self,
        session_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a new memory snippet with its embedding."""
        if not content or not content.strip():
            return

        try:
            engine = self._get_embed_engine()
            vectors = engine.embed([content])
            embedding_json = json.dumps(vectors[0])
        except Exception as e:
            logger.warning("Failed to embed memory, storing without embedding: %s", e)
            embedding_json = None

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO memories (session_id, content, embedding, metadata, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        content,
                        embedding_json,
                        json.dumps(metadata) if metadata else None,
                        time.time(),
                    ),
                )
                conn.commit()

    def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant memories for a query.

        Two-stage retrieval:
        1. Cosine similarity over embeddings (pre-filter)
        2. Optional cross-encoder re-ranking for quality
        """
        top_k = top_k or self.top_k
        if not query or not query.strip():
            return []

        # 1. Encode query
        try:
            engine = self._get_embed_engine()
            query_vectors = engine.embed([query])
            query_embedding = query_vectors[0]
        except Exception as e:
            logger.error("Failed to embed query for memory search: %s", e)
            return []

        # 2. Load candidate memories
        candidates: List[Tuple[int, str, List[float], str]] = []
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                if session_id:
                    rows = conn.execute(
                        "SELECT id, content, embedding FROM memories WHERE session_id = ? AND embedding IS NOT NULL",
                        (session_id,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT id, content, embedding FROM memories WHERE embedding IS NOT NULL"
                    ).fetchall()

        for row in rows:
            mem_id, content, emb_json = row
            try:
                emb = json.loads(emb_json)
                candidates.append((mem_id, content, emb, emb_json))
            except Exception:
                continue

        if not candidates:
            return []

        # 3. Cosine similarity pre-filtering (top 20)
        scored = [
            (mem_id, content, _cosine_similarity(query_embedding, emb))
            for mem_id, content, emb, _ in candidates
        ]
        scored.sort(key=lambda x: x[2], reverse=True)
        prefiltered = scored[: max(top_k * 4, 20)]

        if not prefiltered:
            return []

        # 4. Optional cross-encoder re-ranking
        docs = [content for _, content, _ in prefiltered]
        try:
            reranker = self._get_rerank_engine()
            scores, _ = reranker.score_pairs(query, docs)
            reranked = sorted(
                zip(prefiltered, scores),
                key=lambda x: x[1],
                reverse=True,
            )
            final = [
                {"id": item[0][0], "content": item[0][1], "score": float(item[1])}
                for item in reranked[:top_k]
            ]
        except Exception as e:
            logger.warning("Rerank failed, falling back to embedding scores: %s", e)
            final = [
                {"id": item[0], "content": item[1], "score": float(item[2])}
                for item in prefiltered[:top_k]
            ]

        return final

    def clear(self, session_id: Optional[str] = None) -> int:
        """Delete memories. Returns number of rows deleted."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                if session_id:
                    cur = conn.execute(
                        "DELETE FROM memories WHERE session_id = ?", (session_id,)
                    )
                else:
                    cur = conn.execute("DELETE FROM memories")
                conn.commit()
                return cur.rowcount

    def stats(self) -> Dict[str, Any]:
        """Return store statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM memories"
            ).fetchone()[0]
            sessions = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM memories"
            ).fetchone()[0]
        return {
            "total_memories": total,
            "distinct_sessions": sessions,
            "db_path": self.db_path,
            "embed_model": self.embed_model,
            "rerank_model": self.rerank_model,
        }


# Global singleton
_global_store: Optional[MemoryStore] = None


def get_memory_store() -> Optional[MemoryStore]:
    """Get or create the global memory store if enabled."""
    global _global_store
    if _global_store is not None:
        return _global_store

    enabled = os.environ.get("XMLX_VLM_MEMORY_ENABLED", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    if not enabled:
        return None

    db_path = os.environ.get("XMLX_VLM_MEMORY_DB_PATH")
    top_k = int(os.environ.get("XMLX_VLM_MEMORY_TOP_K", "5"))
    _global_store = MemoryStore(
        db_path=db_path,
        top_k=top_k,
    )
    logger.info(
        "Memory store enabled (model=%s, rerank=%s, top_k=%d)",
        _global_store.embed_model,
        _global_store.rerank_model,
        top_k,
    )
    return _global_store


def set_memory_store(store: Optional[MemoryStore]) -> None:
    global _global_store
    _global_store = store
