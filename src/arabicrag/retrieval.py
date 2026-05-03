"""
Retrieval logic.

Provides a lightweight in-memory `VectorIndex` for benchmarking and a
hybrid retriever that combines dense (embedding) and sparse (BM25)
scores. For production scale you'd swap the in-memory index for
Oracle 26ai vector search via the `oracle_vector` adapter (stub
included as a reference).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .chunking import Chunk
from .embeddings import BM25Retriever, Embedder


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorIndex:
    """Simple in-memory cosine-similarity index.

    Fine for benchmarking up to ~100k chunks. Beyond that, swap for
    Oracle 26ai vector search or another vector DB.
    """

    def __init__(self, embedder: Embedder):
        self._embedder = embedder
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        texts = [c.text for c in self._chunks]
        if hasattr(self._embedder, "fit"):
            self._embedder.fit(texts)
        vectors = self._embedder.embed(texts)
        self._matrix = _l2_normalize(vectors)

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        if self._matrix is None:
            raise RuntimeError("Call .index(chunks) before .search(query).")
        q = self._embedder.embed([query])
        q = _l2_normalize(q)
        sims = (self._matrix @ q.T).ravel()
        top_idx = np.argsort(-sims)[:k]
        return [RetrievalResult(self._chunks[i], float(sims[i])) for i in top_idx]


class HybridRetriever:
    """Combines dense vector and BM25 sparse scoring.

    Score is `alpha * vector_score + (1 - alpha) * bm25_score`,
    after normalizing each to [0, 1] using min-max over the
    candidate pool. This is the production-friendly default.
    """

    def __init__(self, vector_index: VectorIndex, alpha: float = 0.6):
        self._vec = vector_index
        self._bm25 = BM25Retriever()
        self._alpha = alpha
        self._chunks: list[Chunk] = []

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        self._vec.index(chunks)
        self._bm25.fit([c.text for c in chunks])

    def search(self, query: str, k: int = 5, candidate_pool: int = 50) -> list[RetrievalResult]:
        # Get a wider pool from each retriever, then merge
        vec_results = self._vec.search(query, k=candidate_pool)
        bm25_scores = self._bm25.score(query)

        # Build score maps over chunk_id
        vec_map = {r.chunk.chunk_id: r.score for r in vec_results}
        bm25_map = {self._chunks[i].chunk_id: float(s) for i, s in enumerate(bm25_scores)}

        candidates = set(vec_map) | set(bm25_map)

        def normalize(scores: dict, keys: set) -> dict:
            vals = [scores.get(k, 0.0) for k in keys]
            lo, hi = min(vals), max(vals)
            span = hi - lo or 1.0
            return {k: (scores.get(k, 0.0) - lo) / span for k in keys}

        vec_norm = normalize(vec_map, candidates)
        bm25_norm = normalize(bm25_map, candidates)

        merged = {
            cid: self._alpha * vec_norm[cid] + (1 - self._alpha) * bm25_norm[cid]
            for cid in candidates
        }
        chunk_by_id = {c.chunk_id: c for c in self._chunks}
        ranked = sorted(merged.items(), key=lambda kv: -kv[1])[:k]
        return [RetrievalResult(chunk_by_id[cid], score) for cid, score in ranked]


# ---------------------------------------------------------------------------
# Oracle 26ai vector search adapter (reference stub)
# ---------------------------------------------------------------------------
#
# In production, you'd replace VectorIndex with Oracle 26ai's native
# vector search using the VECTOR data type and DBMS_VECTOR. The shape
# of the calls roughly looks like:
#
#     CREATE TABLE chunks (
#         chunk_id  VARCHAR2(200) PRIMARY KEY,
#         doc_id    VARCHAR2(200),
#         section   VARCHAR2(200),
#         text      CLOB,
#         embedding VECTOR(1024, FLOAT32)
#     );
#
#     -- Search:
#     SELECT chunk_id, text,
#            VECTOR_DISTANCE(embedding, :query_vec, COSINE) AS distance
#       FROM chunks
#       ORDER BY distance
#       FETCH FIRST :k ROWS ONLY;
#
# A working OracleVectorIndex adapter using oracledb is left as Week 4
# of the build plan, once you have Autonomous Database 26ai provisioned.
