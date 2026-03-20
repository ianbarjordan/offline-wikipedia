"""
retriever.py

Loads the FAISS index, id_map, and SQLite database once at startup.
Exposes a single public method:

    search(query: str, top_k: int = config.TOP_K) -> list[dict]

Each returned dict has keys: id, title, lead, url_slug.

Design notes
------------
- Embeddings are L2-normalised at build time (normalize_embeddings=True in
  script 04).  Query vectors are normalised here with faiss.normalize_L2()
  before searching, making inner product equivalent to cosine similarity.
- index.nprobe is set at load time (not per-query) because it is a property
  of the index object and does not change between searches.
- SQLite is opened with check_same_thread=False so Gradio's threaded request
  handling can share the same connection safely.
- FAISS may return index value -1 when it cannot fill all top_k slots (e.g.
  nprobe cells contain fewer than top_k vectors).  These are filtered out.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config

# ---------------------------------------------------------------------------
# Title-boost rerank helper
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "of", "in", "to", "and", "or",
    "what", "how", "why", "who", "when", "where", "was", "were",
})


def _title_rerank(query: str, articles: list[dict]) -> list[dict]:
    """
    Boost articles whose title words overlap with the query.

    A perfect title match is worth config.TITLE_BOOST rank positions.
    FAISS rank order is preserved for articles with no title overlap.
    Common stopwords and short tokens are excluded from the query word set
    so that articles with semantically empty query terms aren't over-promoted.
    """
    if not articles:
        return articles
    q_words = {
        w for w in query.lower().split()
        if w not in _STOPWORDS and len(w) > 2
    }
    if not q_words:
        return articles

    def sort_key(item: tuple[int, dict]) -> float:
        rank, art = item
        t_words = set(art["title"].lower().split())
        overlap = len(q_words & t_words) / len(q_words)
        return rank - config.TITLE_BOOST * overlap  # lower value = better rank

    reranked = sorted(enumerate(articles), key=sort_key)
    return [art for _, art in reranked]


class Retriever:
    """
    Wraps the FAISS index, id_map, SQLite DB, and embedding model.
    Intended to be instantiated once and reused for the lifetime of the app.
    """

    def __init__(
        self,
        faiss_path: Optional[Path] = None,
        id_map_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        embed_model: Optional[str] = None,
    ) -> None:
        """
        Load all retrieval components.  Paths default to config values;
        pass alternatives for testing.

        Raises FileNotFoundError if any required file is missing.
        """
        faiss_path = faiss_path or config.FAISS_PATH
        id_map_path = id_map_path or config.ID_MAP_PATH
        db_path = db_path or config.DB_PATH
        model_name = embed_model or config.EMBED_MODEL

        # --- FAISS index ---
        if not faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {faiss_path}\n"
                "Run build/04_embed_and_index.py to generate it."
            )
        self.index: faiss.Index = faiss.read_index(str(faiss_path))

        # Set nprobe now, once.  Silently ignored for flat indexes (no attribute).
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = config.NPROBE

        # --- id_map: {str(faiss_position) -> sqlite_id} ---
        if not id_map_path.exists():
            raise FileNotFoundError(
                f"id_map not found: {id_map_path}\n"
                "Run build/04_embed_and_index.py to generate it."
            )
        raw_map: dict[str, int] = json.loads(id_map_path.read_text(encoding="utf-8"))
        # Convert keys to int for O(1) lookup without str() on every query.
        self._id_map: dict[int, int] = {int(k): v for k, v in raw_map.items()}

        # --- SQLite ---
        if not db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {db_path}\n"
                "Run build/03_build_sqlite.py to generate it."
            )
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(db_path),
            check_same_thread=False,  # safe for Gradio's threaded handlers
        )
        self._conn.row_factory = sqlite3.Row  # access columns by name

        # --- Embedding model (CPU for end-user machines) ---
        self._model: SentenceTransformer = SentenceTransformer(
            model_name, device="cpu"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = config.TOP_K) -> list[dict]:
        """
        Embed *query*, find the top_k nearest article leads, and return
        the corresponding article records from SQLite.

        Parameters
        ----------
        query   : The user's question or search string.
        top_k   : Number of results to return (defaults to config.TOP_K).

        Returns
        -------
        List of dicts, each with keys: id, title, lead, url_slug.
        Ordered by FAISS similarity score (best first).
        May contain fewer than top_k items if the index is small.
        """
        if not query or not query.strip():
            return []

        # 1. Embed — shape (1, EMBEDDING_DIM), float32
        vec: np.ndarray = self._model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalise via sentence-transformers
        ).astype(np.float32)

        # 2. Explicit L2 normalisation as a defensive guarantee.
        #    faiss.normalize_L2 operates in-place on a contiguous array.
        vec = np.ascontiguousarray(vec)
        faiss.normalize_L2(vec)

        # 3. ANN search — returns (distances, indices) each shape (1, top_k)
        #    Index uses METRIC_INNER_PRODUCT; for L2-normalised vectors the
        #    returned distances are inner products, which equal cosine
        #    similarity directly (range 0.0 → 1.0, higher = more similar).
        distances, faiss_indices = self.index.search(vec, top_k)
        raw_indices: list[int] = faiss_indices[0].tolist()
        raw_distances: list[float] = distances[0].tolist()

        # 4. Map FAISS positions → SQLite article IDs, drop -1 sentinels.
        #    Keep cosine similarity score paired with each valid id.
        valid_pairs: list[tuple[int, float]] = [
            (self._id_map[fi], max(0.0, d))
            for fi, d in zip(raw_indices, raw_distances)
            if fi != -1 and fi in self._id_map
        ]
        if not valid_pairs:
            return []

        sqlite_ids: list[int] = [sid for sid, _ in valid_pairs]
        score_map: dict[int, float] = {sid: score for sid, score in valid_pairs}

        # 5. Fetch article records from SQLite (preserve FAISS rank order)
        placeholders = ",".join("?" * len(sqlite_ids))
        rows = self._conn.execute(
            f"SELECT id, title, lead, url_slug FROM articles WHERE id IN ({placeholders})",
            sqlite_ids,
        ).fetchall()

        # Re-order to match FAISS rank (SQL IN does not guarantee order)
        rank: dict[int, int] = {sid: pos for pos, sid in enumerate(sqlite_ids)}
        rows_sorted = sorted(rows, key=lambda r: rank.get(r["id"], 999))

        return _title_rerank(query, [
            {
                "id": r["id"],
                "title": r["title"],
                "lead": r["lead"],
                "url_slug": r["url_slug"],
                "score": score_map.get(r["id"], 0.0),
            }
            for r in rows_sorted
        ])

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the SQLite connection. Call on app shutdown."""
        try:
            self._conn.close()
        except Exception:
            pass

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *_) -> None:
        self.close()
