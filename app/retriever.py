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
import re
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
    # core function words
    "a", "an", "the", "is", "are", "of", "in", "to", "and", "or",
    "what", "how", "why", "who", "when", "where", "was", "were",
    # preamble verbs and filler words (Fix A)
    "tell", "about", "know", "want", "explain", "describe", "show",
    "give", "find", "need", "like", "make", "use", "say", "let",
    "please", "can", "could", "would", "get", "look", "also",
    "have", "has", "had", "do", "did", "does", "be", "been",
    "more", "some", "just", "really", "very", "think", "see",
})

_WORD_RE = re.compile(r'\b\w+\b')

# Fix 6 — Common nickname → canonical first name expansions.
_NICKNAMES: dict[str, str] = {
    "tom": "thomas", "bob": "robert", "bill": "william", "jim": "james",
    "joe": "joseph", "mike": "michael", "dave": "david", "dick": "richard",
    "jack": "john", "meg": "margaret", "maggie": "margaret", "kate": "katherine",
    "liz": "elizabeth", "beth": "elizabeth",
}

# Fix 7 — US state abbreviation expansion (2-letter token → full state name).
_STATE_ABBREVS: dict[str, str] = {
    "al": "Alabama", "ak": "Alaska", "az": "Arizona", "ar": "Arkansas",
    "ca": "California", "co": "Colorado", "ct": "Connecticut", "de": "Delaware",
    "fl": "Florida", "ga": "Georgia", "hi": "Hawaii", "id": "Idaho",
    "il": "Illinois", "in": "Indiana", "ia": "Iowa", "ks": "Kansas",
    "ky": "Kentucky", "la": "Louisiana", "me": "Maine", "md": "Maryland",
    "ma": "Massachusetts", "mi": "Michigan", "mn": "Minnesota", "ms": "Mississippi",
    "mo": "Missouri", "mt": "Montana", "ne": "Nebraska", "nv": "Nevada",
    "nh": "New Hampshire", "nj": "New Jersey", "nm": "New Mexico", "ny": "New York",
    "nc": "North Carolina", "nd": "North Dakota", "oh": "Ohio", "ok": "Oklahoma",
    "or": "Oregon", "pa": "Pennsylvania", "ri": "Rhode Island", "sc": "South Carolina",
    "sd": "South Dakota", "tn": "Tennessee", "tx": "Texas", "ut": "Utah",
    "vt": "Vermont", "va": "Virginia", "wa": "Washington", "wv": "West Virginia",
    "wi": "Wisconsin", "wy": "Wyoming", "dc": "District of Columbia",
}

_TITLE_EXACT_SCORE    = 0.9  # Synthetic score assigned to SQL-injected exact-title
                              # matches; high enough to beat FAISS place articles
                              # after title reranking, while staying in [0, 1].
_MAX_TITLE_SUPPLEMENT = 3    # Max extra articles added to the FAISS pool per query.


def _title_rerank(query: str, articles: list[dict]) -> list[dict]:
    """
    Boost articles whose title words overlap with the query.

    Uses score-based (not rank-based) sorting so a semantically weaker article
    that happens to sit at FAISS rank 0 cannot hold off a better title match.

    Title-length normalisation penalises qualified titles like
    "George Washington, Washington" (3 tokens) relative to the exact title
    "George Washington" (2 tokens) when the query key-words number 2 — ensuring
    the most-specific match wins even when both share identical word overlap.

    Word tokens are extracted with a regex so punctuation (commas, hyphens)
    does not produce false mismatches against the query word set.
    """
    if not articles:
        return articles
    q_words = {
        w for w in _WORD_RE.findall(query.lower())
        if w not in _STOPWORDS and len(w) > 2
    }
    if not q_words:
        return articles

    def sort_key(art: dict) -> float:
        t_tokens = _WORD_RE.findall(art["title"].lower())
        t_words = set(t_tokens)
        overlap = len(q_words & t_words) / len(q_words)
        # Prefer titles whose token count is closest to the query key-word count.
        length_ratio = len(q_words) / max(len(t_tokens), len(q_words))
        return -(art["score"] + config.TITLE_BOOST * overlap * length_ratio)

    return sorted(articles, key=sort_key)


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

        # Fix 7 — Expand state abbreviations before embedding.
        # Only expand tokens that are all-uppercase (genuine abbreviations like "AK",
        # "ME") to avoid corrupting common lowercase words like "me", "in", "or",
        # "de" that happen to share a spelling with a state abbreviation.
        tokens = query.strip().split()
        tokens = [
            _STATE_ABBREVS.get(t.lower().rstrip("?.!"), t)
            if t.rstrip("?.!").isupper() and len(t.rstrip("?.!")) == 2
            else t
            for t in tokens
        ]
        query = " ".join(tokens)

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

        articles: list[dict] = [
            {
                "id": r["id"],
                "title": r["title"],
                "lead": r["lead"],
                "url_slug": r["url_slug"],
                "score": score_map.get(r["id"], 0.0),
            }
            for r in rows_sorted
        ]

        # ── Supplementary title search ─────────────────────────────────────
        # FAISS can miss the best-titled article for entity-name queries when
        # same-name city/place articles embed closer to the question structure.
        # Inject SQL title-matched articles into the candidate pool so that
        # _title_rerank always has the right article available to promote.
        q_ordered = [
            w for w in _WORD_RE.findall(query.lower())
            if w not in _STOPWORDS and len(w) > 2
        ]
        # Fix 6 — Expand nicknames so "tom" searches for "thomas".
        q_ordered = [_NICKNAMES.get(w, w) for w in q_ordered]
        if q_ordered and len(q_ordered) <= 6:  # Fix B: raised from 4 → 6
            existing_ids = {a["id"] for a in articles}
            candidate_title = " ".join(q_ordered)

            # Step 1: exact case-insensitive title match (zero false positives)
            extra_rows = self._conn.execute(
                "SELECT id, title, lead, url_slug FROM articles "
                "WHERE LOWER(title) = ? LIMIT ?",
                (candidate_title, _MAX_TITLE_SUPPLEMENT),
            ).fetchall()

            # Step 2: if exact match found nothing new, try LIKE-per-word.
            # LIKE uses substring matching, so post-filter with whole-word
            # tokenisation to avoid false positives (e.g. "Lightspeed Rescue"
            # matching [%speed%, %light%] for a "speed of light" query).
            if not any(r["id"] not in existing_ids for r in extra_rows):
                like_clauses = " AND ".join(
                    "LOWER(title) LIKE ?" for _ in q_ordered
                )
                like_params = [f"%{w}%" for w in q_ordered] + [_MAX_TITLE_SUPPLEMENT * 4]
                candidates = self._conn.execute(
                    f"SELECT id, title, lead, url_slug FROM articles "
                    f"WHERE {like_clauses} LIMIT ?",
                    like_params,
                ).fetchall()
                q_word_set = set(q_ordered)
                extra_rows = [
                    r for r in candidates
                    if q_word_set.issubset(set(_WORD_RE.findall(r["title"].lower())))
                ][:_MAX_TITLE_SUPPLEMENT]

            for row in extra_rows:
                if row["id"] not in existing_ids:
                    articles.append({
                        "id":       row["id"],
                        "title":    row["title"],
                        "lead":     row["lead"],
                        "url_slug": row["url_slug"],
                        "score":    _TITLE_EXACT_SCORE,
                    })

        return _title_rerank(query, articles)

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
