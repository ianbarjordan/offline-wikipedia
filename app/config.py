"""
config.py

Single source of truth for all tuneable parameters.
Edit values here — no need to hunt through the codebase.

All Path constants are resolved relative to the project root (wiki-offline/)
at import time, so they work correctly regardless of the working directory
from which the app is launched or how PyInstaller packages it.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root: wiki-offline/  (one directory above app/)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K: int = 5    # Articles returned per query
NPROBE: int = 32  # FAISS IVF cells searched per query (higher → better recall,
                  # slightly slower).  Has no effect on flat indexes.
CONFIDENCE_THRESHOLD: float = 0.35  # Min cosine similarity for the top result to
                                     # pass the LLM confidence gate.  Queries whose
                                     # best match falls below this score return a
                                     # canned "not found" response without LLM call.
TITLE_BOOST: float = 2.0             # Rank positions a perfect query-title word
                                     # overlap is worth during post-retrieval rerank.

# ---------------------------------------------------------------------------
# Embedding model  (must match the model used in build/04_embed_and_index.py)
# ---------------------------------------------------------------------------
EMBED_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384

# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------
MODEL_PATH: Path = PROJECT_ROOT / "models" / "phi-3-mini-q4_k_m.gguf"
CTX_WINDOW: int = 4096                        # Token context window
N_GPU_LAYERS: int = 0                         # 0 = CPU-only (end-user machines)
N_THREADS: int = max(1, (os.cpu_count() or 2) - 1)  # Leave one core for the OS

# ---------------------------------------------------------------------------
# Conversation memory  (Option A — LLM context only)
# ---------------------------------------------------------------------------
CHAT_HISTORY_TURNS: int = 3  # Past exchanges included in the LLM prompt

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DB_PATH: Path = PROJECT_ROOT / "data" / "wikipedia.db"
FAISS_PATH: Path = PROJECT_ROOT / "data" / "wikipedia.faiss"
ID_MAP_PATH: Path = PROJECT_ROOT / "data" / "id_map.json"
ARTICLES_DIR: Path = PROJECT_ROOT / "data" / "articles"
