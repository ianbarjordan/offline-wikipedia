"""
04_embed_and_index.py

Embed all article lead paragraphs with sentence-transformers (GPU) and
build a FAISS IndexIVFPQ for fast approximate nearest-neighbour search.

Outputs:
  data/wikipedia.faiss  — FAISS index (nlist=1024, m=16, nbits=8)
  data/id_map.json      — maps FAISS sequential int index → SQLite article id

The index is built on the developer machine (CUDA recommended). The app
ships only the .faiss file and loads it on CPU at query time.

Index design notes
------------------
  Model     : all-MiniLM-L6-v2  (384-dim, L2-normalised → cosine similarity)
  Index type: IndexIVFPQ
    nlist=1024 — number of Voronoi cells (√230000 ≈ 480; 1024 gives good recall)
    m=16       — PQ sub-quantizers (384/16 = 24 dims each)
    nbits=8    — 256 centroids per sub-quantizer
  Metric    : METRIC_INNER_PRODUCT (≡ cosine after L2-normalisation)
  Training  : random sample of up to TRAIN_SAMPLE_SIZE vectors

Usage:
    python build/04_embed_and_index.py                      # auto-detects GPU
    python build/04_embed_and_index.py --device cpu         # force CPU
    python build/04_embed_and_index.py --batch-size 1024    # larger GPU batch
    python build/04_embed_and_index.py --batch-size 256 --train-sample 50000
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Paths relative to wiki-offline/ project root
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "wikipedia.db"
DEFAULT_FAISS_OUT = PROJECT_ROOT / "data" / "wikipedia.faiss"
DEFAULT_ID_MAP_OUT = PROJECT_ROOT / "data" / "id_map.json"

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# FAISS index parameters
NLIST = 1024
M = 16
NBITS = 8

DEFAULT_BATCH_SIZE_GPU = 1024   # GPU can process larger batches efficiently
DEFAULT_BATCH_SIZE_CPU = 256    # Smaller batches keep CPU RAM stable
DEFAULT_TRAIN_SAMPLE = 100_000

# How many vectors to add to the index per call (keeps RAM stable)
ADD_BATCH = 10_000


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


def resolve_device(requested: str) -> str:
    """
    Resolve 'auto' to the best available device; validate explicit choices.

    Priority order for 'auto': cuda → mps → cpu
    """
    try:
        import torch
    except ImportError:
        if requested in ("cuda", "mps"):
            print(
                f"WARNING: torch not importable; cannot use {requested}. "
                "Falling back to CPU.",
                file=sys.stderr,
            )
        return "cpu"

    if requested == "auto":
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"Auto-selected device: cuda  ({name})")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("Auto-selected device: mps  (Apple Silicon)")
            return "mps"
        print("Auto-selected device: cpu  (no GPU detected)")
        return "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            print(
                "WARNING: --device cuda requested but CUDA is not available. "
                "Falling back to CPU.",
                file=sys.stderr,
            )
            return "cpu"
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        return "cuda"

    if requested == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print(
                "WARNING: --device mps requested but MPS is not available. "
                "Falling back to CPU.",
                file=sys.stderr,
            )
            return "cpu"
        return "mps"

    return "cpu"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_articles(db_path: Path) -> tuple[list[int], list[str]]:
    """
    Return (ids, leads) lists loaded from the SQLite database,
    sorted by id ascending so that the FAISS sequential index
    position i corresponds to ids[i].
    """
    print(f"Loading articles from {db_path} ...")
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, lead FROM articles ORDER BY id ASC"
    ).fetchall()
    conn.close()

    ids = [r[0] for r in rows]
    leads = [r[1] for r in rows]
    print(f"  Loaded {len(ids):,} articles.")
    return ids, leads


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


def embed_all(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int,
) -> np.ndarray:
    """
    Embed *texts* in batches using *model*.
    Returns a float32 ndarray of shape (len(texts), EMBEDDING_DIM).
    Vectors are L2-normalised so inner product equals cosine similarity.
    """
    all_vecs: list[np.ndarray] = []

    n_batches = (len(texts) + batch_size - 1) // batch_size
    for i in tqdm(range(n_batches), desc="Embedding batches", unit="batch", dynamic_ncols=True):
        batch = texts[i * batch_size : (i + 1) * batch_size]
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalise → cosine via inner product
        )
        all_vecs.append(vecs.astype(np.float32))

    return np.vstack(all_vecs)


# ---------------------------------------------------------------------------
# FAISS index construction
# ---------------------------------------------------------------------------


def build_ivfpq(
    vectors: np.ndarray,
    nlist: int,
    m: int,
    nbits: int,
    train_sample: int,
) -> faiss.Index:
    """
    Train and populate an IndexIVFPQ with inner-product metric.

    Steps:
      1. Select a random training sample.
      2. Train the quantizer and PQ codebooks.
      3. Add all vectors in batches (prevents peak RAM spikes).
    """
    n, d = vectors.shape

    # IndexIVFPQ has two training requirements:
    #   1. IVF k-means  : ideally ≥ 39 * nlist training vectors
    #   2. PQ codebooks : ≥ 2^nbits = 256 training vectors (with nbits=8)
    # For small datasets fall back to a flat index so the script always works.
    pq_min = 2 ** nbits  # 256 for nbits=8
    if n < pq_min:
        print(
            f"\nBuilding IndexFlatIP  (dataset too small for IVFPQ: "
            f"{n} < {pq_min} = 2^nbits)"
        )
        print(f"  Vectors : {n:,} × {d}  [flat index — no training required]")
        index = faiss.IndexFlatIP(d)
        index.add(vectors)
        print(f"  Index total vectors: {index.ntotal:,}")
        return index

    # Scale nlist down if necessary to avoid IVF k-means failures
    ivf_min = nlist * 39
    if n < ivf_min:
        adjusted = max(1, n // 39)
        print(
            f"  WARNING: scaling nlist from {nlist} → {adjusted} "
            f"(need ≥{ivf_min} vectors for the requested nlist)."
        )
        nlist = adjusted

    print(f"\nBuilding IndexIVFPQ  (nlist={nlist}, m={m}, nbits={nbits})")
    print(f"  Vectors : {n:,} × {d}")

    # Inner-product quantizer: cosine similarity on normalised vectors
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

    # --- Training ---
    sample_size = min(train_sample, n)
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(n, size=sample_size, replace=False)
    train_vecs = vectors[sample_idx]

    print(f"  Training on {sample_size:,} random samples ...")
    index.train(train_vecs)
    assert index.is_trained, "FAISS index failed to train — check parameters."
    print(f"  Training complete.")

    # --- Adding all vectors in batches ---
    print(f"  Adding {n:,} vectors in batches of {ADD_BATCH:,} ...")
    for start in tqdm(range(0, n, ADD_BATCH), desc="Adding vectors", unit="batch", dynamic_ncols=True):
        index.add(vectors[start : start + ADD_BATCH])

    print(f"  Index total vectors: {index.ntotal:,}")
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed article leads and build the FAISS vector index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="SQLite database (output of 03_build_sqlite.py).",
    )
    parser.add_argument(
        "--faiss-out",
        type=Path,
        default=DEFAULT_FAISS_OUT,
        help="Output path for the FAISS index.",
    )
    parser.add_argument(
        "--id-map-out",
        type=Path,
        default=DEFAULT_ID_MAP_OUT,
        help="Output path for the FAISS-index → SQLite-id JSON map.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device for sentence-transformers. "
             "'auto' picks cuda > mps > cpu.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding batch size. "
             f"Defaults to {DEFAULT_BATCH_SIZE_GPU} for GPU, "
             f"{DEFAULT_BATCH_SIZE_CPU} for CPU.",
    )
    parser.add_argument(
        "--train-sample",
        type=int,
        default=DEFAULT_TRAIN_SAMPLE,
        help="Number of vectors used to train the FAISS quantizer.",
    )
    args = parser.parse_args()

    # --- Preflight checks ---
    if not args.db.exists():
        print(
            f"ERROR: Database not found: {args.db}\n"
            "Run 03_build_sqlite.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    device = resolve_device(args.device)
    batch_size = args.batch_size or (
        DEFAULT_BATCH_SIZE_GPU if device in ("cuda", "mps") else DEFAULT_BATCH_SIZE_CPU
    )
    print(f"Device: {device}   Batch size: {batch_size}")

    # --- Load data ---
    ids, leads = load_articles(args.db)
    if not ids:
        print("ERROR: No articles found in the database.", file=sys.stderr)
        sys.exit(1)

    # --- Embed ---
    print(f"\nLoading model '{MODEL_NAME}' on {device} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"  Model loaded.")

    print(f"\nEmbedding {len(leads):,} lead paragraphs (batch_size={batch_size}) ...")
    vectors = embed_all(leads, model, batch_size)
    print(f"  Embedding matrix: {vectors.shape}  dtype={vectors.dtype}")

    # --- Build FAISS index ---
    index = build_ivfpq(
        vectors,
        nlist=NLIST,
        m=M,
        nbits=NBITS,
        train_sample=args.train_sample,
    )

    # --- Save FAISS index ---
    args.faiss_out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving FAISS index to {args.faiss_out} ...")
    faiss.write_index(index, str(args.faiss_out))
    faiss_size_mb = args.faiss_out.stat().st_size / 1_048_576
    print(f"  Saved  ({faiss_size_mb:.1f} MB)")

    # --- Save id_map ---
    # FAISS assigns sequential IDs: position i in the index → ids[i]
    # We store {str(faiss_pos): sqlite_id} for fast lookup by the app.
    id_map = {str(i): art_id for i, art_id in enumerate(ids)}

    print(f"Saving id_map to {args.id_map_out} ...")
    args.id_map_out.write_text(
        json.dumps(id_map, separators=(",", ":")),
        encoding="utf-8",
    )
    id_map_size_kb = args.id_map_out.stat().st_size / 1024
    print(f"  Saved  ({id_map_size_kb:.0f} KB,  {len(id_map):,} entries)")

    # --- Quick sanity check ---
    print("\nRunning quick search sanity check ...")
    if hasattr(index, "nprobe"):
        index.nprobe = min(32, getattr(index, "nlist", 32))  # IVF indexes only
    test_vec = vectors[0:1]  # first article's own embedding
    distances, indices = index.search(test_vec, 3)
    print(f"  Top-3 results for article id={ids[0]} ({leads[0][:60]}...)")
    for rank, (dist, faiss_idx) in enumerate(zip(distances[0], indices[0])):
        mapped_id = id_map.get(str(faiss_idx), "?")
        print(f"    [{rank+1}] FAISS idx={faiss_idx}  sqlite_id={mapped_id}  score={dist:.4f}")

    print(f"\nBuild complete!")
    print(f"  {args.faiss_out}")
    print(f"  {args.id_map_out}")
    print(f"\nNext step: place phi-3-mini-q4_k_m.gguf in models/ and build the app.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
