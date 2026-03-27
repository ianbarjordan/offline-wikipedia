# Plan: Adapting wiki-offline for Full English Wikipedia

## Scale Reality Check

| Metric | Simple English | Full English | Ratio |
|--------|----------------|--------------|-------|
| Articles | 274,630 | ~6.8 million | 25× |
| Compressed dump | ~2 GB (.bz2) | ~22 GB | 11× |
| Leads-only SQLite | ~617 MB | ~12–14 GB | ~22× |
| FAISS index | 8.2 MB | ~300 MB | 37× |
| HTML files (6.8M × 5 KB) | ~15 GB | **~34 GB — not viable** | — |
| Build time (RTX 3060 Ti) | ~2 hrs | **~24–48 hrs** | ~20× |
| Peak disk during build | ~20 GB | **~80–95 GB** | ~4× |

---

## Architectural Decision: Remove Body Storage and HTML Files

The current design writes one `.html` file per article (`data/articles/{id}.html`). For
6.8 million articles this produces 34+ GB of files that take days to write and cannot be
bundled in an installer.

**Decision**: Drop body text entirely from the pipeline. Store only `lead`, `title`, and
`url_slug` in SQLite. Source buttons open `en.wikipedia.org/wiki/{slug}` in the browser
instead of a local file. This is a deliberate trade-off — the LLM already uses only leads
for answering, and English Wikipedia's live articles are authoritative.

This decision propagates through every script:
- `02_parse_articles.py` — stop extracting body, skip `normalise_body_whitespace()`
- `03_build_sqlite.py` — drop `body` column, skip HTML generation, drop `articles/` dir
- `gui.py` — replace `_open_file()` with `webbrowser.open()` to `en.wikipedia.org`
- `wiki-offline.spec` — remove `data/articles/` Tree entry

---

## Critical Bug in `04_embed_and_index.py`: RAM Exhaustion

This is the most serious issue. `load_articles()` calls `.fetchall()` which loads **all
leads into RAM at once**. `embed_all()` then appends every batch to `all_vecs` and calls
`np.vstack()` at the end — keeping **all 6.8M × 384 float32 = 10.5 GB** in RAM
simultaneously. On top of the SQLite leads already in memory, most Windows machines will
OOM before embedding is half done.

**Fix**: Streaming memmap architecture — embed article leads in chunks directly into a
memory-mapped numpy file on disk, then sample from the mmap for FAISS training. See
detailed changes in the `04_embed_and_index.py` section below.

---

## Changes Required by Script

---

### `build/01_download_wiki.py`

**Problem 1 — URL**
The current `build_url()` hardcodes the `simplewiki_content` path under
`cirrus_search_index`. English Wikipedia dumps may use a different URL format and are
likely split across **multiple numbered files** (`-00000`, `-00001`, …) because a single
file would be ~22 GB.

> **ACTION BEFORE CODING**: Visit
> `https://dumps.wikimedia.org/other/cirrus_search_index/`
> Open the latest dated directory and find the `enwiki_content` subdirectory to confirm
> the exact URL pattern and number of part files. Update `build_url()` and
> `local_filename()` accordingly.

**Problem 2 — No resume on partial downloads**
The current code deletes partial downloads on failure and skips if the file exists. For a
22 GB file, a network dropout near the end means starting over. Replace with proper
**HTTP Range-request resumption**: check the existing file size, send
`Range: bytes={existing_size}-`, and append to the file.

**Problem 3 — Multi-file dumps**
If enwiki is split into multiple numbered files, add a `--parts N` argument and a loop
that downloads each part in sequence.

**All changes**:
- Add `--wiki` argument (`simplewiki` / `enwiki`, default `enwiki`)
- Update `build_url()` and `local_filename()` for enwiki URL pattern (verify first)
- Replace `download()` with an HTTP Range-resume version that appends to partial files
- Add multi-part download loop if enwiki dump is split
- Increase `CHUNK_SIZE` from `256 * 1024` to `4 * 1024 * 1024` (4 MB) for throughput
- Add disk-space preflight: warn if destination drive has < 30 GB free

---

### `build/02_parse_articles.py`

**Problem 1 — Body extraction wastes time and disk**
`normalise_body_whitespace()` is called on every article's full body text. English
Wikipedia articles can have 50,000+ word bodies. Storing body in JSONL would produce a
~50–80 GB staging file. Since body is being dropped from the pipeline, skip it entirely.

**Problem 2 — No checkpoint/resume**
For a 6–10 hour parse job, a crash means starting over. `bz2.open()` cannot seek, so
resumption must be line-count based: count lines already written to the output JSONL,
then skip that many input documents before resuming writes.

**Problem 3 — Hard-coded `simplewiki` glob**
`find_dump()` only finds `simplewiki_content-*` files.

**All changes**:
- Drop `body = normalise_body_whitespace(doc.get("text") or "")` entirely
- Write `{"title": ..., "lead": ...}` records (no `body` key)
- Add `--wiki` argument; update `find_dump()` glob pattern to match `enwiki_content-*`
- Add checkpoint/resume:
  - Copy `count_lines()` from `03_build_sqlite.py` into this script
  - At startup, if the output JSONL already exists, call `count_lines()` on it
  - Skip the first N document lines from the input stream (N = existing line count)
  - Set `counts["written"] = N` as the starting baseline
  - Open the output JSONL in append mode (`"a"`) instead of write mode (`"w"`)

---

### `build/03_build_sqlite.py`

**Problem 1 — HTML generation**
All of `render_html()`, `body_to_html()`, `_esc()`, `_HTML_TEMPLATE`, and the
`html_path.write_text()` call inside `build()` must be removed. Writing 6.8M files to
`data/articles/` is not viable — it would take days and produce 34+ GB of data.

**Problem 2 — `body` column**
The schema includes `body TEXT NOT NULL`. Drop this column and remove `body` from the
`INSERT` statement. It is never populated and wastes schema space.

**Problem 3 — `DROP TABLE IF EXISTS` prevents resume**
After a crash at article 4 million, the DB is lost because `create_schema()` always drops
and recreates the table. A 6-hour build must not be unrecoverable.

Fix: replace `create_schema()` with resume-aware logic:
```python
def table_exists(conn):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='articles'"
    ).fetchone()
    return row is not None

def create_schema(conn):
    conn.execute("""
        CREATE TABLE articles (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            title    TEXT NOT NULL,
            lead     TEXT NOT NULL,
            url_slug TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX idx_title ON articles (title)")
    conn.commit()
```

At build startup:
```python
if table_exists(conn):
    already_inserted = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    # skip first already_inserted lines from the JSONL, set inserted = already_inserted
else:
    create_schema(conn)
    already_inserted = 0
```

Add `--fresh` flag that drops and recreates the table if the user explicitly wants a clean
rebuild.

**Problem 4 — FTS5 index needed for fast title search**
Without this, `retriever.py`'s LIKE-based title supplement performs a full-table scan on
6.8M rows. Add FTS5 build at the end of `build()`:
```python
print("Building FTS5 title index (10–20 minutes)...")
conn.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts
    USING fts5(title, content=articles, content_rowid=id)
""")
conn.execute("INSERT INTO articles_fts(articles_fts) VALUES('rebuild')")
conn.commit()
```

**All changes**:
- Remove `body` column from schema and `INSERT`
- Remove `_HTML_TEMPLATE`, `_SECTION_HEAD_RE`, `body_to_html()`, `render_html()`,
  `_esc()`, `DEFAULT_ARTICLES_DIR`, and `articles_dir` parameter from `build()`
- Remove `--articles-dir` argparse argument
- Replace `create_schema()` + `DROP TABLE` with resume-aware version above
- Add `--fresh` flag for explicit clean rebuild
- Add FTS5 build step at end of `build()`
- Increase `COMMIT_EVERY` from `2_000` to `10_000`
- Add disk-space preflight: warn if data drive has < 15 GB free
- Print rows/sec and ETA during build

---

### `build/04_embed_and_index.py`

This script requires the most significant rewrite.

**Problem 1 — OOM from fetchall + vstack (critical)**

Current memory pattern:
```
load_articles()  →  .fetchall()  →  all leads in RAM     (~10 GB)
embed_all()      →  np.vstack()  →  all vectors in RAM   (~10.5 GB)
                                                          ~20+ GB peak
```

New pattern using numpy memmap:
```
query total N from SQLite
pre-allocate raw/embeddings.npy  (N × 384 × float32  =  10.5 GB on disk)
pre-allocate raw/ids_build.npy   (N × int64          =  55 MB on disk)

for each chunk of 50,000 articles (streaming from SQLite with LIMIT/OFFSET):
    embed chunk  →  50,000 × 384 × 4 bytes  =  75 MB in RAM
    write chunk's vectors into embeddings.npy at the right offset
    write chunk's ids    into ids_build.npy  at the right offset
    write chunk index    to   raw/embed_progress.txt

open embeddings.npy as read-only memmap
sample 500,000 rows for FAISS training
add all vectors from memmap in batches of 100,000

save FAISS index
copy ids_build.npy  →  data/id_map.npy  (final output)
delete raw/embeddings.npy and raw/ids_build.npy (optionally, with --keep-mmap flag)
```

Resume logic: at startup, read `embed_progress.txt` to get the last completed chunk
index. Open the existing memmap files and continue writing from that position. Skip the
already-embedded chunks in the SQLite streaming query using `OFFSET`.

**Problem 2 — NLIST too small**
For 6.8M vectors, `NLIST = 1024` gives ~6,640 vectors per Voronoi cell — far too many,
degrading search quality. Use `NLIST = 8192` (~830 vectors/cell). The IVF training
minimum of `39 × nlist = 319,488` is satisfied by the 500K training sample.

**Problem 3 — TRAIN_SAMPLE too small**
`DEFAULT_TRAIN_SAMPLE = 100_000`. At nlist=8192, FAISS recommends ≥ 256 × nlist =
2,097,152. The practical compromise is **500,000** — this is 500K × 384 × 4 bytes = 768
MB in RAM for the training vectors, which fits comfortably.

**Problem 4 — id_map as JSON**
`{str(i): int}` for 6.8M entries = ~200–400 MB JSON file, slow to parse at app startup
(5+ seconds). Replace with numpy binary:
- Save: `np.save('data/id_map.npy', np.array(ids, dtype=np.int64))`
- Load in app: `self.id_map = np.load(config.ID_MAP_PATH)` → `int(self.id_map[pos])`
- Size: exactly 55 MB, loads in < 1 second

**Problem 5 — ADD_BATCH too small**
`ADD_BATCH = 10_000` means 680 loop iterations. Increase to `100_000` to reduce overhead
when adding from the memmap.

**All changes**:
- Remove `load_articles()` and `embed_all()` entirely
- Add `embed_chunked_to_mmap(db_path, mmap_path, ids_path, progress_path, chunk_size, model, batch_size, device)` function
- Add `load_checkpoint(progress_path)` → returns last completed chunk index (0 if none)
- Add `--embed-chunk-size` argument (default `50_000`)
- Add `--keep-mmap` flag to preserve `raw/embeddings.npy` after build (useful for re-indexing)
- Change `NLIST = 1024` → `8192`; add `--nlist` argument
- Change `DEFAULT_TRAIN_SAMPLE = 100_000` → `500_000`
- Change `ADD_BATCH = 10_000` → `100_000`
- Replace JSON id_map with `.npy`; update `DEFAULT_ID_MAP_OUT` to `data/id_map.npy`
- Update sanity check: load id_map as `np.load()`, index directly by integer position
- Add disk-space preflight: warn if `raw/` drive has < 12 GB free for the embeddings mmap

---

### `app/config.py`

| Constant | Current value | New value | Reason |
|----------|---------------|-----------|--------|
| `NPROBE` | `32` | `64` | With nlist=8192, 64 probes searches ~0.8% of cells |
| `ID_MAP_PATH` | `data/id_map.json` | `data/id_map.npy` | numpy binary |
| `ARTICLES_DIR` | `data/articles/` | *(remove)* | No HTML files generated |

Add new constant:
```python
WIKI_BASE_URL = "https://en.wikipedia.org/wiki/"
```

---

### `app/retriever.py`

**Change 1 — numpy id_map loading**

Replace:
```python
self.id_map = json.loads(config.ID_MAP_PATH.read_text())
# usage: sqlite_id = self.id_map[str(pos)]
```
With:
```python
self.id_map = np.load(config.ID_MAP_PATH)   # shape (N,), dtype int64
# usage: sqlite_id = int(self.id_map[pos])
```
Also update the `import` block: remove `import json`, ensure `import numpy as np` is present.

**Change 2 — Replace LIKE title supplement with FTS5**

The existing LIKE fallback in the SQL title supplement performs a full-table scan on 6.8M
rows even with `idx_title`, because `LIKE '%word%'` cannot use a B-tree index. Replace
with FTS5:

```python
# Exact match (uses idx_title B-tree — fast):
rows = conn.execute(
    "SELECT id, title, lead, url_slug FROM articles WHERE LOWER(title) = ?",
    (candidate_title.lower(),)
).fetchall()

# FTS5 fallback (replaces LIKE):
if not rows:
    rows = conn.execute(
        """SELECT a.id, a.title, a.lead, a.url_slug
           FROM articles_fts f
           JOIN articles a ON a.id = f.rowid
           WHERE articles_fts MATCH ?
           LIMIT ?""",
        (candidate_title, _MAX_TITLE_SUPPLEMENT * 4)
    ).fetchall()
```

**Change 3 — Source URL domain**

Any code path constructing a `simple.wikipedia.org` URL should use `config.WIKI_BASE_URL`
(`https://en.wikipedia.org/wiki/`). Search for the string `"simple.wikipedia.org"` in the
file and replace all occurrences.

---

### `app/gui.py`

Replace the source button handler's `_open_file()` call with:

```python
import webbrowser
import urllib.parse

webbrowser.open(
    config.WIKI_BASE_URL + urllib.parse.quote(art["url_slug"], safe=":")
)
```

Remove `import os` and the `_open_file()` function if they are no longer used elsewhere.

---

### `wiki-offline.spec`

- Remove the `data/articles/` `Tree()` entry from the `datas` list
- Update the pre-build checklist comment: note that `data/wikipedia.db` is now ~13 GB
  and the installer output will be ~6–8 GB compressed
- No other structural changes required

---

## Disk Space Budget (Windows Build Machine)

| Path | Size |
|------|------|
| `raw/enwiki_content-*.json.bz2` | 22–25 GB |
| `raw/articles_parsed.jsonl` (leads only, no body) | 6–9 GB |
| `raw/embeddings.npy` (memmap, build-time only) | 10.5 GB |
| `raw/ids_build.npy` (memmap, build-time only) | 55 MB |
| `data/wikipedia.db` (leads + FTS5 index) | 12–14 GB |
| `data/wikipedia.faiss` | 250–350 MB |
| `data/id_map.npy` | 55 MB |
| `dist/WikiOffline/` (PyInstaller output) | 14–16 GB |
| `installer/WikiOffline-Setup.exe` (lzma2/max) | 6–8 GB |
| **Peak total (all phases)** | **~80–95 GB** |

**Minimum recommended free space on build drive: 120 GB**

The two large temporary files (`raw/embeddings.npy` and `raw/articles_parsed.jsonl`) can
be deleted after step 04 completes to recover ~17 GB before the PyInstaller step.

---

## Build Time Estimates (Windows, RTX 3060 Ti)

| Step | Estimated Time |
|------|----------------|
| `01` Download enwiki dump | 4–15 hours (depends on connection) |
| `02` Parse dump → JSONL | 3–6 hours |
| `03` SQLite + FTS5 build | 3–6 hours |
| `04` Embed + FAISS index | 12–20 hours (GPU required) |
| PyInstaller | 30–60 minutes |
| `iscc installer.iss` | 3–6 hours (lzma2/max on 14+ GB) |
| **Total wall-clock** | **~2–4 days** |

> GPU is non-negotiable for step 04. CPU embedding would take 60–120 hours.

It is safe to run steps 01–03 sequentially over one evening/night, then start step 04
before going to bed. Each step has checkpoint/resume so interruptions are recoverable.

---

## FAISS Parameter Rationale

| Parameter | Simple Wiki | Full Wiki | Rationale |
|-----------|-------------|-----------|-----------|
| N | 274,630 | 6,800,000 | total articles |
| `NLIST` | 1,024 | **8,192** | √6.8M ≈ 2,608; 3× for quality |
| `M` | 16 | 16 | keep — good compression/recall balance |
| `NBITS` | 8 | 8 | keep — standard PQ configuration |
| `NPROBE` | 32 | **64** | ~0.8% of cells searched per query |
| Training samples | 100,000 | **500,000** | > 100 × nlist; fits in 768 MB RAM |
| `ADD_BATCH` | 10,000 | **100,000** | reduces loop overhead when adding |

---

## Implementation Order

1. **Verify enwiki dump URL** — visit `https://dumps.wikimedia.org/other/cirrus_search_index/`,
   open the latest dated directory, confirm the `enwiki_content` path structure and number
   of part files before writing any download code.

2. `build/01_download_wiki.py` — URL update, HTTP Range resume, multi-part support

3. `build/02_parse_articles.py` — drop body extraction, add checkpoint/resume

4. `build/03_build_sqlite.py` — drop body/HTML, resume-aware schema, FTS5

5. `build/04_embed_and_index.py` — memmap streaming, checkpoint, nlist/training/id_map

6. `app/config.py` — NPROBE, ID_MAP_PATH, WIKI_BASE_URL, remove ARTICLES_DIR

7. `app/retriever.py` — numpy id_map, FTS5 query, en.wikipedia.org URL

8. `app/gui.py` — webbrowser.open to en.wikipedia.org

9. `wiki-offline.spec` — remove articles Tree entry

10. Full build run on Windows

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| 22 GB download interrupted | HTTP Range-resume; partial file preserved and continued |
| Parse job killed after 4M articles | Line-count checkpoint; resume appends to existing JSONL |
| SQLite build crashes at 5M rows | Row-count checkpoint; no DROP TABLE on resume |
| Embedding OOM | Memmap streaming; peak RAM ~800 MB per chunk, never all-at-once |
| Embedding job killed after 10 hours | `embed_progress.txt` checkpoint; resume from last chunk |
| FAISS training OOM | 500K sample = 768 MB — fits in RAM on any modern machine |
| SQL title search too slow on 6.8M rows | FTS5 virtual table replaces LIKE fallback |
| id_map slow to load at app startup | numpy binary: 55 MB, loads in < 1 second |
| Drive runs out of space mid-build | Disk-space preflight check in each script |
| Multi-part enwiki dump | Download loop; all parts parsed sequentially by 02 |
