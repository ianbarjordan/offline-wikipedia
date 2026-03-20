# wiki-offline — Architecture Context

## Project Summary

A fully offline Windows desktop application that lets non-technical users query
Simple English Wikipedia using a local LLM (RAG pipeline). Zero internet
required after installation.

---

## High-Level Architecture

```
[Build phase — developer machine]          [App — ships to end users]
                                                        │
01_download_wiki.py ──►                                 │
02_parse_articles.py ──►                                │
03_build_sqlite.py   ──►  data/wikipedia.db             │──► retriever.py
                          data/articles/{id}.html        │
04_embed_and_index.py──►  data/wikipedia.faiss           │──► retriever.py
                          data/id_map.json               │
                          models/*.gguf (manual step) ───┤──► llm.py
                                                         │
                                                pipeline.py
                                                         │
                                                    gui.py / main.py
```

---

## Directory Layout

```
wiki-offline/
├── build/                     ← Developer-only; NOT shipped to end users
│   ├── 01_download_wiki.py    ← Download CirrusSearch dump from Wikimedia
│   ├── 02_parse_articles.py   ← Parse dump → raw/articles_parsed.jsonl
│   ├── 03_build_sqlite.py     ← JSONL → data/wikipedia.db + HTML files
│   ├── 04_embed_and_index.py  ← Embed leads → FAISS index + id_map
│   └── requirements_build.txt
├── app/                       ← Everything that ships to end users
│   ├── main.py                ← Entry point; starts Gradio server          [DONE]
│   ├── config.py              ← All tuneable constants                      [DONE]
│   ├── retriever.py           ← FAISS search + SQLite lookup                [DONE]
│   ├── llm.py                 ← llama-cpp-python wrapper                    [DONE]
│   ├── pipeline.py            ← RAG orchestration                           [DONE]
│   ├── gui.py                 ← Gradio Blocks interface                     [DONE]
│   └── requirements_app.txt                                                 [DONE]
├── data/
│   ├── wikipedia.db           ← SQLite: id, title, lead, body, url_slug
│   ├── wikipedia.faiss        ← FAISS IndexIVFPQ (384-dim, IP metric)
│   ├── id_map.json            ← {str(faiss_pos): sqlite_id}
│   └── articles/              ← {id}.html — self-contained offline HTML
├── models/
│   └── phi-3-mini-q4_k_m.gguf  ← Downloaded manually from HuggingFace
├── raw/                       ← Scratch space (gitignored)
│   ├── simplewiki-*.json.gz   ← Original dump download
│   └── articles_parsed.jsonl  ← Staging file between scripts 02 and 03
└── wiki-offline.spec          ← PyInstaller spec (single-folder output)
```

---

## Data Flow

### Build Phase

1. **01 → raw/simplewiki-*.json.gz**
   - Stream-downloads from dumps.wikimedia.org
   - ~2-3 GB compressed

2. **02 → raw/articles_parsed.jsonl**
   - Streams through gzip; processes alternating action/document line pairs
   - Filters: namespace=0, no redirect, no disambiguation
   - Fields: title, lead (≤300 words from `opening_text`), body (`text`)
   - ~230,000 articles for Simple English Wikipedia

3. **03 → data/wikipedia.db + data/articles/{id}.html**
   - SQLite schema: `articles(id PK AUTOINCREMENT, title, lead, body, url_slug)`
   - `url_slug = title.replace(" ", "_")` — Wikipedia convention
   - HTML: self-contained, inline CSS, no external dependencies
   - Individual `cur.execute` + `cur.lastrowid` for reliable ID tracking

4. **04 → data/wikipedia.faiss + data/id_map.json**
   - Loads `(id, lead)` pairs ordered by `id ASC` from SQLite
   - Embeds with `all-MiniLM-L6-v2` (384-dim) on CUDA, L2-normalised
   - FAISS position `i` → `ids[i]` (SQLite article id)
   - `id_map = {str(i): ids[i]}` — JSON for portability
   - IndexIVFPQ: `nlist=1024, m=16, nbits=8`, METRIC_INNER_PRODUCT

### Query Phase (App)

```
User question
    │
    ▼
retriever.search(query, top_k=5)
    ├─ embed query with all-MiniLM-L6-v2 (CPU)
    ├─ index.search() → top-k FAISS positions + squared-L2 distances
    ├─ convert distances → cosine similarity scores (1 - d/2)
    ├─ id_map[pos] → SQLite ids
    ├─ SELECT title, lead, url_slug FROM articles WHERE id IN (...)
    ├─ attach score to each article dict
    └─ _title_rerank(): boost articles whose title words match query
    │
    ▼
pipeline.query(user_message, chat_history)
    ├─ confidence gate: if articles[0]["score"] < CONFIDENCE_THRESHOLD
    │    └─ return canned "not found" generator (skip LLM entirely)
    ├─ retriever results → context block
    ├─ last 3 chat_history exchanges → conversational context
    ├─ build prompt with strict grounding system instructions
    └─ llm.generate(prompt, stream=True)
    │
    ▼
gui.py streaming callback → Gradio Chatbot
    └─ source buttons → webbrowser.open(simple.wikipedia.org/wiki/{slug})
```

---

## Key Dependencies

| Layer | Package | Purpose |
|-------|---------|---------|
| Build | `requests`, `tqdm` | Download with progress |
| Build | `sentence-transformers` | all-MiniLM-L6-v2 (CUDA) |
| Build | `faiss-cpu` | Index construction |
| App | `sentence-transformers` | Query embedding (CPU) |
| App | `faiss-cpu` | ANN search at query time |
| App | `llama-cpp-python` | Phi-3 Mini GGUF inference |
| App | `gradio` | Offline web UI |

---

## Configuration

All tuneable constants live in `app/config.py`:
- `TOP_K`, `NPROBE`, `CONFIDENCE_THRESHOLD`, `TITLE_BOOST` — retrieval
- `N_THREADS`, `N_GPU_LAYERS`, `CTX_WINDOW` — LLM performance
- `EMBED_MODEL`, `EMBEDDING_DIM`, `CHAT_HISTORY_TURNS`
- `MODEL_PATH`, `DB_PATH`, `FAISS_PATH`, `ID_MAP_PATH`, `ARTICLES_DIR`

---

## Entry Points

| Script | Purpose |
|--------|---------|
| `build/01_download_wiki.py` | Download dump |
| `build/02_parse_articles.py` | Parse → JSONL |
| `build/03_build_sqlite.py` | JSONL → DB + HTML |
| `build/04_embed_and_index.py` | DB → FAISS |
| `app/main.py` | Launch app (end user entry point) |

---

## Session State — Where We Are

### Completed
- **All four build scripts** (`build/01`–`04`) — written and smoke-tested end-to-end
  with synthetic data. All filters, slugs, HTML generation, and FAISS construction
  verified. `04` auto-scales nlist and falls back to `IndexFlatIP` for datasets too
  small to train `IndexIVFPQ` (PQ requires ≥ 256 training vectors).
- **`app/config.py`** — all constants in place; paths resolved at import time via
  `Path(__file__).parent.parent.resolve()` so they are absolute and correct regardless
  of working directory.
- **`app/retriever.py`** — written and smoke-tested with 7 behavioural assertions.
- **`scratch/gradio_test.py`** — minimal Gradio spike app used to interrogate the
  installed package (see Gradio Spike Results below).
- **`app/llm.py`** — written. `LLM` class wraps `llama_cpp.Llama`. Loads model once
  at startup. `generate(prompt, stream=True)` returns a token generator when
  `stream=True`, a full string when `stream=False`. Smoke test exits gracefully with
  SKIP when model file is absent (expected on dev machines without the manual download).
  llama-cpp-python 0.3.16 installed. Gradio offline strategy: **Option A** (no patching).
- **`app/pipeline.py`** — written. `Pipeline(retriever, llm)` class. `query(user_message,
  chat_history)` returns `(stream_generator, articles_list)`. Uses Phi-3 chat template
  (`<|system|>...<|end|>` / `<|user|>` / `<|assistant|>`). Context block is numbered
  article list embedded in system message. Last `CHAT_HISTORY_TURNS` (3) exchanges
  included. Prompt building is pure functions (`_build_context`, `_build_prompt`) tested
  independently of data files. 5 assertions pass.
- **`app/gui.py`** — written. `create_ui(pipeline) -> gr.Blocks`. Gradio 6.7 messages
  format (`gr.ChatMessage`). Streaming generator handler. Sources row with `TOP_K` (5)
  fixed `gr.Button` slots — shown/hidden via `gr.update()` after each response. Article
  HTML opened via `_open_file()` (os.startfile Windows / xdg-open Linux). `CSS` public
  constant for `main.py` to pass to `launch()`. 5 assertions pass, no warnings.
- **`app/main.py`** — written. Startup sequence: env var → argparse (`--port`, `--no-browser`)
  → pre-flight file checks (clear per-file errors+hints) → Retriever → LLM → Pipeline →
  create_ui → `_schedule_browser_open()` daemon thread → `demo.launch(quiet=True, css=CSS)`.
  `_wait_for_keypress()` for Windows terminal on error. Tested: pre-flight correctly
  reports all 4 missing files with hints; `--help` shows all args.
- **`app/requirements_app.txt`** — written. Pins all 5 direct deps: gradio==6.7.0,
  llama-cpp-python==0.3.16, faiss-cpu==1.13.2, sentence-transformers==5.2.3, numpy==2.4.2.
  Comments explain torch CPU wheel URL and embedding model pre-download step.
- **`wiki-offline.spec`** — written. `--onedir` PyInstaller spec. Uses `importlib.util.find_spec`
  for portable package discovery. Bundles: Gradio templates/_simple_templates/icons/media_assets/
  hash_seed+package.json, gradio_client data files, all 5 llama_cpp .so files to `llama_cpp/lib/`
  (so ctypes `__file__`-relative lookup works), HF cache snapshot for embedding model,
  data/ and models/ directories. Runtime hook (`hooks/runtime_env.py`) sets `HF_HOME` →
  `sys._MEIPASS/hf_cache` before any import. Warns at spec-build time if pre-build checklist
  steps are incomplete. All 7 data paths validated OK.

### In Progress
- Nothing. App is feature-complete.

### Not Yet Started
- Nothing. All planned files are written.

---

## Gradio Spike Results (scratch/gradio_test.py)

**Installed version:** `6.7.0`

**Static asset directory (Windows path will differ by venv):**
```
<site-packages>/gradio/templates/frontend/
```
On this dev machine:
```
/home/claude-user/.local/lib/python3.11/site-packages/gradio/templates/frontend/
```
Contains 862 files, ~25 MB total.
Sub-structure:
```
frontend/
├── index.html          ← served on every page load (Jinja2 template)
├── share.html          ← share-link variant (same CDN lines)
├── assets/             ← 798 hashed JS/CSS bundles
└── static/fonts/       ← 12 woff2 files, fully bundled (no Google Fonts needed)
```

**CDN reference inventory — every external URL found across all 1,820 files:**

| URL | Tag / location | Unconditional? | Offline impact |
|-----|---------------|---------------|----------------|
| `fonts.googleapis.com` | `<link rel="preconnect">` in `index.html` | Yes | **None.** Preconnect hint only. All `@font-face` rules in the main CSS use local `static/fonts/` woff2 files — no `http://` `url()` calls exist in the CSS. |
| `fonts.gstatic.com` | `<link rel="preconnect">` in `index.html` | Yes | **None.** Same reason — vestigial preconnect, no font actually fetched from Google. |
| `cdnjs.cloudflare.com` / `iframe-resizer 4.3.1` | `<script async src="...">` in `index.html` | **Yes — every page load** | **Fails silently.** `async` = non-blocking. Script is only needed when Gradio is embedded inside a foreign `<iframe>` (for resize signalling). Running standalone at `127.0.0.1:7860`, this code path is never exercised and its absence has no user-visible effect. |
| `unpkg.com/@ffmpeg/core@0.12.9` | Hardcoded in `assets/worker-BAOIWoxA.js` | No — only if FFmpeg invoked | **None for us.** Triggered only by `gr.Video` with FFmpeg processing. We are not using that component. |
| `raw.githubusercontent.com` (header image) | `<meta property="og:image">` in `index.html` | Yes | **None.** Open Graph meta tag — fetched only by social-media crawlers, not by the browser during a normal page load. |

**Overall verdict:** No template patching is required for our standalone offline use case.
The one live network call (iframe-resizer, `async`) fails silently and is functionally
irrelevant. Fonts are 100% local. Analytics suppressed via `GRADIO_ANALYTICS_ENABLED=False`.

**Unresolved GUI strategy question (decide at session start):**
The CDN audit shows patching is not strictly necessary, but we should explicitly decide
between two approaches before writing `gui.py`:
- **Option A (no patch):** Accept the silent async failure. Zero maintenance overhead.
  Recommended — clean, correct, no fragility against Gradio updates.
- **Option B (patch index.html at startup):** In `main.py`, overwrite the two
  preconnect lines and the iframe-resizer `<script>` tag before launching Gradio.
  Eliminates even the silent network attempt. Fragile if Gradio bumps its template.

---

## Design Decisions — retriever.py

Two implementation choices worth preserving for anyone reading the code later:

1. **`index.nprobe` set at load time, not per-query.**
   `nprobe` is a property of the FAISS index object, not a per-search parameter. Setting
   it once in `__init__` is both correct and cheaper. A `hasattr(self.index, "nprobe")`
   guard is used so the same code works unchanged if the index is a `FlatIP` (which has
   no `nprobe` attribute and no cell-probe concept).

2. **SQL results re-sorted after `IN` clause to restore FAISS rank order.**
   `SELECT ... WHERE id IN (...)` does not guarantee row order matching the `IN` list.
   After the query, results are re-sorted using a `{sqlite_id: faiss_rank}` dict so the
   caller always receives articles in descending similarity order, regardless of SQLite's
   internal scan order.

---

## PyInstaller Warning — Path Resolution When Frozen

`config.py` resolves paths via:
```python
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
```
This works correctly when running from source. **When frozen by PyInstaller**,
`__file__` is not a reliable filesystem path inside a `--onedir` bundle:

- In `--onedir` mode (our target): `__file__` resolves correctly relative to the
  extracted bundle directory. `Path(__file__).parent.parent` points to the bundle
  root. **No change needed for `--onedir`.**
- In `--onefile` mode: PyInstaller extracts to a temp dir (`sys._MEIPASS`); `__file__`
  may not exist. If we ever switch to `--onefile`, replace the path resolution with:
  ```python
  import sys
  PROJECT_ROOT = Path(getattr(sys, '_MEIPASS', Path(__file__).parent.parent)).resolve()
  ```
  This is **not needed now** but must be remembered if the spec ever changes from
  `onedir` to `onefile`.

---

## POC Fix Session — Changes Applied

**Issue 2 — Streaming artifact fix (`app/llm.py`)**
`_stream_tokens` now uses defensive `.get()` access:
`choices = chunk.get("choices") if isinstance(chunk, dict) else None`
instead of hard `chunk["choices"][0]["text"]` indexing.

**Issue 3 — Grounding + confidence gate (`app/pipeline.py`, `app/config.py`)**
- System prompt rewritten to say "exclusively from the Wikipedia context below.
  Do NOT use your training knowledge."
- Confidence gate added: if `articles[0]["score"] < CONFIDENCE_THRESHOLD`,
  returns a canned `_LOW_CONFIDENCE_REPLY` via `_const_generator` without calling the LLM.
- `CONFIDENCE_THRESHOLD = 0.15`, `TITLE_BOOST = 2.0` added to `config.py`.

**Issue 4 — Source relevance (`app/retriever.py`)**
- FAISS distances captured; index uses `METRIC_INNER_PRODUCT` so distances ARE
  cosine similarities directly — stored as `score = max(0.0, d)` (no conversion needed).
- Each article dict now includes `"score": float` (cosine similarity, 0–1).
- `_title_rerank()` added: post-retrieval rerank that boosts articles whose
  title words overlap with the query by up to `TITLE_BOOST` rank positions.
- Stopwords and short tokens excluded from query word set.

**Source link fix (`app/gui.py`)**
- Source buttons now open `https://simple.wikipedia.org/wiki/{url_slug}` in the
  default browser via `webbrowser.open()` instead of the local lead-only HTML file.
- `urllib.parse.quote(slug, safe=":")` applied for URL safety.

**Confidence gate bugfix (`app/retriever.py`, `app/config.py`) — commit 96591db**
- Score formula was wrong: `1.0 - d/2.0` is the squared-L2 conversion; the index
  uses `METRIC_INNER_PRODUCT` so the correct formula is just `max(0.0, d)`.
- The wrong formula inflated bad-match scores toward 1.0, making the gate never fire.
- `CONFIDENCE_THRESHOLD` lowered from `0.35` → `0.15` to account for IVF-PQ
  quantization error shrinking inner products below exact cosine similarity.

## Next Action When Resuming

POC fixes complete and pushed. Model swap (Issue 1) still pending for full-Wikipedia build:
- Replace `phi-3-mini-q4_k_m.gguf` with `gemma-2-2b-q4_k_m.gguf`
- Update `config.MODEL_PATH` to match new filename
- Update `wiki-offline.spec` model filename reference

To build the Windows installer:
  1. Complete the 4-step pre-build checklist in `wiki-offline.spec`
  2. `pip install pyinstaller && pyinstaller wiki-offline.spec`
  3. Ship `dist/WikiOffline/` to end users.
