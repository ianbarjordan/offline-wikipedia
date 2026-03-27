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
    ├─ index.search() → top-k FAISS positions + inner-product scores
    ├─ score = max(0.0, d)  [index uses METRIC_INNER_PRODUCT]
    ├─ id_map[pos] → SQLite ids
    ├─ SELECT title, lead, url_slug FROM articles WHERE id IN (...)
    ├─ attach score to each article dict
    └─ _title_rerank(): boost articles whose title words match query
    │
    ▼
pipeline.query(user_message, chat_history)
    ├─ low_confidence = (not articles) or (articles[0]["score"] < CONFIDENCE_THRESHOLD)
    ├─ if low_confidence AND no articles → return canned reply, []
    ├─ if low_confidence AND articles present:
    │    ├─ build prompt with _GROUNDING_LOW (relaxed: "primary source")
    │    ├─ llm.generate(prompt, stream=True)
    │    └─ return _prepend_generator(disclaimer, stream), articles
    └─ if high confidence:
         ├─ build prompt with _GROUNDING_HIGH (strict: "exclusively from context")
         └─ return stream, articles
    │
    ▼
gui.py streaming callback → Gradio Chatbot
    └─ source buttons → _open_file(data/articles/{id}.html)  [local HTML]
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
- `MAX_DISPLAY_SOURCES = 3` — max source buttons shown in GUI per response
- `MAX_LLM_CONTEXT_SOURCES = 3` — articles sliced into the LLM prompt (retrieval pool stays TOP_K)
- `MAX_NEW_TOKENS = 400` — max tokens the LLM may generate per response (~300 words)
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
- **Reverted in UI quality pass (see below):** buttons now correctly open local HTML.

**Confidence gate bugfix (`app/retriever.py`, `app/config.py`) — commit 96591db**
- Score formula was wrong: `1.0 - d/2.0` is the squared-L2 conversion; the index
  uses `METRIC_INNER_PRODUCT` so the correct formula is just `max(0.0, d)`.
- The wrong formula inflated bad-match scores toward 1.0, making the gate never fire.
- `CONFIDENCE_THRESHOLD` lowered from `0.35` → `0.15` to account for IVF-PQ
  quantization error shrinking inner products below exact cosine similarity.

**Tiered confidence gate + grounding-aware prompts (`app/pipeline.py`) — commit 3bf1565**

_What changed:_
- `_prepend_generator(prefix, gen)` added at module level — yields a prefix string
  then delegates to an existing generator, enabling disclaimer injection without
  buffering the LLM stream.
- `_GROUNDING_HIGH` and `_GROUNDING_LOW` module-level string constants encode the
  two distinct grounding instructions:
  - HIGH (default): "Answer exclusively from the Wikipedia context below. Do NOT use
    your training knowledge. If the context does not contain the answer, say you
    don't know."
  - LOW: "Answer using the Wikipedia context below as your primary source. If the
    context is insufficient, say so briefly before answering."
- `_build_prompt` gains `low_confidence: bool = False` parameter; selects
  `_GROUNDING_LOW` or `_GROUNDING_HIGH` accordingly. `_SYSTEM_TEMPLATE` is now
  unused (kept as a historical artefact but no longer called).
- `query()` confidence gate replaced with three-way logic:
  1. No articles at all → `_const_generator(_LOW_CONFIDENCE_REPLY), []`
  2. Articles present but score < threshold → LLM called with LOW grounding, stream
     wrapped via `_prepend_generator` with disclaimer prefix, articles returned
  3. Score ≥ threshold → LLM called with HIGH grounding, stream returned directly

_Why:_ The old binary gate silently discarded low-quality-but-non-empty results.
The new logic lets the LLM attempt an answer while signalling reduced confidence
to the user, matching real-world query behaviour where partial context is better
than no answer.

_Smoke test result:_ `scratch/smoke_test_e2e.py` — **44/44 checks passed** (9.9 s).

---

**Score-based title rerank + TOP_K increase (`app/retriever.py`, `app/config.py`) — commit e809e72**

_Problem:_ `_title_rerank` was rank-based (`rank - TITLE_BOOST * overlap`). For
"Who is George Washington," both the city and person articles had `overlap=1.0` but
the city at FAISS rank 0 permanently beat the person at rank 2. Also, `str.split()`
on "George, Washington" produced `"washington,"` as a token, causing false full-overlap.

_Changes:_
- Sort key changed to `-(score + TITLE_BOOST * overlap * length_ratio)` — score-based,
  not rank-based.
- `length_ratio = len(q_words) / max(len(title_tokens), len(q_words))` — penalises
  qualified titles ("George Washington, Washington", 3 tokens) vs exact matches
  ("George Washington", 2 tokens) for a 2-word query.
- Word tokenisation switched from `str.split()` to `_WORD_RE = re.compile(r'\b\w+\b')`
  to strip punctuation cleanly.
- `TOP_K` increased `5 → 8` for a wider FAISS candidate pool.

**Parser redirect-filter bug fix (`build/02_parse_articles.py`) — commit e0bb281**

_Problem:_ The CirrusSearch `redirect` field on a content article lists pages that
redirect TO it (incoming aliases). The parser treated any non-empty `redirect` as
"this page IS a redirect" and skipped it — silently dropping ~62k articles including
George Washington, Bubonic plague, Fishing, and any popular article with alternate names.

_Fix:_ Removed the `if doc.get("redirect"):` filter entirely. True redirect pages
have no `opening_text` or `text` and are already caught by the existing `no_text` check.

_Result:_ 274,704 articles parsed (was 212,884) — 61,820 articles recovered.
"Who is George Washington" now correctly returns the president article at rank 1 (score=0.900).

---

**SQL title-search supplement (`app/retriever.py`) — commit 2df2b8e**

_Problem:_ After the reranking fix, the LLM correctly said "no info about George
Washington the president" — because the article was never in the FAISS top-8 at all.
`all-MiniLM-L6-v2` embeds "Who is George Washington" close to "X is a Y" sentence
structures, so city lead paragraphs ("George, Washington is a city…") score higher
than the biographical president article.

_Changes (all in `retriever.search()`, after FAISS results are assembled):_
- Extract `q_ordered` (key words, same stopword/length filter as `_title_rerank`).
- For queries with ≤ 4 key-words, run a supplementary SQL lookup:
  1. `WHERE LOWER(title) = ?` exact match (zero false positives)
  2. Fallback to `LIKE`-per-word if exact adds nothing new
- Articles found by SQL but absent from FAISS are appended with
  `score = _TITLE_EXACT_SCORE = 0.9`, ensuring `_title_rerank` promotes them
  above any city article (city score ~0.72 + boost < president 0.9 + boost).
- `_MAX_TITLE_SUPPLEMENT = 3` caps context growth.

_Net effect for "Who is George Washington":_
- FAISS returns city articles; SQL injects "George Washington" with score=0.9
- `_title_rerank` combined scores: president=2.90, city=2.72 → president first
- LLM receives the correct article and answers about the founding father

_Smoke test:_ 44/44 passed. SQL supplement silently adds nothing for the 30-article
synthetic set (no "George Washington" article exists there).

---

**SQL title supplement LIKE word-boundary fix (`app/retriever.py`) — this session**

_Problem:_ The LIKE-per-word fallback in the SQL title supplement used `%word%` substring
matching. For a "speed of light" query, `q_ordered = ["speed", "light"]`, the query
`LIKE '%speed%' AND LIKE '%light%'` matched "Lightspeed (company)", "Lightspeed (magazine)",
and "Power Rangers Lightspeed Rescue" — all injected with score=0.9, polluting displayed
sources.

_Fix:_ After fetching LIKE candidates, apply a Python word-token post-filter using
`_WORD_RE.findall()` to confirm each query word appears as a complete token in the title
(not as a substring of a compound word). Candidates that don't pass are dropped before
appending to the article pool. Increased the initial LIKE fetch limit to `_MAX_TITLE_SUPPLEMENT * 4`
to compensate for post-filtering attrition.

_Result:_ "Speed of light" correctly appears without "Lightspeed Rescue" noise.
Entity queries (George Washington, FIFA World Cup, Napoleon) unaffected — their exact
title tokens already passed the filter.

_Smoke test:_ 44/44 passed.

---

**UI/Quality pass — four fixes — commit (this session)**

1. **New system prompt (`app/pipeline.py`)** — `_SYSTEM_TEMPLATE` restructured to
   embed `{grounding}` inline and add explicit plain-prose formatting rules ("no markdown
   headers or bullet points unless listing 3+ items", "don't say 'According to Wikipedia'",
   etc.). `_GROUNDING_HIGH`/`_GROUNDING_LOW` shortened to single-line strings.
   `_build_prompt` now uses `_SYSTEM_TEMPLATE.format(grounding=..., context=...)`.

2. **Display source cap (`app/pipeline.py`, `app/config.py`)** — `MAX_DISPLAY_SOURCES=3`
   added to config. `pipeline.query()` trims to `[:MAX_DISPLAY_SOURCES]` for GUI source buttons.

3. **Source buttons open local HTML (`app/gui.py`)** — `open_article()` now calls
   `_open_file(config.ARTICLES_DIR / f"{art['id']}.html")` instead of
   `webbrowser.open(simple.wikipedia.org/wiki/{slug})`. Removed `import webbrowser`
   and `import urllib.parse` (no longer used).

4. **Lead noise filter (`build/02_parse_articles.py`)** — `_LEAD_NOISE_RE` regex and
   `clean_lead()` function added. Strips sentences containing Wikipedia maintenance
   banners ("Written by: [Your Name]", "This article needs…", "You can help Wikipedia",
   etc.) from lead text before writing to staging JSONL. `extract_lead()` applies
   `clean_lead()` after truncation. Requires data rebuild (02→03→04) to take effect.

_Smoke test:_ 44/44 passed.

---

**Context window overflow fix (`app/config.py`, `app/pipeline.py`) — this session**

_Problem:_ With `TOP_K=8` articles (~300-word leads each) + 3-turn chat history + system
template, the input prompt consumed ~4,057 of 4,096 tokens, leaving almost no headroom for
generation. This caused: LLM writing 5-paragraph essays (no generation cap), a
`[{'text':…,'type':'text'}]` literal artifact at turn 3 (model confused by near-full context),
and garbled output at turn 4+ (prompt exceeded window).

_Changes:_
- `MAX_LLM_CONTEXT_SOURCES = 3` added to `config.py` — only top 3 articles are embedded in
  the LLM prompt. Retrieval pool stays at `TOP_K=8` for rerank quality.
- `MAX_NEW_TOKENS = 300` added to `config.py` — caps LLM generation at ~220 words per response.
- `pipeline.query()` now calls `_build_context(articles[:config.MAX_LLM_CONTEXT_SOURCES])`.
- `self._llm.generate(prompt, stream=True, max_tokens=config.MAX_NEW_TOKENS)` passes the cap
  through the existing `**kwargs` forwarding in `llm.py` — no change to `llm.py` required.

_Token budget after fix:_ ~2,030 tokens input → ~2,066 tokens available for generation at any
turn depth. With `MAX_NEW_TOKENS=300`, well within budget across 3-turn history.

_Smoke test:_ 44/44 passed.

---

## Current File State

| File | Status | Notes |
|------|--------|-------|
| `app/config.py` | Done | `CONFIDENCE_THRESHOLD=0.15`, `TITLE_BOOST=2.0`, `TOP_K=8`, `MAX_DISPLAY_SOURCES=3`, `MAX_LLM_CONTEXT_SOURCES=3`, `MAX_NEW_TOKENS=400` |
| `app/retriever.py` | Done | Score-based rerank; SQL title supplement (cap ≤6) with whole-word post-filter; nickname expansion (`_NICKNAMES`); state abbreviation expansion (`_STATE_ABBREVS`, uppercase-only); expanded `_STOPWORDS` with preamble verbs |
| `app/llm.py` | Done | Defensive `.get()` stream access; `llama-cpp-python 0.3.16` |
| `app/pipeline.py` | Done | Three-way confidence gate; `_SYSTEM_TEMPLATE`; display source cap; LLM context sliced to `MAX_LLM_CONTEXT_SOURCES`; generation capped at `MAX_NEW_TOKENS`; injection detection (`_INJECTION_RE`, zero-or-more modifier groups); meta-reply handler (`_META_RE`/`_META_REPLY`); expanded conversational handler; `_is_conversational_reaction()` with `_YES_NO_STARTERS` guard; `_truncation_guard()` appends `" [...]"` on mid-sentence cutoff; query augmentation (`_augment_query`, extracts keywords from assistant response first, falls back to user query, broadened to entity-less follow-ups, cap 12 words); `_CANNED_REPLIES` frozenset; stronger grounding (fictional characters); context always injected into current turn |
| `app/gui.py` | Done | `chat_pairs_state` (gr.State) prevents browser serialization of history; source buttons open local `data/articles/{id}.html` via `_open_file()` |
| `app/main.py` | Done | Startup sequence, pre-flight checks, argparse; `--gpu-layers N` flag for dev GPU testing |
| `build/02_parse_articles.py` | Done | `clean_lead()` strips maintenance-banner sentences from leads; `normalise_body_whitespace()` preserves paragraph breaks in body |
| `build/03_build_sqlite.py` | Done | `body_to_html()` renders body paragraphs as `<p>`/`<h2>` tags; CSS updated for formatted body |
| `build/01,04` | Done | All other build scripts written and smoke-tested |
| `wiki-offline.spec` | Done | PyInstaller onedir spec |

---

## Known Issues / Limitations

- **`llama_cpp` must be present** for `pipeline.py` to import (top-level via `llm.py`).
  For CI/test environments, install `llama-cpp-python` even when no GGUF is available.
- **`embeddings.position_ids UNEXPECTED`** warning at Retriever load — benign,
  expected for `all-MiniLM-L6-v2` with sentence-transformers 5.x.
- None outstanding.

---

## Production Data State

Full Simple English Wikipedia data rebuilt (March 2026 dump) with `clean_lead()` noise filter:
- `data/wikipedia.db` — 274,630 articles, 616.5 MB
- `data/wikipedia.faiss` — IVF-PQ index, 8.2 MB
- `data/id_map.json` — 4 MB (274,630 entries)

Rebuild completed March 22 2026. Embedded on RTX 3060 Ti (CUDA 13.1, torch 2.10.0+cu128).

The smoke test writes to `scratch/smoke_data/` (isolated from production):
- `scratch/smoke_data/smoke.db`
- `scratch/smoke_data/smoke.faiss`
- `scratch/smoke_data/smoke_id_map.json`
- `scratch/smoke_data/articles/`

**Smoke test result (March 22 2026): 44/44 checks passed — ALL PASSED (9.3s)**

---

## Smoke Test Inventory

| File | Checks | Focus |
|------|--------|-------|
| `scratch/smoke_test_e2e.py` | 44 | Full build pipeline + Gradio UI integration |
| `scratch/smoke_edge_cases.py` | 43 | Pure-logic unit tests for individual pipeline functions |
| `scratch/smoke_pipeline.py` | 38 | Pipeline routing, confidence gate, truncation guard (e2e), query augmentation, multi-turn respond() accumulation, conversation flows |

`scratch/smoke_pipeline.py` uses `MockRetriever` + `MockLLM` only — no data files, FAISS index, or model required. Run time &lt;5 seconds.

**Result (March 27 2026): 38/38 checks passed — ALL PASSED**

**HTML article formatting fix (March 22 2026):**
- `build/02_parse_articles.py`: Added `normalise_body_whitespace()` — preserves `\n\n` paragraph breaks in body (leads still fully collapsed via `normalise_whitespace()`).
- `build/03_build_sqlite.py`: Added `body_to_html()` — splits body on `\n\n`, wraps each paragraph in `<p>` or `<h2>` (heuristic: ≤60 chars, no sentence-ending punctuation). CSS updated: removed `white-space: pre-wrap`, added `.body-text p` and `.body-text h2` styles.
- Data rebuilt: 274,630 articles. Smoke test: 44/44 passed. FAISS untouched.

---

## Chat History Format Fix — gr.State for chat pairs (March 22 2026)

_Problem:_ Starting at turn 3, LLM responses appeared as `[{'text': "...", 'type': 'text'}]`
instead of plain text. By turn 6 this double-nested. Root cause: Gradio 6.7 serializes
`ChatMessage.content` through the browser as a list of content blocks. `_to_pairs()` in
`gui.py` extracted this list as-is; `_build_prompt()` formatted it via f-string, producing
the literal dict repr. Gemma 2 saw this in context and mimicked the format — a feedback loop.

_Fix (`app/gui.py`):_
- Added `chat_pairs_state = gr.State([])` — lives server-side only, never round-trips
  through the browser, so content always remains a plain Python string.
- `respond()` now accepts `chat_pairs: list` as 3rd parameter and uses it directly
  (no `_to_pairs()` call). Accumulates `response_text` during streaming; appends
  `(message, response_text)` to produce `new_chat_pairs` after the stream ends.
- All yield tuples, `_noop()`, `clear_conversation()`, and event wiring updated to
  include `chat_pairs_state`.
- `_to_pairs()` kept (used in `gui.py` smoke test).

_Fix (`scratch/smoke_test_e2e.py`):_
- Stage 8: call updated to `respond_fn(question, history, [])`, index refs shifted
  (`articles` at `[3]`, `src_row` at `[4]`).
- Stage 9: `cleared_articles` at `[2]`, `row_update` at `[3]`.

_Verification:_ 44/44 smoke tests passed. 6-turn live conversation confirmed zero artifacts.

---

## Model Swap — Gemma 2 2B (completed March 22 2026)

Replaced Phi-3 Mini with Gemma 2 2B IT (Q4_K_M):
- `models/gemma-2-2b-q4_k_m.gguf` — 1.71 GB, downloaded from bartowski/gemma-2-2b-it-GGUF
- `app/config.py` — `MODEL_PATH` updated
- `app/pipeline.py` — `_build_prompt` rewritten for Gemma 2 chat template:
  - No `<|system|>` token; system content prepended to first user turn
  - Format: `<start_of_turn>user\n{system}\n\n{msg}<end_of_turn>\n<start_of_turn>model\n`
  - History turns: `<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>\n`
- `wiki-offline.spec` — model filename and HF source URL updated

## Live Testing Fixes — Grounding + UX (March 26 2026)

Three issues observed in a live 12-turn test conversation:

**Issue 1 — Greeting leakage**: "Hello. What can you do?" triggered FAISS retrieval,
which returned an article called "Hello I Must Be Going". The model obediently described
the retrieved articles instead of explaining its capabilities.

_Fix_: Added `_CONVERSATIONAL_RE` regex in `pipeline.py`. Greetings and meta-questions
(hello, hi, what can you do, who are you, etc.) are detected before retrieval and return
a canned `_GREETING_REPLY` immediately — no FAISS call, no LLM call.

**Issue 2 — Training data leakage**: Gemma 2 ignored grounding instructions for
well-known facts (Trump, Samsung, Slovenia, etc.), answering entirely from parametric
memory while displaying unrelated Wikipedia sources.

_Fix_: `_GROUNDING_HIGH` and `_GROUNDING_LOW` rewritten to be explicit:
- "Do NOT use your training knowledge — not even for famous people, current events,
  or well-known facts."
- Mandates exact fallback phrase: "My Wikipedia database doesn't cover that topic."
- System template `_SYSTEM_TEMPLATE` updated to repeat the rule at the top and in
  the format instructions.

**Issue 3 — System prompt position in multi-turn**: With chat history, the system
prompt (with fresh Wikipedia context) was injected into the *oldest* historical turn.
By turn 3+, the model treated it as stale background, weakening grounding.

_Fix_: `_build_prompt()` restructured — historical turns are now plain user/model
exchanges with no system text. The system prompt + fresh context is always injected
into the **current** user turn, immediately adjacent to generation. Same token budget,
stronger signal.

**Also added**: `--gpu-layers N` flag to `main.py` for dev testing on GPU without
changing `config.py`. `dist/WikiOffline.exe` always defaults to CPU (`N_GPU_LAYERS=0`).

---

## Conversation Quality Fixes — 7 Issues (March 26 2026)

Live testing revealed 7 failure modes. All fixes are in `app/pipeline.py` and
`app/retriever.py` only. Smoke test: **44/44 passed** after all changes.

**Fix 1 — Prompt Injection Detection (`app/pipeline.py`)**
Added `_INJECTION_RE` (compiled regex) that catches "ignore your previous instructions",
"act as", "pretend to be", "jailbreak", `DAN`, etc. Checked via `.search()` before any
other handler — returns `_INJECTION_REPLY` with no retrieval, no LLM call.

**Fix 2 — Stronger Grounding (`app/pipeline.py`)**
- `_SYSTEM_TEMPLATE` gains an explicit rule: "Do not add details about fictional
  characters, movies, TV shows, or celebrities not stated in the Wikipedia articles."
- `_GROUNDING_HIGH` extended: adds "fictional characters" to the prohibited sources list
  and "Copy facts directly from the articles. Do not infer or extrapolate."

**Fix 3 — Expanded Conversational Handler (`app/pipeline.py`)**
- `_CONVERSATIONAL_RE` broadened to cover reactions (`Sweet!`, `Wow!`, `Interesting!`,
  `Lol`, `Nice!`, `Amazing!`), affirmatives (`Really?`, `Are you sure?`), and identity
  questions (`Where are you from?`, `Who made you?`, `Are you an AI?`).
- `_META_RE` + `_META_REPLY` added: identity questions return a dedicated reply explaining
  the app is a local Wikipedia assistant — checked before `_CONVERSATIONAL_RE`.
- Handler order in `query()`: injection → meta → conversational → retrieval.

**Fix 4 & 5 — Query Augmentation for Follow-ups (`app/pipeline.py`)**
Added `_augment_query(user_message, chat_history)`: if the query is ≤ 8 words AND
contains a pronoun (`he`, `it`, `they`, `this`, etc.), up to 4 key words from the
previous user turn are prepended to form `retrieval_query`. The LLM always receives
the original `user_message`. Resolves "Why did he do that?" (after Jefferson) and
"Is it also a state?" (after Virginia).

**Fix 6 — Nickname Expansion (`app/retriever.py`)**
Added `_NICKNAMES` dict (`tom→thomas`, `bob→robert`, `bill→william`, etc.).
Applied to `q_ordered` in the SQL title supplement before building the candidate title,
so "tom jefferson" finds the "Thomas Jefferson" article.

**Fix 7 — State Abbreviation Expansion (`app/retriever.py`)**
Added `_STATE_ABBREVS` dict (all 50 states + DC). Applied to query tokens at the top
of `search()` before embedding, so "Where is va?" expands to "Where is Virginia?" and
retrieves the state article rather than a small Illinois city.

---

## Windows Build — Option A (Native)

PyInstaller runs on the Windows host (not WSL). Steps after `git pull`:
```
mkdir models
curl -L -o models\gemma-2-2b-q4_k_m.gguf https://github.com/ianbarjordan/offline-wikipedia/releases/download/v1-data/gemma-2-2b-q4_k_m.gguf
python -m PyInstaller wiki-offline.spec
```
Data files (`wikipedia.db`, `wikipedia.faiss`, `id_map.json`, `articles/`) must already
be present. Embedding model must be pre-cached via sentence-transformers download.

GitHub Actions Option B workflow exists (`.github/workflows/build-windows.yml`) but
was paused due to YAML/extraction issues — to be revisited later.

Data release `v1-data` on GitHub holds all large assets for re-download.

---

## Query Formulation Overhaul — 8 Failure Modes Fixed (March 26 2026)

A second live test revealed that the retriever's SQL title supplement was silently
failing for natural-language phrasing, and that the augmentation path was too narrow.
All changes in `app/pipeline.py` and `app/retriever.py`. Smoke test: **44/44 passed**.
New edge-case suite: `scratch/smoke_edge_cases.py` — **22/22 passed** (Round 3 extended to **43/43**).

**Fix A — Stopword expansion (`app/retriever.py`)**
Expanded `_STOPWORDS` with ~25 preamble/filler words (`tell`, `want`, `know`,
`explain`, `describe`, `show`, `give`, `find`, `need`, `like`, `make`, `use`,
`say`, `let`, `please`, `can`, `could`, `would`, `get`, `look`, `also`, `have`,
`has`, `had`, `do`, `did`, `does`, `be`, `been`, `more`, `some`, `just`, `really`,
`very`, `think`, `see`).
Effect: "Tell me about ASUS" → `q_ordered = ['asus']` → SQL exact-match finds Asus article.
"I want to know about the most populated US states" → `q_ordered = ['most', 'populated', 'states']` → 3 tokens, SQL runs.

**Fix B — SQL token cap raised to 6 (`app/retriever.py`)**
Cap on SQL title supplement changed from `len(q_ordered) <= 4` to `<= 6`. After
stopword expansion strips preamble words, most queries have fewer tokens. Moderately
complex queries like "South Korean Army members count" (4 tokens after stripping)
now still benefit from the SQL boost.

**Fix C — Short-message reaction catch-all (`app/pipeline.py`)**
Added `_is_conversational_reaction(message)`: returns True when ≤5 words, no
capitalized content word after position 0 (not a named entity), no question word.
Called after `_CONVERSATIONAL_RE.match()`. Catches reactions the regex missed due
to its `\W*$` anchor:
- "Okay, cool" → conversational ✓
- "Only 77k?!" → conversational ✓
- "Are you sure its not 700k?" → 6 words → falls through to augmentation ✓
- "Tell me about ASUS" → has uppercase "ASUS" → real query ✓

**Fix D — Broadened augmentation trigger (`app/pipeline.py`)**
`_augment_query` now triggers on two conditions (OR'd):
- (a) query contains a pronoun — existing logic
- (b) query is ≤6 words AND has no proper noun after position 0 (entity-less follow-up)
Word cap raised from 8 → 12 to cover longer pronoun-containing follow-ups.
Effect: "How many members?" (after South Korean Army context) → augments → correct retrieval.
"Are you sure its not 700k?" (after Alaska) → 6 words, no proper noun → augments with Alaska context.

**Bug fix — State abbreviation expander corrupting natural language (`app/retriever.py`)**
The expander was lowercasing all tokens before dict lookup, so common words like
"me" (→ Maine), "in" (→ Indiana), "or" (→ Oregon), "de" (→ Delaware) were silently
expanded. "Tell me about ASUS" became "Tell Maine about ASUS", and after Fix A's
stopword stripping, `q_ordered = ['maine', 'asus']` → SQL searched "maine asus" → no match.
Fix: only expand tokens where the **original** token is all-uppercase and exactly 2
characters (e.g., `"ME"`, `"AK"`, `"CA"`). Lowercase common words are left unchanged.

---

## Round 3 Fixes — Live Transcript Issues (`app/pipeline.py`, `app/config.py`)

**Fix 1 — Injection regex repaired (`_INJECTION_RE`)**
- Previous regex allowed only ONE optional modifier word before the target noun.
- `"Ignore all previous instructions"` (two modifiers: `all` + `previous`) slipped through.
- Fix: replaced `(your\s+)?(previous|...)?` with `(?:(?:your|previous|...)\s+)*` (zero-or-more).
- Now catches any chain of modifier words.

**Fix 2 — Yes/no question guard in `_is_conversational_reaction`**
- Added `_YES_NO_STARTERS` frozenset (`is`, `are`, `was`, `were`, `do`, etc.).
- When a message starts with a yes/no verb, check the subject: if it is NOT `"you"`, the
  message is a real query (e.g. `"Are red pandas pandas?"`, `"Was he a real person?"`).
- `"Are you sure?"` still correctly returns `True` (subject is `"you"`).

**Fix 3 — Token cap raised + truncation detection**
- `MAX_NEW_TOKENS` raised from 300 → 400 (`config.py`).
- Added `_truncation_guard(gen)` wrapper in `pipeline.py`: buffers all tokens; if the full
  response doesn't end with `.`, `!`, `?`, `"`, or `'`, appends `" [...]"`.
- Applied to every LLM call in `query()`.

**Fix 4 — Augmentation from assistant response, not user query**
- `_augment_query` now extracts keywords from the **previous assistant response** (first
  sentence only) so the actual entity named in the answer (e.g. `"bluebonnet"`) is included.
- Falls back to user query keywords if the assistant gave a canned/empty reply.
- `_CANNED_REPLIES` frozenset added to detect canned replies reliably.

---

## Next Steps When Resuming

1. **NSIS/Inno Setup installer**: wrap `dist\WikiOffline\` in a Setup.exe with Start
   Menu shortcut and Add/Remove Programs entry
2. **Revisit GitHub Actions build** (Option B) if native build proves inconvenient

## Environment Notes

- GPU: NVIDIA GeForce RTX 3060 Ti, 8GB VRAM, CUDA 13.1
- PyTorch: 2.10.0+cu128 (installed system-wide via `--break-system-packages`)
- llama-cpp-python: 0.3.16 (CPU build — no GPU inference for LLM)
- Build deps: sentence-transformers 3.4.1, faiss-cpu 1.13.2, gradio 6.7.0
