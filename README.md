# Wiki Offline

A fully offline Windows desktop application for querying Simple English Wikipedia
using a local LLM. Zero internet required after installation.

> **Architecture**: RAG pipeline — user questions are embedded, matched against
> a FAISS vector index of Wikipedia article leads, and the retrieved context is
> passed to a local Phi-3 Mini model to generate a conversational answer.

---

## Prerequisites

- Python 3.11+
- Windows 10/11 (build machine; end users also need Windows)
- NVIDIA GPU with CUDA 12.x recommended for the embedding step (step 04)
- ~15 GB free disk space during the build

---

## Part 1 — Build Phase (Developer Machine)

Run these steps once to prepare the data artefacts. All scripts live in `build/`.

### 1.1 Create a virtual environment

```bat
python -m venv .venv
.venv\Scripts\activate
```

### 1.2 Install PyTorch with CUDA

Replace `cu121` with your CUDA version (`cu118`, `cu124`, etc.).
Check https://pytorch.org/get-started/locally/ for the correct URL.

```bat
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

To verify:

```python
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 1.3 Install build dependencies

```bat
pip install -r build/requirements_build.txt
```

### 1.4 Run the pipeline scripts in order

Each script prints progress and tells you the next step on completion.

**Step 01 — Download the Wikipedia dump (~2–3 GB)**

```bat
python build/01_download_wiki.py
```

The file is saved to `raw/simplewiki-YYYYMMDD-cirrussearch-content.json.gz`.
Re-running is safe — skips the download if the file already exists.

**Step 02 — Parse articles into a staging file**

```bat
python build/02_parse_articles.py
```

Streams through the compressed dump, filters redirects and disambiguation pages,
extracts title / lead / body. Output: `raw/articles_parsed.jsonl` (~230,000 articles).

**Step 03 — Build SQLite database and HTML files**

```bat
python build/03_build_sqlite.py
```

Creates `data/wikipedia.db` and one HTML file per article in `data/articles/`.
The HTML files are self-contained (inline CSS) and open in any offline browser.

**Step 04 — Embed leads and build FAISS index**

```bat
python build/04_embed_and_index.py
```

Embeds all article leads with `all-MiniLM-L6-v2` on GPU, builds a FAISS
IndexIVFPQ, and saves `data/wikipedia.faiss` and `data/id_map.json`.

To force CPU (slower, ~2–3 hours vs ~15 minutes on GPU):

```bat
python build/04_embed_and_index.py --device cpu
```

---

## Part 2 — Download the Language Model

The LLM is **not** bundled in this repository. Download it manually:

1. Go to: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
2. Download: `Phi-3-mini-4k-instruct-q4.gguf`
   *(alternatively any `phi-3-mini-*-q4_k_m.gguf` build)*
3. Rename it to `phi-3-mini-q4_k_m.gguf`
4. Place it in the `models/` directory:

```
wiki-offline/
└── models/
    └── phi-3-mini-q4_k_m.gguf   ← here
```

File size is approximately 2.2 GB.

> **Note for full Wikipedia build:** This model was selected for POC validation.
> Before the final build targeting 8 GB RAM laptops, revisit the model choice —
> consider **Gemma 2 2B q4\_k\_m** (~1.7 GB) or **Phi-3 Mini q4\_0** (~2 GB) for
> better performance on memory-constrained hardware. The RAG pipeline grounds
> answers in retrieved context, so a smaller model involves minimal quality
> tradeoff.

---

## Part 3 — Run the Application (Development Mode)

```bat
pip install -r app/requirements_app.txt
python app/main.py
```

The app starts a local Gradio server at http://127.0.0.1:7860 and opens your
browser automatically.

---

## Windows Note — llama-cpp-python Pre-built Wheel

`llama-cpp-python` compiles its C++ backend from source during `pip install`.
On Windows this requires Visual Studio Build Tools with the **Desktop development
with C++** workload installed. For most users it is easier to install a
pre-built binary wheel directly from the project's GitHub releases page.

Replace `cp311` with your Python minor version (`cp310`, `cp312`, etc.) if
needed. Check https://github.com/abetlen/llama-cpp-python/releases for the
full list of available wheels.

```bat
pip install "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.16/llama_cpp_python-0.3.16-cp311-cp311-win_amd64.whl"
```

Install this **before** running `pip install -r app/requirements_app.txt` so
the requirements file does not trigger a source build.

---

## Part 4 — Build the Windows Distributable

### 4.1 Install PyInstaller

```bat
pip install pyinstaller
```

### 4.2 Run the spec file

```bat
pyinstaller wiki-offline.spec
```

The output is a `WikiOffline/` folder containing `WikiOffline.exe`.
Distribute the entire `WikiOffline/` folder — do not distribute just the `.exe`.

### 4.3 What's bundled

| Included | Excluded |
|----------|---------|
| `app/` Python source | `build/` scripts |
| `data/wikipedia.db` | `raw/` scratch files |
| `data/wikipedia.faiss` | Source `.json.gz` dump |
| `data/id_map.json` | Developer venv |
| `data/articles/*.html` | |
| `models/*.gguf` | |
| Gradio offline assets | |
| `all-MiniLM-L6-v2` model files | |

---

## Tuneable Parameters

All configuration constants are in `app/config.py`. Edit that file to change:

| Constant | Default | Description |
|----------|---------|-------------|
| `TOP_K` | `5` | Number of articles retrieved per query |
| `CHAT_HISTORY_TURNS` | `3` | Past exchanges included in LLM prompt |
| `N_CTX` | `4096` | LLM context window (tokens) |
| `N_THREADS` | `cpu_count - 1` | CPU threads for inference |
| `N_GPU_LAYERS` | `0` | GPU layers (0 = CPU-only for end users) |

---

## Data Notes

- Source: [Simple English Wikipedia CirrusSearch dump](https://dumps.wikimedia.org/other/cirrussearch/current/)
- Dump format: gzip-compressed NDJSON (alternating action / document line pairs)
- ~230,000 main-namespace articles after filtering
- Lead paragraphs capped at 300 words for embedding efficiency
