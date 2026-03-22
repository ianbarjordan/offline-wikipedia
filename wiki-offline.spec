# wiki-offline.spec
# PyInstaller spec — produces a --onedir bundle for Windows.
#
# ============================================================
# PRE-BUILD CHECKLIST  (run every step before pyinstaller)
# ============================================================
# 1. Complete the full data build pipeline:
#       python build/01_download_wiki.py
#       python build/02_parse_articles.py
#       python build/03_build_sqlite.py
#       python build/04_embed_and_index.py
#    Required output: data/wikipedia.db, data/wikipedia.faiss,
#                     data/id_map.json, data/articles/
#
# 2. Download the LLM model (manual):
#       Place gemma-2-2b-q4_k_m.gguf in models/
#    Source: https://huggingface.co/bartowski/gemma-2-2b-it-GGUF
#
# 3. Pre-cache the embedding model so it can be bundled offline:
#       python -c "from sentence_transformers import SentenceTransformer; \
#                  SentenceTransformer('all-MiniLM-L6-v2')"
#    This populates ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/
#
# 4. Build:
#       pip install pyinstaller
#       pyinstaller wiki-offline.spec
#    Output: dist/WikiOffline/   (ship this folder to end users)
# ============================================================

import importlib.util
import pathlib
import glob

# ---------------------------------------------------------------------------
# Helper: locate an installed package's directory portably.
# ---------------------------------------------------------------------------

def _pkg_dir(name: str) -> pathlib.Path:
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise ImportError(f"Package '{name}' not found. Run: pip install {name}")
    origin = spec.origin or (spec.submodule_search_locations[0] if spec.submodule_search_locations else None)
    if origin is None:
        raise ImportError(f"Cannot locate directory for '{name}'")
    return pathlib.Path(origin).parent


# ---------------------------------------------------------------------------
# Locate installed packages
# ---------------------------------------------------------------------------

GRADIO_DIR       = _pkg_dir("gradio")
GRADIO_CLIENT_DIR = _pkg_dir("gradio_client")
LLAMA_CPP_DIR    = _pkg_dir("llama_cpp")
FAISS_DIR        = _pkg_dir("faiss")
TOKENIZERS_DIR   = _pkg_dir("tokenizers")

# HuggingFace cache for the embedding model (all-MiniLM-L6-v2).
# Run step 3 of the pre-build checklist to populate this.
import os
_hf_home = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache" / "huggingface"))
_embed_model_glob = "models--sentence-transformers--all-MiniLM-L6-v2"
_embed_cache_dirs = list((_hf_home / "hub").glob(_embed_model_glob)) if (_hf_home / "hub").exists() else []

if not _embed_cache_dirs:
    import warnings
    warnings.warn(
        "\n\n"
        "  WARNING: all-MiniLM-L6-v2 not found in HuggingFace cache.\n"
        "  The frozen app will attempt to download it on first launch\n"
        "  (requires internet on the end-user machine).\n"
        "  Run step 3 of the pre-build checklist to avoid this.\n",
        stacklevel=0,
    )

# ---------------------------------------------------------------------------
# Datas: (source, bundle-destination) pairs
# ---------------------------------------------------------------------------

datas = []

# -- Gradio UI assets -------------------------------------------------------
# Gradio serves its frontend from these directories at runtime.
# Directories are always included; individual files are conditional because
# their presence varies across Gradio patch versions (e.g. hash_seed.txt
# was dropped in some builds).
datas += [
    (str(GRADIO_DIR / "templates"),          "gradio/templates"),
    (str(GRADIO_DIR / "_simple_templates"),  "gradio/_simple_templates"),
    (str(GRADIO_DIR / "icons"),              "gradio/icons"),
    (str(GRADIO_DIR / "media_assets"),       "gradio/media_assets"),
]
for _f in ["hash_seed.txt", "package.json"]:
    if (GRADIO_DIR / _f).exists():
        datas.append((str(GRADIO_DIR / _f), "gradio"))

# -- gradio_client ----------------------------------------------------------
datas += [
    (str(GRADIO_CLIENT_DIR / "package.json"), "gradio_client"),
]
for _f in ["types.json"]:
    if (GRADIO_CLIENT_DIR / _f).exists():
        datas.append((str(GRADIO_CLIENT_DIR / _f), "gradio_client"))

# -- llama_cpp shared libraries ---------------------------------------------
# llama_cpp resolves its .so/.dll files as:
#   Path(os.path.dirname(__file__)) / "lib"
# so they must land at llama_cpp/lib/ inside the bundle.
datas += [
    (str(LLAMA_CPP_DIR / "lib"), "llama_cpp/lib"),
]

# -- HuggingFace embedding model cache --------------------------------------
# Bundled to hf_cache/hub/models--.../ inside the bundle.
# hooks/runtime_env.py sets HF_HOME=sys._MEIPASS/hf_cache at startup so
# sentence-transformers and huggingface_hub find it without network access.
for _embed_dir in _embed_cache_dirs:
    datas += [
        (str(_embed_dir), f"hf_cache/hub/{_embed_dir.name}"),
    ]

# -- App data files ---------------------------------------------------------
# These are large; they live OUTSIDE the bundle root so end users can see
# them.  Use SPEC-relative paths (pathlib.Path(SPEC).parent gives the
# wiki-offline/ project root).
import pathlib as _pl
_root = _pl.Path(SPEC).parent          # wiki-offline/

# Verify data files exist at spec-run time and warn if they don't.
for _name, _hint in [
    ("data/wikipedia.db",     "Run build/03_build_sqlite.py"),
    ("data/wikipedia.faiss",  "Run build/04_embed_and_index.py"),
    ("data/id_map.json",      "Run build/04_embed_and_index.py"),
    ("models/gemma-2-2b-q4_k_m.gguf", "Download from HuggingFace"),
]:
    if not (_root / _name).exists():
        import warnings as _w
        _w.warn(f"\n  WARNING: {_name} not found → {_hint}", stacklevel=0)

datas += [
    (str(_root / "data"),   "data"),
    (str(_root / "models"), "models"),
]

# ---------------------------------------------------------------------------
# Hidden imports
# ---------------------------------------------------------------------------
# PyInstaller cannot statically discover every dynamically-imported module.
# List them explicitly here.

hiddenimports = [
    # uvicorn (used by Gradio's FastAPI server) loads these via string names
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.loops.asyncio",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.http.httptools_impl",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.wsproto_impl",
    "uvicorn.protocols.websockets.websockets_impl",
    # Gradio dynamic component loading
    "gradio.components",
    "gradio.components.chatbot",
    "gradio.components.textbox",
    "gradio.components.button",
    "gradio.components.markdown",
    "gradio.components.row",
    "gradio.components.column",
    "gradio.components.state",
    "gradio.themes",
    "gradio.themes.base",
    "gradio.themes.default",
    # llama_cpp sub-modules imported at runtime
    "llama_cpp.llama_cpp",
    "llama_cpp.llama",
    "llama_cpp.llama_chat_format",
    "llama_cpp.llama_grammar",
    "llama_cpp.llama_cache",
    "llama_cpp.llama_types",
    # faiss
    "faiss",
    "faiss.swigfaiss",
    # tokenizers (Rust extension, used by sentence-transformers)
    "tokenizers",
    # sentence-transformers
    "sentence_transformers",
    "sentence_transformers.models",
    # misc
    "sklearn.utils._cython_blas",
    "scipy.special._comb",
]

# ---------------------------------------------------------------------------
# PyInstaller Analysis
# ---------------------------------------------------------------------------

block_cipher = None

a = Analysis(
    ["app/main.py"],
    pathex=["app"],          # makes `import config`, `import retriever`, etc. work
    binaries=[],             # .so binaries are included via datas (see above)
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["hooks/runtime_env.py"],
    excludes=[
        # Exclude GPU-specific torch backends — we run CPU-only
        "torch.cuda",
        "torch.backends.cuda",
        "torch.backends.cudnn",
        # Test frameworks — not needed at runtime
        "pytest",
        "unittest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,   # --onedir: binaries go in COLLECT, not embedded in EXE
    name="WikiOffline",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,                # UPX-compress the EXE if UPX is installed (optional)
    console=True,            # Show the console window so users see startup progress
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[
        # Never UPX-compress .so files — it breaks dynamic loading
        "*.so",
        "*.so.*",
        "*.dll",
    ],
    name="WikiOffline",      # output directory: dist/WikiOffline/
)
