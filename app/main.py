"""
main.py

Entry point for the wiki-offline desktop application.

Startup sequence
----------------
1. Suppress Gradio analytics (env var must be set before Gradio is imported).
2. Parse CLI arguments.
3. Validate that all required data/model files exist; print actionable errors
   and exit early if anything is missing.
4. Load Retriever (FAISS + SQLite + embedding model).
5. Load LLM (GGUF model — may take 10-30 s on first run).
6. Construct Pipeline.
7. Build Gradio UI.
8. Schedule browser open in a background thread (fires after BROWSER_DELAY_S).
9. Launch Gradio server — blocks until the user closes the app / Ctrl-C.
10. Cleanup (close SQLite connection).

Usage
-----
    # From the wiki-offline/ project root:
    python app/main.py

    # Override the default port:
    python app/main.py --port 7861

    # Disable automatic browser launch:
    python app/main.py --no-browser
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import webbrowser

# ---------------------------------------------------------------------------
# CRITICAL: suppress Gradio analytics BEFORE Gradio is imported.
# gui.py imports gradio at module level, so this must come first.
# ---------------------------------------------------------------------------
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import config                        # noqa: E402 (must follow env setup)
from gui import CSS, create_ui       # noqa: E402 (pulls in gradio)
from llm import LLM                  # noqa: E402
from pipeline import Pipeline        # noqa: E402
from retriever import Retriever      # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVER_HOST: str = "127.0.0.1"
BROWSER_DELAY_S: float = 1.5   # seconds to wait before opening the browser


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wiki-offline",
        description="Offline Wikipedia assistant powered by a local LLM.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the local web server (default: 7860).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser tab automatically on startup.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _check_required_files() -> bool:
    """
    Verify all data and model files exist before attempting to load them.
    Prints a clear error for each missing file.
    Returns True if everything is present, False otherwise.
    """
    required = [
        (config.FAISS_PATH,   "FAISS index",   "Run build/04_embed_and_index.py"),
        (config.ID_MAP_PATH,  "id_map",         "Run build/04_embed_and_index.py"),
        (config.DB_PATH,      "SQLite database","Run build/03_build_sqlite.py"),
        (config.MODEL_PATH,   "LLM model",
         "Download phi-3-mini-q4_k_m.gguf from HuggingFace → models/"),
    ]
    all_ok = True
    for path, label, hint in required:
        if not path.exists():
            print(f"  MISSING  {label}: {path}")
            print(f"           → {hint}")
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Browser helper
# ---------------------------------------------------------------------------

def _schedule_browser_open(url: str, delay: float) -> None:
    """Open *url* in the default browser after *delay* seconds (daemon thread)."""
    def _open() -> None:
        time.sleep(delay)
        webbrowser.open(url)

    t = threading.Thread(target=_open, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    url = f"http://{SERVER_HOST}:{args.port}"

    print()
    print("=" * 56)
    print("  Wikipedia Assistant  —  offline edition")
    print("=" * 56)

    # 1. Pre-flight
    print("\nChecking required files…")
    if not _check_required_files():
        print(
            "\nOne or more required files are missing.  "
            "Follow the hints above and re-run."
        )
        _wait_for_keypress()
        sys.exit(1)
    print("  All required files found.")

    # 2. Load Retriever
    print("\nLoading retriever (FAISS + SQLite + embedding model)…", flush=True)
    t0 = time.perf_counter()
    try:
        retriever = Retriever()
    except Exception as exc:
        print(f"\nERROR loading retriever: {exc}")
        _wait_for_keypress()
        sys.exit(1)
    print(f"  Retriever ready  ({time.perf_counter() - t0:.1f} s)")

    # 3. Load LLM
    print("\nLoading language model (this may take 10-30 s)…", flush=True)
    t1 = time.perf_counter()
    try:
        llm = LLM()
    except Exception as exc:
        print(f"\nERROR loading model: {exc}")
        retriever.close()
        _wait_for_keypress()
        sys.exit(1)
    print(f"  Model ready  ({time.perf_counter() - t1:.1f} s)")

    # 4. Build Pipeline
    pipeline = Pipeline(retriever, llm)

    # 5. Build UI
    print("\nBuilding interface…", flush=True)
    demo = create_ui(pipeline)

    # 6. Schedule browser open
    if not args.no_browser:
        _schedule_browser_open(url, BROWSER_DELAY_S)

    # 7. Launch — blocks until Ctrl-C or server shutdown
    print(f"\nStarting server at {url}")
    print("Press Ctrl-C to quit.\n")
    try:
        demo.launch(
            server_name=SERVER_HOST,
            server_port=args.port,
            inbrowser=False,    # we handle browser opening ourselves
            share=False,
            quiet=True,
            css=CSS,
        )
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down…")
        retriever.close()
        print("Done.")


# ---------------------------------------------------------------------------
# Windows-friendly exit pause
# ---------------------------------------------------------------------------

def _wait_for_keypress() -> None:
    """
    On Windows the terminal window closes immediately on exit.
    Pause so the user can read the error message.
    On other platforms this is a no-op.
    """
    if sys.platform == "win32":
        print("\nPress Enter to close…")
        input()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
