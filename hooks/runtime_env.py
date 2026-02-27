"""
hooks/runtime_env.py — PyInstaller runtime hook.

Runs before app/main.py when the app is launched from a frozen PyInstaller
bundle.  Sets environment variables that must be in place before any import.

This file is referenced in wiki-offline.spec via runtime_hooks=[...].
"""

import os
import sys

if getattr(sys, "frozen", False):
    _bundle = sys._MEIPASS

    # Redirect HuggingFace model cache so sentence-transformers finds the
    # bundled all-MiniLM-L6-v2 model without hitting the network.
    # The spec copies the developer's HF cache snapshot to hf_cache/ inside
    # the bundle; setting HF_HOME here makes every HF library (transformers,
    # sentence-transformers, huggingface_hub) look there first.
    os.environ.setdefault("HF_HOME", os.path.join(_bundle, "hf_cache"))

    # Belt-and-suspenders: suppress Gradio analytics even if main.py's
    # os.environ.setdefault() hasn't run yet.
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
