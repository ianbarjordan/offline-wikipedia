"""
llm.py

Thin wrapper around llama-cpp-python (Llama class).

Loads the GGUF model once at startup and exposes a single public method:

    generate(prompt: str, stream: bool = True) -> Generator[str] | str

When stream=True  → returns a generator that yields token strings one at a time.
When stream=False → returns the full completion as a single string.

Design notes
------------
- n_gpu_layers=0 : CPU-only; end-user machines are unlikely to have a CUDA GPU.
- verbose=False  : llama.cpp prints extensive load diagnostics by default;
                   suppress them in production. Flip to True when debugging
                   model-load issues.
- echo=False     : we pass a fully-formed prompt; we do not want the input
                   echoed back in the token stream.
- temperature=0.2, top_p=0.9 : sensible defaults for factual Q&A; callers can
                   override both via generate() kwargs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional, Union

from llama_cpp import Llama

import config


class LLM:
    """
    Wraps a llama-cpp-python Llama instance.
    Intended to be instantiated once and reused for the lifetime of the app.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        """
        Load the GGUF model.  Defaults to config.MODEL_PATH.

        Raises FileNotFoundError if the model file is absent.
        """
        model_path = model_path or config.MODEL_PATH

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Download gemma-2-2b-q4_k_m.gguf from HuggingFace and place it "
                "in the models/ directory."
            )

        self._llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=config.N_GPU_LAYERS,   # 0 = CPU-only
            n_ctx=config.CTX_WINDOW,
            n_threads=config.N_THREADS,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        stream: bool = True,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> Union[Generator[str, None, None], str]:
        """
        Run inference on *prompt*.

        Parameters
        ----------
        prompt      : Fully-formed prompt string to send to the model.
        stream      : True  → return a generator yielding token strings.
                      False → return the full completion as a single string.
        max_tokens  : Maximum number of tokens to generate.
        temperature : Sampling temperature (lower = more deterministic).
        top_p       : Nucleus sampling cutoff.

        Returns
        -------
        Generator[str, None, None] when stream=True.
        str when stream=False.
        """
        kwargs = dict(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            echo=False,
            stream=stream,
        )

        if stream:
            return self._stream_tokens(prompt, **kwargs)

        result = self._llm(prompt, **kwargs)
        return result["choices"][0]["text"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _stream_tokens(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Yield individual token strings from a streaming llama.cpp response."""
        for chunk in self._llm(prompt, **kwargs):
            choices = chunk.get("choices") if isinstance(chunk, dict) else None
            token = choices[0].get("text", "") if choices else ""
            if token:
                yield token


# ---------------------------------------------------------------------------
# Smoke test  (run directly: python llm.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import time

    PROMPT = "What is the capital of France? Answer in one sentence."

    print(f"Model path : {config.MODEL_PATH}")
    print(f"n_ctx      : {config.CTX_WINDOW}")
    print(f"n_threads  : {config.N_THREADS}")
    print(f"n_gpu_layers: {config.N_GPU_LAYERS}")
    print()

    try:
        print("Loading model…", flush=True)
        t0 = time.perf_counter()
        llm = LLM()
        load_time = time.perf_counter() - t0
        print(f"Model loaded in {load_time:.1f}s\n")
    except FileNotFoundError as exc:
        print(f"SKIP — model file missing:\n  {exc}", file=sys.stderr)
        sys.exit(0)

    # --- streaming test ---
    print(f"Prompt: {PROMPT!r}")
    print("Response (streaming):")
    print("-" * 40)

    tokens: list[str] = []
    t1 = time.perf_counter()
    for token in llm.generate(PROMPT, stream=True, max_tokens=64):
        print(token, end="", flush=True)
        tokens.append(token)
    elapsed = time.perf_counter() - t1

    print()
    print("-" * 40)
    print(f"\nTokens received : {len(tokens)}")
    print(f"Generation time : {elapsed:.2f}s")
    print(f"Tokens/sec      : {len(tokens) / elapsed:.1f}")

    # --- non-streaming test ---
    print("\nResponse (non-streaming):")
    full = llm.generate(PROMPT, stream=False, max_tokens=64)
    print(repr(full))

    print("\nSmoke test PASSED.")
