"""
pipeline.py

RAG orchestration layer.  Glues the Retriever and LLM together.

Public API
----------
    pipeline = Pipeline(retriever, llm)
    stream, articles = pipeline.query(user_message, chat_history)

    # stream  : Generator[str, None, None] — token strings from the LLM
    # articles: list[dict]                 — retrieved Wikipedia articles
    #             keys: id, title, lead, url_slug
    #           Used by gui.py to render "Open source" buttons.

Prompt format
-------------
Phi-3 Mini uses special tokens for its chat template:

    <|system|>
    {system_message}<|end|>
    <|user|>
    {user_turn}<|end|>
    <|assistant|>
    {assistant_turn}<|end|>
    <|user|>
    {current_message}<|end|>
    <|assistant|>

The system message embeds the retrieved Wikipedia context so the model
answers from that context rather than from its parametric memory.

Design notes
------------
- The Pipeline owns no persistent state beyond its Retriever and LLM
  references.  All per-query state is local to query().
- chat_history is a list of (user_str, assistant_str) tuples — the same
  format Gradio's gr.Chatbot widget produces.  Only the last
  config.CHAT_HISTORY_TURNS exchanges are included to keep the prompt
  within the context window.
- Retrieval is done on the raw user_message (not the history-augmented
  prompt), which consistently produces the most relevant results.
"""

from __future__ import annotations

from typing import Generator

import config
from llm import LLM
from retriever import Retriever

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are a helpful assistant that answers questions exclusively from the Wikipedia \
context provided below. Do NOT use your training knowledge or add information \
not present in the context. If the context does not clearly answer the question, \
reply with: "I couldn't find reliable information on that in the provided Wikipedia \
articles." Be concise, accurate, and cite sources using [N] notation where helpful.

Wikipedia context:
{context}"""

_LOW_CONFIDENCE_REPLY = (
    "I couldn't find reliable information on that in Wikipedia. "
    "The search didn't return closely related articles — try rephrasing "
    "or asking about a more specific topic."
)


def _const_generator(text: str) -> Generator[str, None, None]:
    """Yield a single string as a one-shot generator (mirrors the streaming API)."""
    yield text


class Pipeline:
    """
    Orchestrates retrieval → prompt construction → LLM generation.
    Instantiate once and reuse for the lifetime of the app.
    """

    def __init__(self, retriever: Retriever, llm: LLM) -> None:
        self._retriever = retriever
        self._llm = llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        user_message: str,
        chat_history: list[tuple[str, str]],
    ) -> tuple[Generator[str, None, None], list[dict]]:
        """
        Answer *user_message* using RAG.

        Parameters
        ----------
        user_message  : The current question from the user.
        chat_history  : List of (user, assistant) string pairs — previous
                        turns of the conversation.  May be empty.

        Returns
        -------
        (stream, articles)
        stream   : Generator that yields token strings from the LLM.
        articles : List of retrieved article dicts (id, title, lead, url_slug),
                   ordered by FAISS similarity.  May be empty if retrieval
                   found nothing.
        """
        articles = self._retriever.search(user_message)

        # Confidence gate: if the top result's cosine similarity is below the
        # threshold the retrieved context is too weak to ground a useful answer.
        # Return a canned reply immediately rather than passing noise to the LLM.
        best_score = articles[0]["score"] if articles else 0.0
        if best_score < config.CONFIDENCE_THRESHOLD:
            return _const_generator(_LOW_CONFIDENCE_REPLY), articles

        context = _build_context(articles)
        prompt = _build_prompt(user_message, chat_history, context)
        stream = self._llm.generate(prompt, stream=True)
        return stream, articles


# ---------------------------------------------------------------------------
# Module-level prompt helpers (pure functions — easy to unit-test)
# ---------------------------------------------------------------------------

def _build_context(articles: list[dict]) -> str:
    """
    Format a list of retrieved article dicts into a numbered context block.

    Each entry is:
        [N] Article Title
        <lead paragraph text>
    """
    if not articles:
        return "No relevant Wikipedia articles were found for this question."

    blocks: list[str] = []
    for i, art in enumerate(articles, 1):
        blocks.append(f"[{i}] {art['title']}\n{art['lead']}")
    return "\n\n".join(blocks)


def _build_prompt(
    user_message: str,
    chat_history: list[tuple[str, str]],
    context: str,
) -> str:
    """
    Assemble the full Phi-3 chat prompt string.

    Includes:
    - System message with embedded Wikipedia context
    - Last config.CHAT_HISTORY_TURNS (user, assistant) exchanges
    - Current user message
    - Opening <|assistant|> token to prime generation
    """
    system_text = _SYSTEM_TEMPLATE.format(context=context)

    parts: list[str] = [f"<|system|>\n{system_text}<|end|>\n"]

    recent_history = chat_history[-config.CHAT_HISTORY_TURNS :]
    for user_turn, assistant_turn in recent_history:
        parts.append(f"<|user|>\n{user_turn}<|end|>\n")
        parts.append(f"<|assistant|>\n{assistant_turn}<|end|>\n")

    parts.append(f"<|user|>\n{user_message}<|end|>\n")
    parts.append("<|assistant|>\n")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Smoke test  (run directly: python pipeline.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("pipeline.py smoke test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: prompt building (no data files required)
    # ------------------------------------------------------------------
    print("\n[1] Prompt construction (no data files needed)")

    mock_articles = [
        {
            "id": 1,
            "title": "Paris",
            "lead": "Paris is the capital and largest city of France.",
            "url_slug": "Paris",
            "score": 0.85,
        },
        {
            "id": 2,
            "title": "France",
            "lead": "France is a country in Western Europe.",
            "url_slug": "France",
            "score": 0.72,
        },
    ]
    mock_history = [
        ("What continent is France on?", "France is on the continent of Europe."),
    ]
    mock_question = "What is the capital of France?"

    context = _build_context(mock_articles)
    prompt = _build_prompt(mock_question, mock_history, context)

    print(f"  Articles in context : {len(mock_articles)}")
    print(f"  History turns used  : {min(len(mock_history), config.CHAT_HISTORY_TURNS)}")
    print(f"  Prompt length (chars): {len(prompt)}")

    # Structural assertions
    assert "<|system|>" in prompt,          "Missing <|system|> token"
    assert "Paris" in prompt,               "Context not embedded"
    assert "What continent" in prompt,      "History not included"
    assert mock_question in prompt,         "User message missing"
    assert prompt.endswith("<|assistant|>\n"), "Prompt must end with <|assistant|>"

    print("  All structural assertions PASSED.")

    # Empty-history edge case
    prompt_no_hist = _build_prompt(mock_question, [], context)
    assert "What continent" not in prompt_no_hist, "Empty history leaked into prompt"
    print("  Empty-history edge case PASSED.")

    # Empty-articles edge case
    empty_ctx = _build_context([])
    assert "No relevant" in empty_ctx, "Empty articles should produce fallback message"
    print("  Empty-articles edge case PASSED.")

    # ------------------------------------------------------------------
    # Test 2: full pipeline (requires data files + model)
    # ------------------------------------------------------------------
    print("\n[2] Full pipeline (requires data files and model)")

    missing: list[str] = []
    for label, path in [
        ("FAISS index", config.FAISS_PATH),
        ("id_map", config.ID_MAP_PATH),
        ("SQLite DB", config.DB_PATH),
        ("GGUF model", config.MODEL_PATH),
    ]:
        if not path.exists():
            missing.append(label)

    if missing:
        print(f"  SKIP — missing: {', '.join(missing)}")
        print("\nSmoke test PASSED (prompt tests only).")
        sys.exit(0)

    from retriever import Retriever
    from llm import LLM

    print("  Loading Retriever…", flush=True)
    retriever = Retriever()
    print("  Loading LLM…", flush=True)
    llm = LLM()

    pipeline = Pipeline(retriever, llm)

    question = "What is the speed of light?"
    print(f"  Query: {question!r}")
    print("  Response (streaming):")
    print("  " + "-" * 40)

    stream, articles = pipeline.query(question, [])
    tokens: list[str] = []
    for token in stream:
        print(token, end="", flush=True)
        tokens.append(token)

    print()
    print("  " + "-" * 40)
    print(f"  Tokens received : {len(tokens)}")
    print(f"  Sources returned: {len(articles)}")
    for art in articles:
        print(f"    - [{art['id']}] {art['title']}")

    assert len(tokens) > 0,    "No tokens received from stream"
    assert len(articles) > 0,  "No articles returned by retriever"

    retriever.close()
    print("\nSmoke test PASSED (full pipeline).")
