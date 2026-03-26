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
Gemma 2 IT uses the following chat template (no dedicated system token):

    <start_of_turn>user
    {system_message}

    {user_turn}<end_of_turn>
    <start_of_turn>model
    {assistant_turn}<end_of_turn>
    <start_of_turn>user
    {current_message}<end_of_turn>
    <start_of_turn>model

The system message (with Wikipedia context) is prepended to the first
user turn, which is the standard approach for Gemma 2 instruction-tuned
models that have no <|system|> special token.

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

import re
from typing import Generator

import config
from llm import LLM
from retriever import Retriever

# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are a Wikipedia assistant. {grounding}

Rules:
- Answer using ONLY the Wikipedia articles provided below — not your training knowledge.
- If the articles do not contain the answer, say exactly: "My Wikipedia database doesn't cover that topic."
- Use plain prose. No markdown headers or bullet points unless listing 3+ distinct items.
- Keep answers concise: 2–4 sentences for simple facts, one short paragraph for explanations.
- Do not repeat the question. Do not say "According to Wikipedia" or "The context says".
- Do not add details about fictional characters, movies, TV shows, or celebrities
  that are not explicitly stated in the Wikipedia articles below.

Wikipedia context:
{context}"""

_LOW_CONFIDENCE_REPLY = (
    "I couldn't find reliable information on that in my Wikipedia database. "
    "Try rephrasing or asking about a more specific topic."
)

_GREETING_REPLY = (
    "I'm a Wikipedia assistant. Ask me anything — I'll look it up in "
    "Simple English Wikipedia and answer based on what the articles say."
)

_META_REPLY = (
    "I'm a Wikipedia assistant running locally on your machine. "
    "I was built to answer questions using Simple English Wikipedia articles. "
    "Ask me about any topic and I'll search my database."
)

_INJECTION_REPLY = (
    "I can only answer questions using my Wikipedia database. "
    "I'm not able to follow instructions that ask me to change how I work."
)

# Fix 1 — Prompt injection detection. Checked before all other handlers.
_INJECTION_RE = re.compile(
    r"(ignore|disregard|forget|override|bypass|cancel|reset)\s+"
    r"(your\s+)?(previous|prior|above|all|the|my|any)?\s*"
    r"(instruction|prompt|rule|constraint|guideline|system|context|order)s?"
    r"|you\s+are\s+now\s+|act\s+as\s+|pretend\s+(to\s+be\s+|you\s+are\s+)"
    r"|roleplay\s+as|jailbreak|DAN\b",
    re.IGNORECASE,
)

# Fix 3 — Identity/meta questions (subset of conversational, gets its own reply).
_META_RE = re.compile(
    r"where\s+are\s+you\s+from|who\s+(made|built|created)\s+you|"
    r"are\s+you\s+(an?\s+)?(ai|robot|human|bot)|who\s+are\s+you",
    re.IGNORECASE,
)

# Fix 3 — Expanded conversational handler covering reactions, exclamations, meta.
_CONVERSATIONAL_RE = re.compile(
    r"^\s*("
    # Greetings
    r"hi+|hello+|hey+|howdy|greetings|good\s+(morning|afternoon|evening|day)|"
    # Reactions / exclamations
    r"sweet[!.]?|cool[!.]?|wow[!.]?|nice[!.]?|great[!.]?|awesome[!.]?|"
    r"interesting[!.]?|amazing[!.]?|lol[!.]?|haha+|lmao|spoiler[!.]?|"
    r"really\??|are\s+you\s+sure\??|ok(ay)?[!.]?|sure[!.]?|"
    r"thanks?(\s+you)?[!.]?|thank\s+you[!.]?|"
    # Meta / identity questions about the assistant
    r"what\s+(can|do|are|will)\s+you|who\s+are\s+you|what\s+is\s+this|"
    r"where\s+are\s+you\s+from|who\s+(made|built|created)\s+you|"
    r"are\s+you\s+(an?\s+)?(ai|robot|human|bot)|"
    r"help(\s+me)?"
    r")\W*$",
    re.IGNORECASE,
)

# Fix 4/5 — Pronoun-based follow-up detection for query augmentation.
_PRONOUN_RE = re.compile(
    r"\b(he|she|it|they|his|her|its|their|him|them|this|that|these|those)\b",
    re.IGNORECASE,
)

_AUGMENT_STOPWORDS = {
    "the", "a", "an", "is", "was", "are", "were", "be", "been",
    "do", "did", "does", "have", "has", "had", "will", "would",
    "can", "could", "should", "may", "might", "what", "where",
    "when", "why", "how", "who", "which", "and", "or", "but",
    "in", "on", "at", "to", "of", "for", "with", "about",
}

# Fix C — Short-message reaction catch-all sets.
_QUESTION_WORDS = frozenset({
    "what", "where", "who", "when", "how", "why", "which",
})


def _is_conversational_reaction(message: str) -> bool:
    """
    Return True for short reactions/affirmations that aren't valid Wikipedia queries.
    Triggers when: ≤5 words, no proper noun after the first word, no question word.
    """
    words = message.strip().rstrip("?!.,").split()
    if len(words) > 5:
        return False
    # A question word signals a real query.
    if any(w.lower() in _QUESTION_WORDS for w in words):
        return False
    # A capitalized content word after position 0 signals a named entity → real query.
    content_words = words[1:]
    if any(w[0].isupper() for w in content_words if w.isalpha()):
        return False
    return True


def _augment_query(user_message: str, chat_history: list[tuple[str, str]]) -> str:
    """
    Augment the retrieval query with context from the previous user turn when:
      (a) the query contains a pronoun (existing logic), OR
      (b) the query is short (≤6 words) and has no proper noun — entity-less follow-up.
    Word cap raised from 8 → 12 to cover longer pronoun-containing follow-ups. (Fix D)
    """
    words = user_message.split()
    has_pronoun = bool(_PRONOUN_RE.search(user_message))

    # Short query with no proper noun after position 0 = entity-less follow-up.
    content_words = words[1:]
    has_proper_noun = any(w[0].isupper() for w in content_words if w.isalpha())
    is_entity_less = len(words) <= 6 and not has_proper_noun

    if len(words) > 12 or not (has_pronoun or is_entity_less):
        return user_message
    if not chat_history:
        return user_message
    prev_user = chat_history[-1][0]
    key_words = [
        w.strip(".,?!") for w in prev_user.split()
        if w.lower().strip(".,?!") not in _AUGMENT_STOPWORDS and len(w) > 2
    ][:4]
    if not key_words:
        return user_message
    return " ".join(key_words) + " " + user_message


def _const_generator(text: str) -> Generator[str, None, None]:
    """Yield a single string as a one-shot generator (mirrors the streaming API)."""
    yield text


def _prepend_generator(prefix: str, gen) -> Generator[str, None, None]:
    yield prefix
    yield from gen


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
        # Fix 1 — Block prompt injection before any other handler.
        if _INJECTION_RE.search(user_message):
            return _const_generator(_INJECTION_REPLY), []

        # Fix 3 — Identity questions get a dedicated reply.
        if _META_RE.search(user_message):
            return _const_generator(_META_REPLY), []

        # Short-circuit conversational inputs — no retrieval, no LLM call.
        if _CONVERSATIONAL_RE.match(user_message):
            return _const_generator(_GREETING_REPLY), []

        # Fix C — Catch short reactions the regex misses (e.g. "Only 77k?!", "Okay, cool").
        if _is_conversational_reaction(user_message):
            return _const_generator(_GREETING_REPLY), []

        # Fix 4/5 — Augment pronoun-heavy follow-up queries with prior context.
        retrieval_query = _augment_query(user_message, chat_history)
        articles = self._retriever.search(retrieval_query)

        low_confidence = (not articles) or (articles[0]["score"] < config.CONFIDENCE_THRESHOLD)

        if low_confidence and not articles:
            return _const_generator(_LOW_CONFIDENCE_REPLY), []

        context = _build_context(articles[:config.MAX_LLM_CONTEXT_SOURCES])
        # Always pass original user_message (not augmented) to the LLM.
        prompt = _build_prompt(user_message, chat_history, context, low_confidence=low_confidence)
        stream = self._llm.generate(prompt, stream=True, max_tokens=config.MAX_NEW_TOKENS)

        display = articles[:config.MAX_DISPLAY_SOURCES]

        if low_confidence:
            return _prepend_generator(
                "I didn't find a strong Wikipedia match for your question, so this answer may be incomplete.\n\n",
                stream,
            ), display

        return stream, display


# ---------------------------------------------------------------------------
# Module-level prompt helpers (pure functions — easy to unit-test)
# ---------------------------------------------------------------------------

_GROUNDING_HIGH = (
    "You MUST use ONLY the Wikipedia articles below to answer. "
    "Do NOT use your training knowledge — not even for famous people, "
    "fictional characters, current events, or well-known facts. "
    "Copy facts directly from the articles. Do not infer or extrapolate. "
    "If the articles do not contain the answer, say: "
    "'My Wikipedia database doesn't cover that topic.'"
)

_GROUNDING_LOW = (
    "The Wikipedia articles below are a weak match for this question. "
    "Answer strictly from what the articles say. "
    "If they don't contain the answer, say: "
    "'My Wikipedia database doesn't cover that topic.'"
)


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
    low_confidence: bool = False,
) -> str:
    """
    Assemble the full Gemma 2 chat prompt string.

    Includes:
    - System message (prepended to first user turn) with embedded Wikipedia context
    - Last config.CHAT_HISTORY_TURNS (user, assistant) exchanges
    - Current user message
    - Opening <start_of_turn>model token to prime generation
    """
    grounding = _GROUNDING_LOW if low_confidence else _GROUNDING_HIGH
    system_text = _SYSTEM_TEMPLATE.format(grounding=grounding, context=context)

    recent_history = chat_history[-config.CHAT_HISTORY_TURNS :]
    parts: list[str] = []

    # Historical turns — no system text (keeps prompt compact).
    for user_turn, assistant_turn in recent_history:
        parts.append(f"<start_of_turn>user\n{user_turn}<end_of_turn>\n")
        parts.append(f"<start_of_turn>model\n{assistant_turn}<end_of_turn>\n")

    # System + fresh context always injected into the CURRENT user turn so the
    # model sees the relevant Wikipedia articles immediately before generating.
    parts.append(f"<start_of_turn>user\n{system_text}\n\n{user_message}<end_of_turn>\n")
    parts.append("<start_of_turn>model\n")

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
    assert "<start_of_turn>model" in prompt,        "Missing <start_of_turn>model token"
    assert "Paris" in prompt,                       "Context not embedded"
    assert "What continent" in prompt,              "History not included"
    assert mock_question in prompt,                 "User message missing"
    assert prompt.endswith("<start_of_turn>model\n"), "Prompt must end with <start_of_turn>model"

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
