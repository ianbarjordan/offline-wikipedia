"""
gui.py

Gradio Blocks user interface for the wiki-offline app.

Public API
----------
    demo = create_ui(pipeline)
    demo.launch(server_name="127.0.0.1", server_port=7860, ...)

Layout
------
    Title + subtitle
    ┌─────────────────────────────────┐
    │  Chatbot (scrollable, 480px)    │
    ├─────────────────────────────────┤
    │  [Textbox (9/10)]  [Ask (1/10)] │
    │  [Clear conversation]           │
    └─────────────────────────────────┘
    Sources (hidden until first answer)
    ┌─────────────────────────────────┐
    │  Sources  [Btn1][Btn2][Btn3]... │
    └─────────────────────────────────┘

Streaming
---------
The respond() generator:
  1. Appends user message + empty assistant placeholder → yields immediately
     so the user sees their message appear without waiting.
  2. Streams tokens from pipeline.query(), accumulating into the assistant
     placeholder → yields on each token.
  3. After the stream ends, reveals the sources row with article buttons.

Sources
-------
Up to TOP_K (5) buttons, one per retrieved article.  Clicking a button
calls _open_article() which opens the pre-rendered HTML file in the default
browser via os.startfile() on Windows or xdg-open on Linux/macOS.

Offline strategy
----------------
Option A (confirmed): no Gradio template patching.  The one async CDN call
(iframe-resizer) fails silently and is never exercised when running standalone
at 127.0.0.1.  Analytics suppressed by setting GRADIO_ANALYTICS_ENABLED=False
in the OS environment before importing Gradio (done in main.py).

Design notes
------------
- history is stored as a list of gr.ChatMessage objects (role/content).
  This is the native format for Gradio 6.7 — no 'type' parameter needed.
- _to_pairs() converts ChatMessage history to (user, assistant) tuples for
  pipeline.query(), which expects the same format as Gradio's old tuple API.
- Source buttons are a fixed set of TOP_K widgets; show/hide via gr.update().
  This avoids dynamic component creation which Gradio does not support well.
- _open_article() is cross-platform: os.startfile on Windows, subprocess
  xdg-open/open elsewhere.  Gracefully skips if the HTML file is missing
  (article was indexed but the HTML build step wasn't run yet).
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import gradio as gr

import config

if TYPE_CHECKING:
    from pipeline import Pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TOP_K = config.TOP_K   # number of source buttons (always rendered, show/hide)

CSS = """
#sources-label { font-weight: 600; margin-bottom: 4px; }
.source-btn { text-align: left !important; }
footer { display: none !important; }
"""


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_ui(pipeline: "Pipeline") -> gr.Blocks:
    """
    Build and return the Gradio Blocks application.

    Parameters
    ----------
    pipeline : Initialised Pipeline instance.  Must not be None.

    Returns
    -------
    gr.Blocks instance (not yet launched).
    """

    # -- Event handlers ------------------------------------------------------

    def respond(
        message: str,
        history: list,
    ) -> Generator:
        """
        Streaming event handler.  A Gradio generator — yields on each token.

        Yields
        ------
        Tuple of (cleared_input, updated_history, articles_state,
                  src_row_update, btn0_update, …, btn{TOP_K-1}_update)
        """
        message = message.strip()
        if not message:
            yield from _noop(history)
            return

        # 1. Append user message + empty assistant placeholder immediately.
        history = history + [
            gr.ChatMessage(role="user", content=message),
            gr.ChatMessage(role="assistant", content=""),
        ]
        yield ("", history, [], gr.update(visible=False),
               *[gr.update(visible=False)] * _TOP_K)

        # 2. Retrieve + stream.
        chat_pairs = _to_pairs(history[:-2])   # exclude the just-added pair
        stream, articles = pipeline.query(message, chat_pairs)

        for token in stream:
            history[-1].content += token
            yield ("", history, articles, gr.update(visible=False),
                   *[gr.update(visible=False)] * _TOP_K)

        # 3. Streaming done — reveal sources.
        src_updates = _build_source_updates(articles)
        show_row = gr.update(visible=bool(articles))
        yield ("", history, articles, show_row, *src_updates)

    def open_article(articles: list, btn_idx: int) -> None:
        """Open the local HTML file for source button *btn_idx*."""
        if btn_idx < len(articles):
            art = articles[btn_idx]
            _open_file(config.ARTICLES_DIR / f"{art['id']}.html")

    def clear_conversation() -> tuple:
        """Reset chat and hide sources."""
        hidden = [gr.update(visible=False)] * _TOP_K
        return [], [], gr.update(visible=False), *hidden

    # -- Layout --------------------------------------------------------------

    with gr.Blocks(title="Wikipedia Assistant") as demo:

        gr.Markdown("# Wikipedia Assistant")
        gr.Markdown(
            "Ask questions about anything in Simple English Wikipedia — "
            "fully offline, no internet required."
        )

        chatbot = gr.Chatbot(
            label="Chat",
            show_label=False,
            height=480,
            placeholder="Your answers will appear here.",
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder="Ask a question…",
                label="",
                scale=9,
                container=False,
                autofocus=True,
                submit_btn=False,
            )
            ask_btn = gr.Button("Ask", variant="primary", scale=1, min_width=80)

        with gr.Row():
            clear_btn = gr.Button("Clear conversation", size="sm", variant="secondary")

        # Sources row — hidden until the first answer completes.
        with gr.Row(visible=False) as src_row:
            gr.Markdown("**Sources**", elem_id="sources-label")
            src_buttons = [
                gr.Button(
                    "—",
                    visible=False,
                    size="sm",
                    variant="secondary",
                    elem_classes=["source-btn"],
                )
                for _ in range(_TOP_K)
            ]

        # State: list[dict] of retrieved articles for the current response.
        articles_state = gr.State([])

        # -- Wire up events --------------------------------------------------

        respond_inputs = [msg, chatbot]
        respond_outputs = [msg, chatbot, articles_state, src_row, *src_buttons]

        msg.submit(respond, respond_inputs, respond_outputs)
        ask_btn.click(respond, respond_inputs, respond_outputs)

        clear_btn.click(
            clear_conversation,
            inputs=[],
            outputs=[chatbot, articles_state, src_row, *src_buttons],
        )

        # Source button click → open HTML article.
        for idx, btn in enumerate(src_buttons):
            btn.click(
                fn=lambda arts, i=idx: open_article(arts, i),
                inputs=[articles_state],
                outputs=[],
            )

    return demo


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _to_pairs(history: list) -> list[tuple[str, str]]:
    """
    Convert a ChatMessage list to (user, assistant) tuple pairs.

    Handles both gr.ChatMessage objects and plain dicts (Gradio may pass
    either depending on context).  Incomplete final user turns are dropped.
    """
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None

    for msg in history:
        role = msg.role if hasattr(msg, "role") else msg["role"]
        content = msg.content if hasattr(msg, "content") else msg["content"]
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, content))
            pending_user = None

    return pairs


def _build_source_updates(articles: list[dict]) -> list:
    """
    Build a list of gr.update() calls for the TOP_K source buttons.
    Visible buttons get the article title as their label; extras are hidden.
    """
    updates = []
    for i in range(_TOP_K):
        if i < len(articles):
            updates.append(gr.update(visible=True, value=articles[i]["title"]))
        else:
            updates.append(gr.update(visible=False, value="—"))
    return updates


def _noop(history: list) -> Generator:
    """Yield a single no-op update (empty message — do nothing)."""
    yield ("", history, [], gr.update(visible=False),
           *[gr.update(visible=False)] * _TOP_K)


def _open_file(path: Path) -> None:
    """
    Open *path* in the system default application.
    Uses os.startfile() on Windows; xdg-open / open on Linux / macOS.
    Silently skips if the file does not exist.
    """
    if not path.exists():
        return
    system = platform.system()
    if system == "Windows":
        os.startfile(str(path))
    elif system == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


# ---------------------------------------------------------------------------
# Smoke test  (run directly: python gui.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock

    print("gui.py smoke test — structural / layout only (no model required)")
    print("=" * 60)

    # Build a mock pipeline so create_ui() can be called without data files.
    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = (
        iter(["The ", "speed ", "of ", "light ", "is ", "fast."]),
        [
            {"id": 1, "title": "Speed of light", "lead": "...", "url_slug": "Speed_of_light"},
            {"id": 2, "title": "Light",           "lead": "...", "url_slug": "Light"},
        ],
    )

    demo = create_ui(mock_pipeline)
    assert demo is not None, "create_ui() returned None"
    print("create_ui() returned a Blocks instance: OK")

    # Verify expected components exist on the demo.
    component_types = {type(c).__name__ for c in demo.blocks.values()}
    print(f"Component types present: {sorted(component_types)}")
    assert "Chatbot"  in component_types, "Chatbot component missing"
    assert "Textbox"  in component_types, "Textbox component missing"
    assert "Button"   in component_types, "Button component missing"
    assert "Markdown" in component_types, "Markdown component missing"
    print("All expected component types present: OK")

    # Exercise _to_pairs() with mock ChatMessage objects.
    h = [
        gr.ChatMessage(role="user",      content="Hi"),
        gr.ChatMessage(role="assistant", content="Hello"),
        gr.ChatMessage(role="user",      content="What is light?"),
        gr.ChatMessage(role="assistant", content="Light is electromagnetic radiation."),
    ]
    pairs = _to_pairs(h)
    assert pairs == [("Hi", "Hello"), ("What is light?", "Light is electromagnetic radiation.")], \
        f"_to_pairs() wrong: {pairs}"
    print("_to_pairs() correctness: OK")

    # Exercise _to_pairs() with incomplete tail (no assistant reply yet).
    h2 = h + [gr.ChatMessage(role="user", content="Follow-up?")]
    pairs2 = _to_pairs(h2)
    assert len(pairs2) == 2, f"Incomplete tail should be dropped, got: {pairs2}"
    print("_to_pairs() incomplete-tail edge case: OK")

    # Exercise _build_source_updates().
    articles = [
        {"id": 1, "title": "Alpha", "lead": "", "url_slug": "Alpha"},
        {"id": 2, "title": "Beta",  "lead": "", "url_slug": "Beta"},
    ]
    updates = _build_source_updates(articles)
    assert len(updates) == _TOP_K
    assert updates[0]["value"] == "Alpha"
    assert updates[1]["value"] == "Beta"
    assert updates[2]["value"] == "—"
    assert updates[2]["visible"] is False
    print("_build_source_updates() correctness: OK")

    print()
    print("All assertions PASSED.")
    print()
    print("To launch interactively (mock pipeline, no model needed):")
    print("  cd wiki-offline && python app/gui.py --launch")

    if "--launch" in sys.argv:
        print()
        print("Launching on http://127.0.0.1:7860 — press Ctrl+C to stop")
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False, css=CSS)
