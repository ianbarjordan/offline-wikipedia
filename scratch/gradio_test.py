"""
scratch/gradio_test.py

Minimal Gradio smoke-test used to:
  1. Confirm the installed version.
  2. Locate the static asset directory.
  3. Check for external CDN references in the bundled JS/CSS.

Run without a live server:
    python scratch/gradio_test.py --info-only

Run as a real server (blocks until Ctrl-C):
    python scratch/gradio_test.py
"""

import argparse
import os

# Must be set BEFORE importing gradio to suppress analytics.
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import gradio as gr  # noqa: E402


def echo(text: str) -> str:
    """Return whatever the user typed, prefixed with a label."""
    return f"You said: {text}"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Wiki Offline — test") as demo:
        gr.Markdown("## Gradio connectivity test")
        inp = gr.Textbox(label="Input", placeholder="Type something…")
        btn = gr.Button("Submit")
        out = gr.Textbox(label="Output", interactive=False)
        btn.click(fn=echo, inputs=inp, outputs=out)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal Gradio test / asset inspector."
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Print version + asset info and exit without starting the server.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port (default: 7860)"
    )
    args = parser.parse_args()

    print(f"Gradio version : {gr.__version__}")
    print(f"Analytics env  : {os.environ.get('GRADIO_ANALYTICS_ENABLED')}")

    if args.info_only:
        return

    demo = build_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_api=False,
    )


if __name__ == "__main__":
    main()
