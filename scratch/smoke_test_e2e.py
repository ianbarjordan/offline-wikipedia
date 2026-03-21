"""
scratch/smoke_test_e2e.py

Full end-to-end smoke test for the wiki-offline app.

Stages
------
  1. Generate 30 synthetic articles → raw/smoke_e2e.jsonl
  2. Run build/03 → data/wikipedia.db + data/articles/
  3. Run build/04 → data/wikipedia.faiss + data/id_map.json
  4. Instantiate real Retriever — verify search results
  5. Instantiate mock LLM — verify token generator
  6. Instantiate real Pipeline — verify query() output
  7. Create Gradio UI — verify component structure
  8. Simulate respond() generator — drive the streaming callback end-to-end
  9. Simulate clear_conversation() callback
  10. Print pass/fail summary

Run from the wiki-offline/ project root:
    python scratch/smoke_test_e2e.py
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path
from typing import Generator

# ---------------------------------------------------------------------------
# Ensure app/ is on the path for config / retriever / pipeline / gui imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
APP_DIR = PROJECT_ROOT / "app"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "build"))

import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# ---------------------------------------------------------------------------
# Synthetic article corpus (30 articles across varied topics so FAISS recall
# can be checked with topical queries)
# ---------------------------------------------------------------------------

ARTICLES = [
    ("Sun",         "The Sun is the star at the center of the Solar System. It is a nearly perfect ball of hot plasma.",
                    "The Sun accounts for about 99.86% of the total mass of the Solar System."),
    ("Moon",        "The Moon is Earth's only natural satellite. It orbits Earth at a distance of about 384,400 km.",
                    "The Moon is the fifth-largest satellite in the Solar System."),
    ("Mars",        "Mars is the fourth planet from the Sun. It is often called the Red Planet because of its reddish appearance.",
                    "Mars has the tallest volcano and the deepest canyon in the Solar System."),
    ("Water",       "Water is a transparent, tasteless, odourless, and nearly colourless chemical substance.",
                    "Water is the most abundant substance on Earth and is essential for life."),
    ("Photosynthesis", "Photosynthesis is the process used by plants to convert sunlight into food.",
                       "Photosynthesis takes place mainly in the leaves of plants using chlorophyll."),
    ("Gravity",     "Gravity is a natural force that attracts objects toward each other.",
                    "Isaac Newton described gravity as a force. Albert Einstein described it as a curvature of spacetime."),
    ("DNA",         "DNA, or deoxyribonucleic acid, is a molecule that carries the genetic instructions in living things.",
                    "DNA is found in the nucleus of cells and is made up of four chemical bases."),
    ("Atom",        "An atom is the smallest unit of ordinary matter that forms a chemical element.",
                    "Atoms consist of a nucleus made of protons and neutrons, surrounded by electrons."),
    ("Light",       "Light is electromagnetic radiation that can be seen by the human eye.",
                    "The speed of light in a vacuum is approximately 299,792 kilometres per second."),
    ("Electricity", "Electricity is the flow of electric charge through a conductor.",
                    "Electricity powers most modern technology, including computers and lighting."),
    ("Volcano",     "A volcano is an opening in Earth's crust through which hot lava, ash, and gases escape.",
                    "There are about 1,500 active volcanoes on Earth."),
    ("Earthquake",  "An earthquake is the shaking of the ground caused by movement in Earth's crust.",
                    "Most earthquakes occur along tectonic plate boundaries."),
    ("Ocean",       "An ocean is a large body of salt water that covers most of Earth's surface.",
                    "The Pacific Ocean is the largest and deepest ocean on Earth."),
    ("Dinosaur",    "Dinosaurs were a group of reptiles that dominated the land for over 160 million years.",
                    "Non-avian dinosaurs became extinct about 66 million years ago."),
    ("Evolution",   "Evolution is the process by which living things change over many generations.",
                    "Charles Darwin described the theory of evolution by natural selection."),
    ("Cell",        "A cell is the basic unit of life. All living things are made of one or more cells.",
                    "There are two main types of cells: prokaryotic and eukaryotic."),
    ("Bacteria",    "Bacteria are tiny single-celled organisms that live almost everywhere on Earth.",
                    "Some bacteria are helpful to humans, while others can cause disease."),
    ("Virus",       "A virus is a tiny infectious agent that replicates inside the cells of a living host.",
                    "Viruses are not considered fully alive because they cannot reproduce on their own."),
    ("Human brain", "The human brain is the central organ of the human nervous system.",
                    "The brain contains about 86 billion neurons that communicate through synapses."),
    ("Heart",       "The heart is a muscular organ that pumps blood through the circulatory system.",
                    "The human heart beats about 100,000 times per day."),
    ("Oxygen",      "Oxygen is a chemical element with the symbol O and atomic number 8.",
                    "Oxygen makes up about 21 percent of Earth's atmosphere and is essential for respiration."),
    ("Carbon",      "Carbon is a chemical element that is the basis of all known life on Earth.",
                    "Carbon forms more compounds than any other element."),
    ("Temperature", "Temperature is a measure of how hot or cold something is.",
                    "Temperature is measured using thermometers in units such as Celsius and Fahrenheit."),
    ("Pressure",    "Pressure is the force exerted per unit area on a surface.",
                    "Atmospheric pressure at sea level is about 101,325 pascals."),
    ("Sound",       "Sound is a vibration that travels through a medium as a wave of pressure.",
                    "Sound travels at about 343 metres per second in air at room temperature."),
    ("Magnetism",   "Magnetism is a force that attracts or repels certain materials.",
                    "The Earth itself acts as a giant magnet with a north and a south pole."),
    ("Cloud",       "A cloud is a mass of water droplets or ice crystals suspended in the atmosphere.",
                    "Clouds form when water vapour cools and condenses around tiny particles."),
    ("Rain",        "Rain is liquid water in the form of droplets that fall from clouds.",
                    "Rain is a major component of the water cycle and is essential for most life."),
    ("Tree",        "A tree is a tall plant with a trunk, branches, and leaves.",
                    "Trees absorb carbon dioxide and release oxygen through photosynthesis."),
    ("Fish",        "Fish are aquatic animals that breathe through gills and typically have fins and scales.",
                    "There are more than 33,000 known species of fish."),
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
_results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = PASS if condition else FAIL
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    _results.append((label, condition, detail))


def section(title: str) -> None:
    print(f"\n{'='*56}")
    print(f"  {title}")
    print(f"{'='*56}")


# ---------------------------------------------------------------------------
# Stage 1 — Generate synthetic JSONL
# ---------------------------------------------------------------------------

def stage_1_generate_jsonl() -> Path:
    section("Stage 1 — Generate synthetic JSONL")
    raw_dir = PROJECT_ROOT / "raw"
    raw_dir.mkdir(exist_ok=True)
    jsonl_path = raw_dir / "smoke_e2e.jsonl"

    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for title, lead, body in ARTICLES:
            fh.write(json.dumps({"title": title, "lead": lead, "body": body}) + "\n")

    lines = sum(1 for _ in open(jsonl_path))
    check("JSONL written", lines == len(ARTICLES), f"{lines} articles")
    print(f"  → {jsonl_path}")
    return jsonl_path


# ---------------------------------------------------------------------------
# Stage 2 — Build SQLite DB + HTML articles
# ---------------------------------------------------------------------------

SMOKE_DATA_DIR = PROJECT_ROOT / "scratch" / "smoke_data"
SMOKE_DB_PATH       = SMOKE_DATA_DIR / "smoke.db"
SMOKE_FAISS_PATH    = SMOKE_DATA_DIR / "smoke.faiss"
SMOKE_ID_MAP_PATH   = SMOKE_DATA_DIR / "smoke_id_map.json"
SMOKE_ARTICLES_DIR  = SMOKE_DATA_DIR / "articles"


def stage_2_build_sqlite(jsonl_path: Path) -> None:
    section("Stage 2 — Build SQLite DB + HTML articles (build/03)")
    SMOKE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SMOKE_ARTICLES_DIR.mkdir(parents=True, exist_ok=True)
    import subprocess
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "build" / "03_build_sqlite.py"),
            "--staging", str(jsonl_path),
            "--db", str(SMOKE_DB_PATH),
            "--articles-dir", str(SMOKE_ARTICLES_DIR),
        ],
        capture_output=True,
        text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip())
    check("build/03 exit code 0", result.returncode == 0, f"rc={result.returncode}")

    check("smoke.db created", SMOKE_DB_PATH.exists(), str(SMOKE_DB_PATH))

    import sqlite3
    conn = sqlite3.connect(SMOKE_DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    conn.close()
    check("Row count matches", count == len(ARTICLES), f"{count} rows")

    html_files = list(SMOKE_ARTICLES_DIR.glob("*.html"))
    check("HTML files created", len(html_files) == len(ARTICLES),
          f"{len(html_files)} html files")


# ---------------------------------------------------------------------------
# Stage 3 — Build FAISS index + id_map
# ---------------------------------------------------------------------------

def stage_3_build_faiss() -> None:
    section("Stage 3 — Build FAISS index + id_map (build/04)")
    import subprocess
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "build" / "04_embed_and_index.py"),
            "--device", "cpu",
            "--db", str(SMOKE_DB_PATH),
            "--faiss-out", str(SMOKE_FAISS_PATH),
            "--id-map-out", str(SMOKE_ID_MAP_PATH),
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    print(result.stdout.strip()[-1000:])  # last 1000 chars to avoid flooding
    if result.returncode != 0:
        print(result.stderr.strip()[-500:])
    check("build/04 exit code 0", result.returncode == 0, f"rc={result.returncode}")

    check("smoke.faiss created",    SMOKE_FAISS_PATH.exists())
    check("smoke_id_map.json created", SMOKE_ID_MAP_PATH.exists())

    id_map = json.loads(SMOKE_ID_MAP_PATH.read_text())
    check("id_map non-empty", len(id_map) == len(ARTICLES),
          f"{len(id_map)} entries")


# ---------------------------------------------------------------------------
# Stage 4 — Real Retriever
# ---------------------------------------------------------------------------

def stage_4_retriever() -> "Retriever":
    section("Stage 4 — Real Retriever (FAISS + SQLite + embedding model)")
    from retriever import Retriever

    t0 = time.perf_counter()
    r = Retriever(
        faiss_path=SMOKE_FAISS_PATH,
        id_map_path=SMOKE_ID_MAP_PATH,
        db_path=SMOKE_DB_PATH,
    )
    elapsed = time.perf_counter() - t0
    check("Retriever loaded", True, f"{elapsed:.1f}s")

    # Basic search
    results = r.search("what is the speed of light?", top_k=3)
    check("search returns results", len(results) > 0, f"got {len(results)}")
    check("results have required keys",
          all("id" in x and "title" in x and "lead" in x and "url_slug" in x
              for x in results))
    # Light / Electricity should be in top results for this query
    titles = [x["title"] for x in results]
    print(f"  Query: 'speed of light'  →  {titles}")
    check("'Light' article retrieved", "Light" in titles, f"got {titles}")

    # Query about plants
    results2 = r.search("how do plants make food from sunlight?", top_k=3)
    titles2 = [x["title"] for x in results2]
    print(f"  Query: 'plants sunlight'  →  {titles2}")
    check("'Photosynthesis' article retrieved", "Photosynthesis" in titles2,
          f"got {titles2}")

    # Empty query edge case
    empty = r.search("")
    check("empty query returns []", empty == [], f"got {empty!r}")

    return r


# ---------------------------------------------------------------------------
# Stage 5 — Mock LLM
# ---------------------------------------------------------------------------

class MockLLM:
    """
    Drop-in replacement for app/llm.py LLM.
    Returns a token generator over a fixed response string.
    Does NOT load any GGUF file.
    """
    RESPONSE = (
        "Based on the Wikipedia articles provided, "
        "light travels at approximately 299,792 kilometres per second in a vacuum. "
        "This is one of the fundamental constants of physics."
    )

    def generate(
        self,
        prompt: str,
        stream: bool = True,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        if stream:
            return self._stream()
        return self.RESPONSE

    def _stream(self) -> Generator[str, None, None]:
        """Yield the response word-by-word to simulate streaming."""
        for word in self.RESPONSE.split():
            yield word + " "


def stage_5_mock_llm() -> MockLLM:
    section("Stage 5 — Mock LLM (fixed token generator)")
    llm = MockLLM()

    # streaming
    tokens = list(llm.generate("dummy prompt", stream=True))
    full = "".join(tokens).strip()
    check("streaming yields tokens", len(tokens) > 5, f"{len(tokens)} tokens")
    check("joined text matches RESPONSE", full == MockLLM.RESPONSE.strip(),
          repr(full[:60]))

    # non-streaming
    direct = llm.generate("dummy prompt", stream=False)
    check("non-streaming returns string", isinstance(direct, str))
    print(f"  Sample tokens: {tokens[:4]!r}")
    return llm


# ---------------------------------------------------------------------------
# Stage 6 — Real Pipeline
# ---------------------------------------------------------------------------

def stage_6_pipeline(retriever, llm) -> "Pipeline":
    section("Stage 6 — Real Pipeline (Retriever + MockLLM)")
    from pipeline import Pipeline, _build_context, _build_prompt
    import config

    pipeline = Pipeline(retriever, llm)

    query = "What is the speed of light?"
    stream, articles = pipeline.query(query, [])

    check("query() returns tuple", isinstance(articles, list))
    check("articles non-empty", len(articles) > 0, f"{len(articles)} articles")

    tokens = list(stream)
    response = "".join(tokens).strip()
    check("stream yields tokens", len(tokens) > 0, f"{len(tokens)} tokens")
    check("response non-empty", len(response) > 10, repr(response[:60]))

    # Multi-turn: pass one history exchange
    history = [("What is gravity?", "Gravity is a force that attracts objects.")]
    stream2, articles2 = pipeline.query("Tell me more about gravity", history)
    tokens2 = list(stream2)
    check("multi-turn query works", len(tokens2) > 0, f"{len(tokens2)} tokens")

    print(f"  Response preview: {response[:80]!r}")
    print(f"  Sources: {[a['title'] for a in articles]}")
    return pipeline


# ---------------------------------------------------------------------------
# Stage 7 — Gradio UI creation
# ---------------------------------------------------------------------------

def stage_7_create_ui(pipeline) -> "gr.Blocks":
    section("Stage 7 — Gradio UI creation")
    import gradio as gr
    from gui import create_ui

    demo = create_ui(pipeline)
    check("create_ui() returns Blocks", isinstance(demo, gr.Blocks))

    component_types = {type(c).__name__ for c in demo.blocks.values()}
    check("Chatbot present",  "Chatbot"  in component_types)
    check("Textbox present",  "Textbox"  in component_types)
    check("Button present",   "Button"   in component_types)
    check("Markdown present", "Markdown" in component_types)
    check("State present",    "State"    in component_types)
    print(f"  Component types: {sorted(component_types)}")
    return demo


# ---------------------------------------------------------------------------
# Stage 8 — Simulate the respond() generator end-to-end
# ---------------------------------------------------------------------------

def stage_8_respond_generator(demo) -> None:
    section("Stage 8 — Simulate respond() streaming callback")
    import gradio as gr

    # In Gradio 6.7, event handlers are stored in demo.fns as BlockFunction
    # objects with .name and .fn attributes.
    respond_fn = next(
        (bf.fn for bf in demo.fns.values() if bf.name == "respond"),
        None,
    )
    check("respond() function found in demo.fns", respond_fn is not None)
    if respond_fn is None:
        return

    # Drive the generator: empty history, simple question
    history = []
    question = "What is the speed of light?"
    all_yields = list(respond_fn(question, history))

    check("respond() yields at least 3 times", len(all_yields) >= 3,
          f"yielded {len(all_yields)} times")

    # Each yield is a tuple: (cleared_input, history, articles, row_update, *btn_updates)
    first_yield = all_yields[0]
    check("first yield is a tuple", isinstance(first_yield, tuple))

    cleared_input  = first_yield[0]
    first_history  = first_yield[1]
    check("input cleared to ''", cleared_input == "", repr(cleared_input))
    check("history has 2 messages after first yield", len(first_history) == 2,
          f"len={len(first_history)}")
    check("first message is user role",
          first_history[0].role == "user", first_history[0].role)
    check("second message is assistant role",
          first_history[1].role == "assistant", first_history[1].role)

    # Last yield should have non-empty assistant content and visible sources
    last_yield     = all_yields[-1]
    last_history   = last_yield[1]
    last_articles  = last_yield[2]
    src_row_update = last_yield[3]

    assistant_content = last_history[-1].content
    check("final assistant message non-empty", len(assistant_content) > 10,
          repr(assistant_content[:60]))
    check("articles list returned", isinstance(last_articles, list))
    row_visible = src_row_update.get("visible") if isinstance(src_row_update, dict) else getattr(src_row_update, "visible", None)
    check("sources row made visible", row_visible is True,
          str(src_row_update))

    print(f"  Total yields     : {len(all_yields)}")
    print(f"  Final response   : {assistant_content[:80]!r}")
    print(f"  Sources returned : {[a['title'] for a in last_articles]}")


# ---------------------------------------------------------------------------
# Stage 9 — clear_conversation()
# ---------------------------------------------------------------------------

def stage_9_clear(demo) -> None:
    section("Stage 9 — clear_conversation() callback")
    import gradio as gr

    clear_fn = next(
        (bf.fn for bf in demo.fns.values() if bf.name == "clear_conversation"),
        None,
    )
    check("clear_conversation() found in demo.fns", clear_fn is not None)
    if clear_fn is None:
        return

    result = clear_fn()
    check("returns a tuple", isinstance(result, tuple))
    cleared_history = result[0]
    cleared_articles = result[1]
    row_update = result[2]
    check("history reset to []", cleared_history == [], str(cleared_history))
    check("articles reset to []", cleared_articles == [], str(cleared_articles))
    row_hidden = row_update.get("visible") if isinstance(row_update, dict) else getattr(row_update, "visible", None)
    check("sources row hidden", row_hidden is False,
          str(row_update))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary() -> None:
    section("Summary")
    total  = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed

    for label, ok, detail in _results:
        mark = PASS if ok else FAIL
        print(f"  [{mark}] {label}")

    print()
    print(f"  {passed}/{total} checks passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
    else:
        print("  — ALL PASSED")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()

    jsonl_path = stage_1_generate_jsonl()
    stage_2_build_sqlite(jsonl_path)
    stage_3_build_faiss()

    retriever  = stage_4_retriever()
    llm        = stage_5_mock_llm()
    pipeline   = stage_6_pipeline(retriever, llm)
    demo       = stage_7_create_ui(pipeline)
    stage_8_respond_generator(demo)
    stage_9_clear(demo)

    retriever.close()
    print_summary()

    elapsed = time.perf_counter() - t_start
    print(f"  Total time: {elapsed:.1f}s")

    failed = sum(1 for _, ok, _ in _results if not ok)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
