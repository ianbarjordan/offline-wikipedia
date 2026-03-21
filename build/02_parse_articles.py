"""
02_parse_articles.py

Stream-parse the Simple English Wikipedia CirrusSearch dump and write a
staging JSONL file for the downstream SQLite and embedding scripts.

CirrusSearch dump format
------------------------
The .json.bz2 file contains pairs of newline-delimited JSON lines:

  {"index":{"_id":"12345","_type":"page"}}   <- action line (skip)
  { ... full article JSON ... }              <- document line (process)
  {"index":{"_id":"12346","_type":"page"}}
  { ... full article JSON ... }
  ...

Relevant document fields:
  namespace     : int  — 0 = main article namespace
  redirect      : list — non-empty means this page is a redirect
  title         : str  — article title
  opening_text  : str  — lead paragraph, already plain text (no wiki markup)
  text          : str  — full article body, plain text
  category      : list — category strings (used to detect disambiguation)

Output
------
  raw/articles_parsed.jsonl  — one JSON object per line:
    {"title": "...", "lead": "...", "body": "..."}

Usage:
    python build/02_parse_articles.py
    python build/02_parse_articles.py --dump raw/simplewiki_content-20250101-00000.json.bz2
    python build/02_parse_articles.py --out raw/articles_parsed.jsonl
"""

import argparse
import bz2
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm

# Paths relative to the wiki-offline/ project root
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "raw"
DEFAULT_OUT = DEFAULT_RAW_DIR / "articles_parsed.jsonl"

MAX_LEAD_WORDS = 300


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def truncate_words(text: str, max_words: int) -> str:
    """Return *text* truncated to at most *max_words* whitespace-split words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def normalise_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including newlines) to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


# Patterns that identify Wikipedia maintenance/template sentences to strip.
# These appear in opening_text when a page was created with a template that
# left placeholder text, or when maintenance banners leaked into the lead.
_LEAD_NOISE_RE = re.compile(
    r'\[Your (?:Name|name)\]'            # unfilled template placeholders
    r'|\bWritten by\b'                   # article creation template
    r'|This article (?:needs|may|does|has|is a|was|contains)'  # maintenance banners
    r'|You can help Wikipedia'
    r'|Please help (?:improve|expand)',
    re.IGNORECASE,
)


def clean_lead(text: str) -> str:
    """
    Remove sentences containing Wikipedia maintenance/template noise.
    Splits on sentence boundaries, drops noisy sentences, rejoins.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    clean = [s for s in sentences if not _LEAD_NOISE_RE.search(s)]
    return " ".join(clean).strip()


def is_disambiguation(doc: dict) -> bool:
    """Return True if this document is a disambiguation page."""
    title: str = doc.get("title", "")
    if "(disambiguation)" in title.lower():
        return True
    categories = doc.get("category") or []
    return any("disambiguation" in cat.lower() for cat in categories)


def extract_lead(doc: dict) -> str:
    """
    Return the lead paragraph text for *doc*, max MAX_LEAD_WORDS words.

    Prefers the CirrusSearch 'opening_text' field (already the lead paragraph).
    Falls back to the first MAX_LEAD_WORDS words of 'text' if opening_text is
    absent or empty.
    """
    opening = normalise_whitespace(doc.get("opening_text") or "")
    if opening:
        return clean_lead(truncate_words(opening, MAX_LEAD_WORDS))

    body = normalise_whitespace(doc.get("text") or "")
    return clean_lead(truncate_words(body, MAX_LEAD_WORDS))


# ---------------------------------------------------------------------------
# Dump discovery
# ---------------------------------------------------------------------------


def find_dump(raw_dir: Path) -> Path:
    """Return the newest simplewiki CirrusSearch dump in *raw_dir*."""
    candidates = sorted(
        raw_dir.glob("simplewiki_content-*-00000.json.bz2")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No simplewiki dump found in {raw_dir}/. "
            "Run 01_download_wiki.py first."
        )
    return candidates[-1]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def parse_dump(dump_path: Path, out_path: Path) -> dict:
    """
    Stream-parse *dump_path* and write filtered articles to *out_path*.

    Returns a dict with counts for the final summary.
    """
    counts = {
        "written": 0,
        "disambiguation": 0,
        "non_main_ns": 0,
        "no_text": 0,
        "parse_error": 0,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        bz2.open(dump_path, "rt", encoding="utf-8", errors="replace") as bz,
        open(out_path, "w", encoding="utf-8") as out_fh,
        tqdm(desc="Parsing articles", unit="doc", dynamic_ncols=True) as bar,
    ):
        pending_doc = False  # True when the next line should be a document

        for raw_line in bz:
            line = raw_line.strip()
            if not line:
                continue

            # ---- Detect action lines by their distinctive leading key ----
            # Action lines look like: {"index":{"_id":"...","_type":"..."}}
            # They are reliably identifiable without full JSON parsing.
            if line.startswith('{"index"'):
                pending_doc = True
                continue

            if not pending_doc:
                # Unexpected non-action line with no preceding action — skip
                continue

            pending_doc = False
            bar.update(1)

            # ---- Parse the document line ----
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                counts["parse_error"] += 1
                continue

            # ---- Filters ----
            if doc.get("namespace", -1) != 0:
                counts["non_main_ns"] += 1
                continue

            # NOTE: The CirrusSearch "redirect" field on a content article lists
            # pages that redirect TO it (incoming aliases) — it does NOT mean
            # this page itself is a redirect.  True redirect pages have no
            # opening_text or text, so they are caught by the no_text check
            # below.  Filtering on redirect here incorrectly drops popular
            # articles like "George Washington" that have many incoming aliases.

            if is_disambiguation(doc):
                counts["disambiguation"] += 1
                continue

            title = (doc.get("title") or "").strip()
            if not title:
                counts["no_text"] += 1
                continue

            lead = extract_lead(doc)
            if not lead:
                counts["no_text"] += 1
                continue

            body = normalise_whitespace(doc.get("text") or "")

            # ---- Write staging record ----
            record = {"title": title, "lead": lead, "body": body}
            out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["written"] += 1

    return counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse the CirrusSearch Wikipedia dump into a staging JSONL file."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="Path to the .json.bz2 dump (auto-detected in raw/ if omitted).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSONL staging file path.",
    )
    args = parser.parse_args()

    dump_path = args.dump or find_dump(DEFAULT_RAW_DIR)
    size_gb = dump_path.stat().st_size / 1_073_741_824

    print(f"Dump file : {dump_path}  ({size_gb:.2f} GB compressed)")
    print(f"Output    : {args.out}")
    print(f"Max lead  : {MAX_LEAD_WORDS} words")
    print()

    counts = parse_dump(dump_path, args.out)

    out_size_mb = args.out.stat().st_size / 1_048_576
    print(f"\nParsing complete.")
    print(f"  Articles written         : {counts['written']:>10,}")
    print(f"  Skipped — disambiguation : {counts['disambiguation']:>10,}")
    print(f"  Skipped — non-main ns    : {counts['non_main_ns']:>10,}")
    print(f"  Skipped — no text/redir  : {counts['no_text']:>10,}")
    print(f"  Parse errors             : {counts['parse_error']:>10,}")
    print(f"\n  Staging file : {args.out}  ({out_size_mb:.1f} MB)")
    print(f"\nNext step: python build/03_build_sqlite.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
