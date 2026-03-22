"""
03_build_sqlite.py

Read the staged JSONL file from 02_parse_articles.py and:

  1. Insert every article into data/wikipedia.db
       Table: articles (id, title, lead, body, url_slug)
       url_slug = Wikipedia title slug (spaces → underscores)

  2. Write a self-contained HTML file per article to data/articles/{id}.html
       - No external dependencies (inline CSS)
       - Opens correctly in any Windows browser with no internet connection
       - Includes title (H1), lead paragraph, body text, and a link to the
         live Simple Wikipedia page for reference

Usage:
    python build/03_build_sqlite.py
    python build/03_build_sqlite.py --staging raw/articles_parsed.jsonl
    python build/03_build_sqlite.py --db data/wikipedia.db --articles-dir data/articles
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

from tqdm import tqdm

# Paths relative to the wiki-offline/ project root
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_STAGING = PROJECT_ROOT / "raw" / "articles_parsed.jsonl"
DEFAULT_DB = PROJECT_ROOT / "data" / "wikipedia.db"
DEFAULT_ARTICLES_DIR = PROJECT_ROOT / "data" / "articles"

COMMIT_EVERY = 2_000  # Commit to SQLite every N inserts

# ---------------------------------------------------------------------------
# HTML template  ({{ / }} are escaped braces for str.format())
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title_escaped}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: Georgia, 'Times New Roman', serif;
      max-width: 860px;
      margin: 40px auto;
      padding: 0 24px 60px;
      line-height: 1.8;
      color: #1a1a1a;
      background: #f8f8f6;
    }}
    h1 {{
      font-size: 1.9rem;
      font-weight: bold;
      border-bottom: 2px solid #3366cc;
      padding-bottom: 0.35em;
      margin-bottom: 0.6em;
      color: #222;
    }}
    .lead {{
      font-size: 1.08rem;
      color: #222;
      background: #eef2fb;
      border-left: 4px solid #3366cc;
      padding: 14px 18px;
      margin-bottom: 1.8em;
      border-radius: 0 6px 6px 0;
    }}
    .body-text p {{
      font-size: 0.97rem;
      color: #333;
      margin-bottom: 1em;
    }}
    .body-text h2 {{
      font-size: 1.1rem;
      font-weight: bold;
      color: #222;
      margin: 1.6em 0 0.4em;
      border-bottom: 1px solid #ddd;
      padding-bottom: 0.2em;
    }}
    .footer {{
      margin-top: 2.5em;
      font-size: 0.82rem;
      color: #777;
      border-top: 1px solid #ddd;
      padding-top: 0.8em;
    }}
    .footer a {{ color: #3366cc; text-decoration: none; }}
    .footer a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>{title_escaped}</h1>
  <div class="lead">{lead_escaped}</div>
  <div class="body-text">{body_html}</div>
  <div class="footer">
    Source:&nbsp;<a href="https://simple.wikipedia.org/wiki/{slug_escaped}">
    simple.wikipedia.org/wiki/{slug_escaped}</a>
  </div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_slug(title: str) -> str:
    """Wikipedia-style URL slug: spaces become underscores."""
    return re.sub(r" ", "_", title)


def _esc(text: str) -> str:
    """Minimal HTML-safe escaping for text content and attribute values."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )


# Section header pattern: short line that looks like a heading
# (≤60 chars, no trailing punctuation except colon)
_SECTION_HEAD_RE = re.compile(r'^.{1,60}:?\s*$')


def body_to_html(body: str) -> str:
    """
    Convert a body string (paragraphs separated by \\n\\n) to HTML.
    Short single-sentence paragraphs that look like section headers
    are wrapped in <h2>; all others in <p>.
    """
    if not body:
        return ""
    paragraphs = body.split("\n\n")
    parts = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        escaped = _esc(para)
        # Heuristic: ≤60 chars, no sentence-ending punctuation mid-string → heading
        if len(para) <= 60 and not re.search(r'[.!?]\s', para):
            parts.append(f"<h2>{escaped}</h2>")
        else:
            parts.append(f"<p>{escaped}</p>")
    return "\n".join(parts)


def render_html(title: str, lead: str, body: str, url_slug: str) -> str:
    return _HTML_TEMPLATE.format(
        title_escaped=_esc(title),
        lead_escaped=_esc(lead),
        body_html=body_to_html(body),
        slug_escaped=_esc(url_slug),
    )


def count_lines(path: Path) -> int:
    """Fast line count via raw byte scanning."""
    count = 0
    with open(path, "rb") as fh:
        for _ in fh:
            count += 1
    return count


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS articles")
    conn.execute(
        """
        CREATE TABLE articles (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            title    TEXT    NOT NULL,
            lead     TEXT    NOT NULL,
            body     TEXT    NOT NULL,
            url_slug TEXT    NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX idx_title ON articles (title)")
    conn.commit()


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def build(
    staging_path: Path,
    db_path: Path,
    articles_dir: Path,
) -> int:
    """
    Stream the staging JSONL, insert into SQLite, and write HTML files.
    Returns the total number of articles inserted.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    articles_dir.mkdir(parents=True, exist_ok=True)

    print("Counting records in staging file...")
    total = count_lines(staging_path)
    print(f"  {total:,} records to process.")
    print()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")  # faster bulk writes
    conn.execute("PRAGMA synchronous=NORMAL")
    create_schema(conn)
    cur = conn.cursor()

    inserted = 0

    with (
        open(staging_path, "r", encoding="utf-8") as fh,
        tqdm(total=total, desc="Building DB + HTML", unit="articles", dynamic_ncols=True) as bar,
    ):
        for line in fh:
            bar.update(1)
            line = line.strip()
            if not line:
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = (rec.get("title") or "").strip()
            lead = (rec.get("lead") or "").strip()
            body = (rec.get("body") or "").strip()

            if not title or not lead:
                continue

            url_slug = make_slug(title)

            cur.execute(
                "INSERT INTO articles (title, lead, body, url_slug) VALUES (?,?,?,?)",
                (title, lead, body, url_slug),
            )
            article_id = cur.lastrowid

            # Write the HTML file
            html_path = articles_dir / f"{article_id}.html"
            html_path.write_text(
                render_html(title, lead, body, url_slug),
                encoding="utf-8",
            )

            inserted += 1
            if inserted % COMMIT_EVERY == 0:
                conn.commit()

    conn.commit()
    conn.close()
    return inserted


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(db_path: Path) -> int:
    """Return the row count from the finished database."""
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    conn.close()
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SQLite database and per-article HTML files from staged JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--staging",
        type=Path,
        default=DEFAULT_STAGING,
        help="Input JSONL staging file (output of 02_parse_articles.py).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Output SQLite database path.",
    )
    parser.add_argument(
        "--articles-dir",
        type=Path,
        default=DEFAULT_ARTICLES_DIR,
        help="Output directory for per-article HTML files.",
    )
    args = parser.parse_args()

    if not args.staging.exists():
        print(
            f"ERROR: Staging file not found: {args.staging}\n"
            "Run 02_parse_articles.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Staging file : {args.staging}")
    print(f"Database     : {args.db}")
    print(f"HTML dir     : {args.articles_dir}")
    print()

    inserted = build(args.staging, args.db, args.articles_dir)
    db_count = verify(args.db)

    db_size_mb = args.db.stat().st_size / 1_048_576
    print(f"\nDone.")
    print(f"  Articles inserted   : {inserted:>10,}")
    print(f"  DB row count (check): {db_count:>10,}  ✓")
    print(f"  Database size       : {db_size_mb:>9.1f} MB  ({args.db})")
    print(f"  HTML files          : {args.articles_dir}/{{id}}.html")

    if inserted != db_count:
        print(
            f"\nWARNING: inserted={inserted} != db_count={db_count}. "
            "Something may have gone wrong — investigate before proceeding.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nNext step: python build/04_embed_and_index.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
