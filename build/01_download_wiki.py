"""
01_download_wiki.py

Download the Simple English Wikipedia CirrusSearch dump from:

  https://dumps.wikimedia.org/other/cirrus_search_index/{YYYYMMDD}/
      index_name=simplewiki_content/
          simplewiki_content-{YYYYMMDD}-00000.json.bz2

The script discovers the most recent dated directory automatically by
scraping https://dumps.wikimedia.org/other/cirrus_search_index/

Saved to the raw/ scratch directory (gitignored).

Usage:
    python build/01_download_wiki.py
    python build/01_download_wiki.py --out-dir path/to/raw
    python build/01_download_wiki.py --date 20250101
    python build/01_download_wiki.py --dry-run
"""

import argparse
import re
import sys
from pathlib import Path

import requests
from tqdm import tqdm

CIRRUS_INDEX_BASE = "https://dumps.wikimedia.org/other/cirrus_search_index/"
# Dated directories appear in the listing as href="20250101/"
DATE_RE = re.compile(r'href="(\d{8})/"')

# Path relative to the wiki-offline/ project root
DEFAULT_RAW_DIR = Path(__file__).parent.parent / "raw"

CHUNK_SIZE = 256 * 1024  # 256 KB per read chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_latest_date() -> str:
    """Scrape the cirrus_search_index listing and return the newest YYYYMMDD."""
    print(f"Querying dump index: {CIRRUS_INDEX_BASE}")
    try:
        resp = requests.get(CIRRUS_INDEX_BASE, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach dump server: {exc}") from exc

    dates = DATE_RE.findall(resp.text)
    if not dates:
        raise RuntimeError(
            f"No dated directories found at {CIRRUS_INDEX_BASE}. "
            "Check the URL manually — the dump layout may have changed."
        )

    latest = sorted(dates)[-1]
    print(f"Latest dump date: {latest}")
    return latest


def build_url(date: str) -> str:
    """Construct the full download URL for the given YYYYMMDD date string."""
    return (
        f"{CIRRUS_INDEX_BASE}{date}/"
        f"index_name=simplewiki_content/"
        f"simplewiki_content-{date}-00000.json.bz2"
    )


def local_filename(date: str) -> str:
    """Return the local filename to save the dump as."""
    return f"simplewiki_content-{date}-00000.json.bz2"


def download(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest*, showing a tqdm progress bar."""
    print(f"Downloading: {url}")
    try:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total_bytes = int(resp.headers.get("content-length", 0))

            with open(dest, "wb") as fh, tqdm(
                total=total_bytes or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
                dynamic_ncols=True,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    fh.write(chunk)
                    bar.update(len(chunk))

    except requests.RequestException as exc:
        # Remove partial file so a re-run starts fresh
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Download failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Simple English Wikipedia CirrusSearch dump.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory to save the downloaded .json.bz2 file.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        metavar="YYYYMMDD",
        help=(
            "Dump date to download. "
            "Auto-detected from the index listing if not specified."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and print the download URL without downloading anything.",
    )
    args = parser.parse_args()

    date = args.date or find_latest_date()
    url = build_url(date)
    fname = local_filename(date)
    dest = args.out_dir / fname

    print(f"URL      : {url}")
    print(f"Local    : {dest}")

    if args.dry_run:
        print("\n--dry-run: no download performed.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        size_gb = dest.stat().st_size / 1_073_741_824
        print(f"File already exists ({size_gb:.2f} GB): {dest}")
        print("Delete it and re-run to force a fresh download.")
        return

    download(url, dest)

    size_gb = dest.stat().st_size / 1_073_741_824
    print(f"\nDownload complete.")
    print(f"  Saved to : {dest}")
    print(f"  Size     : {size_gb:.2f} GB")
    print(f"\nNext step: python build/02_parse_articles.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
