"""
01_download_wiki.py

Download the Simple English Wikipedia CirrusSearch JSON dump from
https://dumps.wikimedia.org/other/cirrussearch/current/

The target file matches: simplewiki-YYYYMMDD-cirrussearch-content.json.gz
Saved to the raw/ scratch directory (gitignored).

Usage:
    python build/01_download_wiki.py
    python build/01_download_wiki.py --out-dir path/to/raw
    python build/01_download_wiki.py --filename simplewiki-20250101-cirrussearch-content.json.gz
"""

import argparse
import re
import sys
from pathlib import Path

import requests
from tqdm import tqdm

DUMP_BASE_URL = "https://dumps.wikimedia.org/other/cirrussearch/current/"
FILENAME_RE = re.compile(r"simplewiki-\d{8}-cirrussearch-content\.json\.gz")

# Path relative to the wiki-offline/ project root
DEFAULT_RAW_DIR = Path(__file__).parent.parent / "raw"

CHUNK_SIZE = 256 * 1024  # 256 KB per read chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_dump_filename() -> str:
    """Fetch the cirrussearch listing page and return the simplewiki filename."""
    print(f"Querying dump listing: {DUMP_BASE_URL}")
    try:
        resp = requests.get(DUMP_BASE_URL, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach dump server: {exc}") from exc

    matches = FILENAME_RE.findall(resp.text)
    if not matches:
        raise RuntimeError(
            f"No 'simplewiki-*-cirrussearch-content.json.gz' entry found at "
            f"{DUMP_BASE_URL}. Check the URL manually — the dump may have moved."
        )

    # Sort and pick the newest date if multiple exist
    filename = sorted(matches)[-1]
    print(f"Found dump file: {filename}")
    return filename


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
        help="Directory to save the downloaded .json.gz file.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help=(
            "Exact dump filename to download. "
            "Auto-detected from the listing page if not specified."
        ),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    filename = args.filename or find_dump_filename()
    dest = args.out_dir / filename

    if dest.exists():
        size_gb = dest.stat().st_size / 1_073_741_824
        print(f"File already exists ({size_gb:.2f} GB): {dest}")
        print("Delete it and re-run to force a fresh download.")
        return

    url = DUMP_BASE_URL + filename
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
