"""
cache_bodies.py
Parses all NewsBank PDFs and writes body text to a local cache file.

Preferred output: data/newsbank_bodies.parquet  (requires pyarrow or fastparquet)
Fallback output:  data/newsbank_bodies.pkl.gz   (standard library, always works)

Both formats are read transparently by run_bertopic.py.

This file is intentionally excluded from the git repository (see .gitignore)
because it contains article body text subject to NewsBank licensing. Run this
script once after downloading PDF exports; subsequent pipeline runs will load
from the cache rather than re-parsing all PDFs.

Usage:
    python cache_bodies.py [--force] [--format parquet|pickle]

Options:
    --force             Re-parse all PDFs even if cache already exists
    --format parquet    Force parquet output (requires pyarrow)
    --format pickle     Force pickle+gzip output

Requirements:
    apt install poppler-utils
    pip install pandas pyarrow   # pyarrow optional; pickle used if absent
"""

import argparse
import gzip
import pickle
import sys
from pathlib import Path

import pandas as pd

from config import DATA_DIR, NEWSBANK_FOLDERS, NEWSBANK_ROOT
from parse_newsbank import parse_folder

DATA_DIR.mkdir(exist_ok=True)

PARQUET_PATH = DATA_DIR / "newsbank_bodies.parquet"
PICKLE_PATH  = DATA_DIR / "newsbank_bodies.pkl.gz"


def _parquet_available() -> bool:
    try:
        import pyarrow       # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import fastparquet   # noqa: F401
        return True
    except ImportError:
        return False


def _save(df: pd.DataFrame, fmt: str):
    if fmt == "parquet":
        df.to_parquet(PARQUET_PATH, index=False, compression="snappy")
        size = PARQUET_PATH.stat().st_size / 1_048_576
        print(f"Saved {len(df):,} records → {PARQUET_PATH}  ({size:.1f} MB)")
    else:
        with gzip.open(PICKLE_PATH, "wb") as f:
            pickle.dump(df, f, protocol=5)
        size = PICKLE_PATH.stat().st_size / 1_048_576
        print(f"Saved {len(df):,} records → {PICKLE_PATH}  ({size:.1f} MB)")


def load_cache() -> pd.DataFrame:
    """Load whichever cache format exists (called by run_bertopic.py)."""
    _cols = ["title", "date", "publication", "body"]
    if PARQUET_PATH.exists():
        return pd.read_parquet(PARQUET_PATH, columns=_cols)
    if PICKLE_PATH.exists():
        with gzip.open(PICKLE_PATH, "rb") as f:
            df = pickle.load(f)
        # Older cache files may lack 'publication'; include it only if present
        available = [c for c in _cols if c in df.columns]
        return df[available]
    raise FileNotFoundError(
        "No NewsBank body cache found. Run:  python cache_bodies.py"
    )


def build_cache(fmt: str) -> pd.DataFrame:
    all_records = []

    for folder_name, meta in NEWSBANK_FOLDERS.items():
        folder_path = NEWSBANK_ROOT / folder_name
        if not folder_path.exists():
            print(f"  WARNING: {folder_path} not found — skipping")
            continue

        recs = parse_folder(
            folder_path,
            publication=meta["publication"],
            content_type_override=meta["content_type"],
            folder_label=folder_name,
        )
        for r in recs:
            all_records.append({
                "title":       r["title"],
                "date":        r["date"],
                "publication": r["publication"],
                "folder":      r["folder"],
                "filename":    r["filename"],
                "body":        r["body"],
            })

    df = pd.DataFrame(all_records)

    # Deduplicate across folders — same article can appear in multiple NewsBank
    # exports (e.g. Australian_Editor overlapping TheAustralian_SpecificEditors).
    # Keep the first occurrence (preserves folder/filename provenance of earliest match).
    before = len(df)
    df = df.drop_duplicates(subset=["title", "date", "publication"], keep="first")
    dupes = before - len(df)
    if dupes:
        print(f"  Deduplication: {before:,} → {len(df):,} records ({dupes:,} duplicates removed)")
    else:
        print(f"  Deduplication: no duplicates found across {before:,} records")

    _save(df, fmt)
    return df


def main(force: bool = False, fmt: str = "auto"):
    cache_exists = PARQUET_PATH.exists() or PICKLE_PATH.exists()
    if cache_exists and not force:
        path = PARQUET_PATH if PARQUET_PATH.exists() else PICKLE_PATH
        size_mb = path.stat().st_size / 1_048_576
        print(f"Cache already exists: {path}  ({size_mb:.1f} MB)")
        print("Run with --force to rebuild.")
        return

    if fmt == "auto":
        fmt = "parquet" if _parquet_available() else "pickle"
    print(f"Output format: {fmt}")

    n_pdfs = sum(1 for _ in NEWSBANK_ROOT.rglob("*.pdf"))
    print(f"Parsing ~{n_pdfs} PDFs from {len(NEWSBANK_FOLDERS)} folders …\n")
    build_cache(fmt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cache NewsBank body text for BERTopic pipeline"
    )
    parser.add_argument("--force",  action="store_true",
                        help="Rebuild even if cache exists")
    parser.add_argument("--format", choices=["auto", "parquet", "pickle"],
                        default="auto", dest="fmt",
                        help="Output format (default: parquet if available, else pickle)")
    args = parser.parse_args()
    main(force=args.force, fmt=args.fmt)
