"""
score_and_classify.py
Scores each article for climate relevance and applies the hybrid inclusion
criterion used in the corpus construction for "A Climate of Opinion".

Hybrid criterion — an article is INCLUDED if ANY of the following hold:
  (a) core_count >= CC_CORE_THRESHOLD   (default 3)
  (b) Any CORE_CLIMATE_PHRASE appears in the title (case-insensitive)
  (c) core_count >= 1 AND climate_mentions >= CLIMATE_MENTIONS_THRESHOLD (default 3)

where:
  core_count        = occurrences of ANY phrase in CORE_CLIMATE_PHRASES in body text
  climate_mentions  = occurrences of ANY term in CLIMATE_TERMS (from config.py)

CORE_CLIMATE_PHRASES spans four dimensions:
  (1) Canonical identifiers     : "climate change", "global warming"
  (2) Contemporary equivalents  : "climate emergency", "climate crisis"
  (3) Scientific mechanism      : "greenhouse gas", "ipcc"
  (4) Policy mechanisms/targets : "carbon tax", "carbon price", "carbon pricing",
                                  "emissions trading", "carbon trading",
                                  "paris agreement", "net zero", "net-zero"

Legacy columns cc_count and gw_count are retained for traceability.
cg_total now stores the full core phrase count (not just cc+gw).

Usage:
    from score_and_classify import score_articles
    df = score_articles(df)          # adds scoring columns + 'relevance'
"""

import re
from typing import Optional
import pandas as pd

from config import (CLIMATE_TERMS, CORE_CLIMATE_PHRASES,
                    CC_CORE_THRESHOLD, CLIMATE_MENTIONS_THRESHOLD)

# Pre-compile patterns for speed
_CC_PAT   = re.compile(r"\bclimate\s+change\b",  re.IGNORECASE)   # legacy column
_GW_PAT   = re.compile(r"\bglobal\s+warming\b",  re.IGNORECASE)   # legacy column
_CORE_PAT = re.compile(
    "|".join(r"\b" + re.escape(t) + r"\b" for t in CORE_CLIMATE_PHRASES),
    re.IGNORECASE,
)
_CM_PAT = re.compile(
    # Left-boundary only: prevents "coal" matching "coalition" etc., while
    # preserving prefix terms ("decarboni" → decarbonise/ization, "adapt" → adaptation).
    "|".join(r"\b" + re.escape(t) for t in CLIMATE_TERMS),
    re.IGNORECASE,
)


def _safe_str(text) -> str:
    """Return text as string, or empty string if None/NaN."""
    return text if isinstance(text, str) else ""


def score_article(title: str, body: str) -> dict:
    """
    Score a single article.  Returns a dict with:
        cc_count, gw_count, cg_total, climate_mentions, title_hit, relevance
    """
    body_s  = _safe_str(body)
    title_s = _safe_str(title)

    cc  = len(_CC_PAT.findall(body_s))          # legacy: "climate change" count
    gw  = len(_GW_PAT.findall(body_s))          # legacy: "global warming" count
    cg  = len(_CORE_PAT.findall(body_s))        # core phrase count (all 14 phrases)
    th  = bool(_CORE_PAT.search(title_s))       # any core phrase in title

    if cg >= CC_CORE_THRESHOLD or th:
        cm = 0      # already included — skip full vocabulary count
        inc = True
    else:
        cm  = len(_CM_PAT.findall(body_s))
        inc = cg >= 1 and cm >= CLIMATE_MENTIONS_THRESHOLD

    return {
        "cc_count":         cc,
        "gw_count":         gw,
        "cg_total":         cg,
        "climate_mentions": cm,
        "title_hit":        th,
        "relevance":        "Include" if inc else "Exclude",
    }


def score_articles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add scoring columns to a DataFrame that has 'title' and 'body' columns.
    Existing relevance / scoring columns are overwritten.

    Parameters
    ----------
    df : DataFrame with at minimum 'title' and 'body' columns.

    Returns
    -------
    df with additional columns:
        cc_count, gw_count, cg_total, climate_mentions, title_hit, relevance
    """
    scores = df.apply(
        lambda row: score_article(
            row.get("title", ""), row.get("body", "")
        ),
        axis=1,
        result_type="expand",
    )
    # Drop any pre-existing scoring columns so we replace cleanly
    drop_cols = [c for c in scores.columns if c in df.columns]
    df = df.drop(columns=drop_cols)
    return pd.concat([df, scores], axis=1)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Score a CSV of articles for climate relevance and apply the "
            "hybrid inclusion criterion. Adds columns: cc_count, gw_count, "
            "cg_total, climate_mentions, title_hit, relevance."
        )
    )
    parser.add_argument("input",  help="Path to input CSV file")
    parser.add_argument("output", help="Path to write scored CSV file")
    parser.add_argument(
        "--title-col", default="title", metavar="COL",
        help="Column name for article title (default: title)",
    )
    parser.add_argument(
        "--body-col", default="body", metavar="COL",
        help="Column name for article body text (default: body)",
    )
    parser.add_argument(
        "--include-only", action="store_true",
        help="Write only articles that pass the inclusion criterion",
    )
    args = parser.parse_args()

    print(f"Reading {args.input} …")
    df = pd.read_csv(args.input, low_memory=False)

    # Rename non-standard column names to the expected 'title' / 'body'
    rename = {}
    if args.title_col != "title":
        rename[args.title_col] = "title"
    if args.body_col != "body":
        rename[args.body_col] = "body"
    if rename:
        df = df.rename(columns=rename)

    missing = [c for c in ("title", "body") if c not in df.columns]
    if missing:
        print(f"Error: column(s) not found in input: {missing}", file=sys.stderr)
        print(f"Available columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(1)

    print(f"Scoring {len(df):,} articles …")
    result = score_articles(df)

    if args.include_only:
        before = len(result)
        result = result[result["relevance"] == "Include"]
        print(f"Retained {len(result):,} / {before:,} articles after inclusion criterion")
    else:
        n_include = (result["relevance"] == "Include").sum()
        print(f"Include: {n_include:,}  |  Exclude: {len(result) - n_include:,}")

    result.to_csv(args.output, index=False)
    print(f"Written to {args.output}")
