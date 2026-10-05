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
    # Quick smoke-test on a toy dataframe
    sample = pd.DataFrame([
        {
            "title": "Business approach to the carbon tax",
            "body":  "The carbon tax and carbon pricing debate continues. "
                     "Carbon pricing mechanisms and emissions trading dominate. "
                     "The Paris Agreement underpins climate policy.",
        },
        {
            "title": "Local council budget approved",
            "body":  "The council approved its annual budget with no mention "
                     "of environmental issues.",
        },
        {
            "title": "Climate change threatens reef",
            "body":  "Scientists warn about impacts.",
        },
        {
            "title": "Net zero by 2050",
            "body":  "Australia's net zero target requires net-zero emissions "
                     "across all sectors by 2050.",
        },
    ])
    result = score_articles(sample)
    print(result[["title", "cc_count", "gw_count", "cg_total",
                  "climate_mentions", "title_hit", "relevance"]])
