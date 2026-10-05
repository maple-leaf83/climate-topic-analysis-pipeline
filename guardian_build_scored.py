"""
guardian_build_scored.py
------------------------
Builds a scored, relevance-filtered corpus of Guardian opinion, analysis,
and commentary articles for the comparative BERTopic analysis.

Source
  guardian_articles_v2.csv  (one level above repo root)
  Produced by fetch_guardian.py — already tone-filtered (no news) and
  edition-tagged (UK / AU) with content_type derived from Guardian tone tags.

Relevance criterion (same as Australian corpus)
  An article is included if ANY of the following hold:
    (a) Any CORE_CLIMATE_PHRASE appears >= CC_CORE_THRESHOLD times in body
    (b) Any CORE_CLIMATE_PHRASE appears in the title
    (c) Any CORE_CLIMATE_PHRASE appears >= 1 time AND
        climate_mentions >= CLIMATE_MENTIONS_THRESHOLD

Output
  data/guardian/guardian_articles_scored.csv   — all articles + scores (UK + AU)
  data/guardian/guardian_relevant.csv          — relevant articles only (input to BERTopic)

Note on era coverage
  commentisfree launched in 2006, so Pre-Howard and most of the Howard era
  are not represented. Era labels are assigned for completeness.
"""

import sys
from pathlib import Path
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config import GUARDIAN_V2_CSV
from score_and_classify import score_articles

OUT_DIR = ROOT / "data" / "guardian"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Government era boundaries (Australian federal elections / transitions) ──────
ERA_BOUNDARIES = [
    ("Pre-Howard",       None,          "1996-03-11"),
    ("Howard",           "1996-03-11",  "2007-11-24"),
    ("Rudd/Gillard",     "2007-11-24",  "2013-09-18"),
    ("Abbott",           "2013-09-18",  "2015-09-15"),
    ("Turnbull/Morrison","2015-09-15",  "2022-05-23"),
    ("Albanese",         "2022-05-23",  None),
]


def assign_era(date_series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(date_series, errors="coerce")
    era   = pd.Series("Unknown", index=date_series.index)
    for label, start, end in ERA_BOUNDARIES:
        mask = pd.Series(True, index=dates.index)
        if start:
            mask &= dates >= pd.Timestamp(start)
        if end:
            mask &= dates < pd.Timestamp(end)
        era[mask] = label
    return era


# ── Load ───────────────────────────────────────────────────────────────────────
print(f"Loading {GUARDIAN_V2_CSV} …")
df = pd.read_csv(GUARDIAN_V2_CSV)
print(f"  Total articles loaded: {len(df):,}")

# ── Validate expected columns ──────────────────────────────────────────────────
required = {"id", "edition", "publication", "content_type", "body"}
missing  = required - set(df.columns)
if missing:
    raise ValueError(
        f"Missing columns in v2 CSV: {missing}\n"
        f"Re-run fetch_guardian.py to regenerate guardian_articles_v2.csv."
    )

# ── Summary of what was fetched ────────────────────────────────────────────────
print("\n  Breakdown by edition and content_type:")
print(df.groupby(["edition", "content_type"]).size().to_string())

# ── Parse dates, add year and era ─────────────────────────────────────────────
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["year"] = df["date"].dt.year
df["era"]  = assign_era(df["date"])

print(f"\n  Year range: {df['year'].min()} – {df['year'].max()}")
print(f"\n  Era distribution (all editions):")
print(df["era"].value_counts().reindex(
    [e[0] for e in ERA_BOUNDARIES], fill_value=0
).to_string())

# ── folder column (for compat with Australian pipeline) ───────────────────────
df["folder"] = df["edition"].map({"UK": "guardian_uk", "AU": "guardian_au"})

# ── Apply relevance scoring ────────────────────────────────────────────────────
print("\nScoring articles for climate relevance …")
df = score_articles(df)

n_relevant = (df["relevance"] == "Include").sum()
n_total    = len(df)
print(f"\n  Relevant: {n_relevant:,} / {n_total:,}  ({100*n_relevant/n_total:.1f}%)")

# ── Relevance by edition ───────────────────────────────────────────────────────
print("\n  Relevant articles by edition:")
rel = df[df["relevance"] == "Include"]
print(rel.groupby(["edition", "content_type"]).size().to_string())

# ── Relevance by era ───────────────────────────────────────────────────────────
print("\n  Relevant articles by era:")
era_counts = rel["era"].value_counts().reindex(
    [e[0] for e in ERA_BOUNDARIES], fill_value=0
)
print(era_counts.to_string())

# ── Inclusion reason breakdown ─────────────────────────────────────────────────
inc = df[df["relevance"] == "Include"]
reason_a = (inc["cg_total"] >= 3).sum()
reason_b = ((inc["cg_total"] < 3) & inc["title_hit"]).sum()
reason_c = ((inc["cg_total"] < 3) & ~inc["title_hit"] & (inc["climate_mentions"] >= 3)).sum()
print(f"\n  Inclusion reasons:")
print(f"    (a) Core phrase count >= 3:  {reason_a:,}")
print(f"    (b) Core phrase in title:    {reason_b:,}")
print(f"    (c) Broad vocabulary match:  {reason_c:,}")

# ── Derive final_status ────────────────────────────────────────────────────────
def final_status(row) -> str:
    if not isinstance(row["body"], str) or not row["body"].strip():
        return "Excluded-NoBody"
    if row["relevance"] == "Include":
        ct = row.get("content_type", "")
        if "Opinion" in ct or "Op-Ed" in ct:
            return "Included-Opinion"
        elif ct == "Analysis":
            return "Included-Analysis"
        elif ct == "Feature":
            return "Included-Feature"
        elif ct == "Editorial":
            return "Included-Editorial"
        else:
            return "Included-Other"
    return "Excluded-NotRelevant"

df["final_status"] = df.apply(final_status, axis=1)

# ── Save ───────────────────────────────────────────────────────────────────────
keep_cols = [
    "id", "title", "date", "year", "era", "author", "section",
    "edition", "publication", "content_type", "folder", "url",
    "cc_count", "gw_count", "cg_total", "climate_mentions",
    "title_hit", "relevance", "final_status", "body",
]
keep_cols = [c for c in keep_cols if c in df.columns]

scored_path   = OUT_DIR / "guardian_articles_scored.csv"
relevant_path = OUT_DIR / "guardian_relevant.csv"

df[keep_cols].to_csv(scored_path, index=False)
print(f"\nSaved all scored articles → {scored_path.relative_to(ROOT)}")

# Exclude letters and empty-body articles from the BERTopic input.
# Letters are retained in the scored CSV for completeness but excluded from
# topic modelling and all downstream analyses.
# Empty-body articles (cartoons, videos, audio embeds) pass the keyword
# relevance filter via their title but have no text to embed — excluding
# them ensures the BERTopic input count matches topic_assignments output.
MIN_BODY_LEN = 50  # characters; filters multimedia stubs

relevant_mask = (
    (df["relevance"] == "Include") &
    (df["content_type"] != "Letters") &
    (df["body"].fillna("").str.len() >= MIN_BODY_LEN)
)
n_relevant_no_letters = relevant_mask.sum()
n_letters_excl  = ((df["relevance"] == "Include") & (df["content_type"] == "Letters")).sum()
n_no_body_excl  = ((df["relevance"] == "Include") & (df["content_type"] != "Letters") &
                   (df["body"].fillna("").str.len() < MIN_BODY_LEN)).sum()

df[relevant_mask][keep_cols].to_csv(relevant_path, index=False)
print(f"Saved relevant articles   → {relevant_path.relative_to(ROOT)}")
print(f"  Letters excluded:        {n_letters_excl:,}")
print(f"  Empty/stub body excl.:   {n_no_body_excl:,}")
print(f"  Final BERTopic input:    {n_relevant_no_letters:,}")

print(f"\nDone.  {n_relevant_no_letters:,} articles ready for BERTopic.")
