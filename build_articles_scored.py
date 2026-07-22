#!/usr/bin/env python3
"""
build_articles_scored.py
------------------------
Reconstructs the ground-truth inclusion/exclusion record for
"A Climate of Opinion" from the raw body-text sources.

Pipeline
--------
1. Load NewsBank body cache (newsbank_bodies.pkl.gz) — assign content_type
   and publication from NEWSBANK_FOLDERS config.
2. Load Guardian articles (guardian_articles.csv).
3. Combine and deduplicate by (title, date, publication).
4. Score each article for climate relevance (score_and_classify.py).
5. Apply two-step exclusion:
     Step 1 — Relevance:  keyword criterion (see config.py)
     Step 2 — Scope:      Guardian "Australia news" section excluded as
                          straight news reporting; other included articles
                          split into editorials vs. letters.
6. Write repo/data/articles_scored.csv  (one row per article).
7. Print a PRISMA-style summary.

Output columns
--------------
  title, author, publication, section, content_type, date, year, folder,
  cc_count, gw_count, cg_total, climate_mentions, title_hit,
  relevance_step1   : Include / Exclude
  scope_step2       : Editorial / Letter / Excluded-News / —
  final_status      : one of:
                        Included-Editorial
                        Included-Letter
                        Excluded-NotRelevant
                        Excluded-ScopeNews

Usage
-----
    python build_articles_scored.py

Paths default to siblings of the repo folder; override with env vars:
    GUARDIAN_CSV    path to guardian_articles.csv
    NB_BODIES       path to newsbank_bodies.pkl.gz
"""

import gzip
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
REPO      = Path(__file__).parent
DATA_OUT  = REPO / "data"
PARENT    = REPO.parent

GUARDIAN_CSV  = Path(os.environ.get("GUARDIAN_CSV",  PARENT / "guardian_articles.csv"))
NB_BODIES     = Path(os.environ.get("NB_BODIES",     DATA_OUT / "newsbank_bodies.pkl.gz"))
OUT_CSV       = DATA_OUT / "articles_scored.csv"

DATA_OUT.mkdir(exist_ok=True)

# ── Import project modules ────────────────────────────────────────────────────
sys.path.insert(0, str(REPO))
from config import NEWSBANK_FOLDERS, CT_COLUMNISTS
from score_and_classify import score_articles


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load NewsBank
# ══════════════════════════════════════════════════════════════════════════════

def load_newsbank(path: Path) -> pd.DataFrame:
    print(f"[newsbank] Loading {path} …")
    with gzip.open(path, "rb") as f:
        df = pickle.load(f)

    # Map folder → publication and content_type from config
    folder_pub  = {k: v["publication"]  for k, v in NEWSBANK_FOLDERS.items()}
    folder_ct   = {k: v["content_type"] for k, v in NEWSBANK_FOLDERS.items()}

    # Strip trailing path separators from folder names for matching
    df["folder_key"] = df["folder"].apply(lambda x: str(x).rstrip("/\\").split("/")[-1])

    df["publication"]  = df["folder_key"].map(folder_pub).fillna(df.get("publication", "Unknown"))
    df["content_type"] = df["folder_key"].map(folder_ct)

    # Auto-classify CT_Opinion and CT_opinion_analysis by author name
    ct_mask = df["content_type"].isna()
    if ct_mask.any():
        def _classify_ct(row):
            author = str(row.get("author", "") or "").lower()
            if any(c in author for c in CT_COLUMNISTS):
                return "Columnist"
            body = str(row.get("body", "") or "")
            # Letters are typically short and start with "Dear" or "Sir"
            if len(body.split()) < 250 and any(
                body.strip().lower().startswith(s)
                for s in ("dear", "sir,", "madam", "to the editor", "i write")
            ):
                return "Letters"
            return "Opinion/Op-Ed"

        df.loc[ct_mask, "content_type"] = df[ct_mask].apply(_classify_ct, axis=1)

    df["section"]     = df["content_type"]   # NewsBank has no section; use content_type
    df["author"]      = df.get("author", None)
    df["word_count"]  = df["body"].apply(lambda x: len(str(x or "").split()))
    df["year"]        = pd.to_datetime(df["date"], errors="coerce").dt.year

    print(f"[newsbank] {len(df):,} articles across "
          f"{df['publication'].nunique()} publications")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Load Guardian
# ══════════════════════════════════════════════════════════════════════════════

def load_guardian(path: Path) -> pd.DataFrame:
    print(f"[guardian] Loading {path} …")
    df = pd.read_csv(path)

    df["publication"]  = "The Guardian"
    df["folder"]       = "guardian_api"
    df["filename"]     = df.get("url", df.get("id", ""))
    df["content_type"] = df["section"].apply(
        lambda s: "Analysis" if str(s).lower() == "environment"
        else "Opinion/Op-Ed" if str(s).lower() in ("opinion", "commentisfree")
        else "News"
    )
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year

    print(f"[guardian] {len(df):,} articles — sections: "
          f"{df['section'].value_counts().to_dict()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Combine & deduplicate
# ══════════════════════════════════════════════════════════════════════════════

KEEP_COLS = [
    "title", "author", "publication", "section", "content_type",
    "date", "year", "folder", "filename", "word_count", "body",
]

def combine(nb: pd.DataFrame, guardian: pd.DataFrame) -> pd.DataFrame:
    for col in KEEP_COLS:
        if col not in nb.columns:
            nb[col] = None
        if col not in guardian.columns:
            guardian[col] = None

    combined = pd.concat([nb[KEEP_COLS], guardian[KEEP_COLS]], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["title", "date", "publication"])
    print(f"[dedup] {before:,} → {len(combined):,} "
          f"({before - len(combined):,} duplicates removed)")
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 4 & 5. Score relevance + apply two-step exclusion
# ══════════════════════════════════════════════════════════════════════════════

MIN_BODY_CHARS = 150   # articles shorter than this are multimedia/stub entries
YEAR_MIN      = 1987
YEAR_MAX      = 2026

def apply_exclusions(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure body and title are strings (NaN-safe)
    df["body"]  = df["body"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)

    # Step 0a — year-range gate: exclude articles outside the corpus window.
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    out_of_scope_mask = ~df["year"].between(YEAR_MIN, YEAR_MAX)
    n_oos = out_of_scope_mask.sum()
    if n_oos:
        print(f"[year]  Step 0a — {n_oos:,} articles outside {YEAR_MIN}–{YEAR_MAX} "
              f"→ Excluded-OutOfScope")

    # Step 0b — body-length gate: videos, cartoons, stubs have no scoreable text.
    # Mark these now so they cannot pass via title-hit alone.
    no_body_mask = (~out_of_scope_mask) & (df["body"].str.len() < MIN_BODY_CHARS)
    n_no_body = no_body_mask.sum()
    if n_no_body:
        print(f"[body]  Step 0b — {n_no_body:,} articles below {MIN_BODY_CHARS}-char "
              f"body threshold → Excluded-NoBody")

    # Step 1 — relevance scoring (vectorised, single-pass)
    # Articles failing the body-length gate are forced to Exclude regardless of title.
    print("[score] Applying relevance criterion …")
    import re
    from config import (CLIMATE_TERMS, CORE_CLIMATE_PHRASES,
                        CC_CORE_THRESHOLD, CLIMATE_MENTIONS_THRESHOLD)

    # Single pattern matching any core climate identifier (word-boundary aware).
    # Covers canonical phrases, contemporary equivalents, scientific mechanism terms,
    # policy mechanisms, and international framework terms — see config.py.
    _core_pat = re.compile(
        "|".join(r"\b" + re.escape(t) + r"\b" for t in CORE_CLIMATE_PHRASES),
        re.IGNORECASE,
    )
    # Retain separate cc/gw patterns so legacy columns (cc_count, gw_count) are
    # still populated for traceability and backward compatibility.
    _cc_pat = re.compile(r"\bclimate\s+change\b", re.IGNORECASE)
    _gw_pat = re.compile(r"\bglobal\s+warming\b", re.IGNORECASE)
    # Full climate vocabulary for criterion (c).
    # Left-boundary only: prevents "coal" matching "coalition" etc., while
    # preserving prefix terms ("decarboni" → decarbonise/ization, "adapt" → adaptation).
    _cm_pat = re.compile(
        "|".join(r"\b" + re.escape(t) for t in CLIMATE_TERMS),
        re.IGNORECASE,
    )

    bodies     = df["body"].tolist()
    titles     = df["title"].tolist()
    no_body_ix    = set(df.index[no_body_mask])
    out_of_scope_ix = set(df.index[out_of_scope_mask])

    cc_list, gw_list, cg_list, cm_list, th_list, rel_list = [], [], [], [], [], []
    for i, (idx, body, title) in enumerate(zip(df.index, bodies, titles)):
        # Force-exclude out-of-scope and no-body articles
        if idx in out_of_scope_ix or idx in no_body_ix:
            cc_list.append(0); gw_list.append(0); cg_list.append(0)
            cm_list.append(0); th_list.append(False)
            rel_list.append("Exclude")
            continue

        title_lower = title.lower()
        th  = bool(_core_pat.search(title_lower))   # any core phrase in title
        cc  = len(_cc_pat.findall(body))             # legacy column
        gw  = len(_gw_pat.findall(body))             # legacy column
        cg  = len(_core_pat.findall(body))           # core phrase count (all 14 phrases)
        # Only compute climate_mentions if needed (cg < threshold and no title hit)
        if cg >= CC_CORE_THRESHOLD or th:
            cm  = 0   # already included — no need to count
            inc = True
        else:
            cm  = len(_cm_pat.findall(body))
            inc = cg >= 1 and cm >= CLIMATE_MENTIONS_THRESHOLD

        cc_list.append(cc)
        gw_list.append(gw)
        cg_list.append(cg)
        cm_list.append(cm)
        th_list.append(th)
        rel_list.append("Include" if inc else "Exclude")

        if (i + 1) % 10_000 == 0:
            print(f"  … {i+1:,} / {len(bodies):,}")

    df["cc_count"]         = cc_list
    df["gw_count"]         = gw_list
    df["cg_total"]         = cg_list
    df["climate_mentions"] = cm_list
    df["title_hit"]        = th_list
    df["relevance"]        = rel_list

    n_inc = (df["relevance"] == "Include").sum()
    n_exc = (df["relevance"] == "Exclude").sum()
    print(f"[score] Step 1 — Include: {n_inc:,}  |  Exclude (not relevant): {n_exc:,}")

    # Step 2 — scope restriction
    # Guardian "Australia news" section = straight news reporting → exclude
    guardian_news_mask = (
        (df["publication"] == "The Guardian") &
        (df["section"].str.lower().isin(["australia news", "australia-news"]))
    )

    # Letters are included but tracked separately.
    # Primary signal: content_type tag from NewsBank / config.py folder mapping.
    # Secondary signal: title pattern — catches cases where a letters-page was
    # exported from a non-Letters folder (e.g. TheAge_Analysis/Missingyears)
    # and inherited an incorrect content_type ("Opinion/Op-Ed" or "Analysis").
    _letter_title = df["title"].str.strip().str.lower().str.match(
        r"^(letters?|letters?\s*[-–&]\s*|feedback|letters?\s*&\s*emails?)"
    )
    letters_mask = (
        (df["relevance"] == "Include") &
        (~guardian_news_mask) &
        (
            df["content_type"].str.lower().isin(["letters", "letter"]) |
            _letter_title.fillna(False)
        )
    )

    editorial_mask = (
        (df["relevance"] == "Include") &
        (~guardian_news_mask) &
        (~letters_mask)
    )

    # Assign final_status
    def _status(row_idx):
        if out_of_scope_mask.loc[row_idx]:
            return "Excluded-OutOfScope"
        if no_body_mask.loc[row_idx]:
            return "Excluded-NoBody"
        if df.loc[row_idx, "relevance"] == "Exclude":
            return "Excluded-NotRelevant"
        if guardian_news_mask.loc[row_idx]:
            return "Excluded-ScopeNews"
        if letters_mask.loc[row_idx]:
            return "Included-Letter"
        return "Included-Editorial"

    df["final_status"] = [_status(i) for i in df.index]

    n_oos_excl     = (df["final_status"] == "Excluded-OutOfScope").sum()
    n_no_body_excl = (df["final_status"] == "Excluded-NoBody").sum()
    n_news         = guardian_news_mask.sum()
    n_letters      = letters_mask.sum()
    n_ed           = (df["final_status"] == "Included-Editorial").sum()
    print(f"[year]   Excluded (out of scope):   {n_oos_excl:,}")
    print(f"[body]   Excluded (no/short body):  {n_no_body_excl:,}")
    print(f"[scope]  Step 2 — Excluded (news): {n_news:,}  |  "
          f"Included-Editorial: {n_ed:,}  |  Included-Letter: {n_letters:,}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 6. Save
# ══════════════════════════════════════════════════════════════════════════════

SAVE_COLS = [
    "title", "author", "publication", "section", "content_type",
    "date", "year", "folder",
    "cc_count", "gw_count", "cg_total", "climate_mentions", "title_hit",
    "relevance", "final_status",
    "body",   # body text retained so run_bertopic.py never needs to re-merge source files
]

def save(df: pd.DataFrame, path: Path):
    out = df[[c for c in SAVE_COLS if c in df.columns]]
    out.to_csv(path, index=False)
    print(f"\n[done] Saved {len(out):,} articles → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. PRISMA summary
# ══════════════════════════════════════════════════════════════════════════════

def prisma_summary(df: pd.DataFrame):
    W = 66
    SEP = "─" * (W - 2)

    print("\n" + "═" * W)
    print(" PRISMA-STYLE SUMMARY")
    print("═" * W)

    # ── Core counts ───────────────────────────────────────────────────────────
    g_df  = df[df["folder"] == "guardian_api"].copy()
    nb_df = df[df["folder"] != "guardian_api"].copy()

    g_total    = len(g_df)
    nb_total   = len(nb_df)
    pre_screen = len(df)

    excl_rel  = (df["final_status"] == "Excluded-NotRelevant").sum()
    excl_news = (df["final_status"] == "Excluded-ScopeNews").sum()
    inc_ed    = (df["final_status"] == "Included-Editorial").sum()
    inc_let   = (df["final_status"] == "Included-Letter").sum()

    # Guardian section breakdown
    g_df["section_norm"] = g_df["section"].str.lower().str.strip()
    G_SECTIONS = {
        "environment":    "Environment desk",
        "opinion":        "Opinion (Comment is Free)",
        "australia news": "Australia news",
    }

    # ── STAGE 1: Identification ───────────────────────────────────────────────
    print(f"\n  STAGE 1 — IDENTIFICATION")
    print(f"  {SEP}")
    print(f"  {'Guardian API (3 sections):':<48} {g_total:>6,}")
    for sec_key, sec_label in G_SECTIONS.items():
        n = (g_df["section_norm"] == sec_key).sum()
        print(f"    {sec_label:<46} {n:>6,}")
    other_g = (~g_df["section_norm"].isin(G_SECTIONS)).sum()
    if other_g:
        print(f"    {'Other':<46} {other_g:>6,}")

    print(f"  {'NewsBank PDFs:':<48} {nb_total:>6,}")
    nb_pub_order = ["Sydney Morning Herald", "The Age", "Canberra Times"]
    nb_pubs = nb_df["publication"].value_counts()
    for pub in nb_pub_order:
        if pub in nb_pubs.index:
            print(f"    {pub:<46} {nb_pubs[pub]:>6,}")
    for pub, n in nb_pubs.items():
        if pub not in nb_pub_order:
            print(f"    {pub:<46} {n:>6,}")

    print(f"  {SEP}")
    print(f"  {'Combined pre-screening total (after deduplication):':<48} {pre_screen:>6,}")

    # ── STAGE 2: Relevance screening ─────────────────────────────────────────
    passed_rel = pre_screen - excl_rel
    print(f"\n  STAGE 2 — RELEVANCE SCREENING")
    print(f"  {SEP}")
    print(f"  Criterion: core_phrases ≥ 3  OR  core in title  OR  core ≥ 1 AND climate_mentions ≥ 3")
    print(f"  {'Excluded — not climate-relevant:':<48} {excl_rel:>6,}")
    print(f"  {'Passed relevance screen:':<48} {passed_rel:>6,}")

    # ── STAGE 3: Scope restriction ────────────────────────────────────────────
    au_news_total = (g_df["section_norm"] == "australia news").sum()
    au_news_rel   = (
        (g_df["section_norm"] == "australia news") &
        (g_df["relevance"] == "Include")
    ).sum()
    au_news_notrel = au_news_total - au_news_rel

    print(f"\n  STAGE 3 — SCOPE RESTRICTION")
    print(f"  {SEP}")
    print(f"  Guardian Australia news: {au_news_total:,} total")
    print(f"    {au_news_notrel:,} excluded at Stage 2 (not climate-relevant)")
    print(f"    {au_news_rel:,} passed Stage 2 → excluded here as straight news reporting")
    print(f"  {'Excluded — Guardian Australia news (scope):':<48} {excl_news:>6,}")
    print(f"  {'Remaining after scope restriction:':<48} {passed_rel - excl_news:>6,}")

    # ── STAGE 4: Included — per-publication table ────────────────────────────
    print(f"\n  STAGE 4 — INCLUDED")
    print(f"  {SEP}")
    col_w = 28
    print(f"  {'Publication':<{col_w}} {'Pre-screen':>10} {'Editorial':>10} "
          f"{'Letter':>8} {'Total':>7}")
    print(f"  {SEP}")

    PUB_ORDER = ["The Guardian", "The Age", "Sydney Morning Herald", "Canberra Times"]
    all_pubs  = df["publication"].value_counts().index.tolist()
    ordered   = [p for p in PUB_ORDER if p in all_pubs] + \
                [p for p in all_pubs if p not in PUB_ORDER]

    grand = {"pre": 0, "ed": 0, "let": 0}
    for pub in ordered:
        pub_df = df[df["publication"] == pub]
        pre    = len(pub_df)
        ed     = (pub_df["final_status"] == "Included-Editorial").sum()
        let    = (pub_df["final_status"] == "Included-Letter").sum()
        grand["pre"] += pre
        grand["ed"]  += ed
        grand["let"] += let
        print(f"  {pub:<{col_w}} {pre:>10,} {ed:>10,} {let:>8,} {ed+let:>7,}")

    print(f"  {SEP}")
    print(f"  {'TOTAL':<{col_w}} {grand['pre']:>10,} {grand['ed']:>10,} "
          f"{grand['let']:>8,} {grand['ed']+grand['let']:>7,}")
    print("═" * W)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not NB_BODIES.exists():
        sys.exit(f"NewsBank bodies not found at {NB_BODIES}")
    if not GUARDIAN_CSV.exists():
        sys.exit(f"Guardian CSV not found at {GUARDIAN_CSV}")

    nb       = load_newsbank(NB_BODIES)
    guardian = load_guardian(GUARDIAN_CSV)
    combined = combine(nb, guardian)
    scored   = apply_exclusions(combined)
    save(scored, OUT_CSV)
    prisma_summary(scored)


if __name__ == "__main__":
    main()
