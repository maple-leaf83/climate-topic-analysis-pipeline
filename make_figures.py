"""
make_figures.py
Generates Figures 2a and 2b for "A Climate of Opinion"
Reads data/articles_scored.csv (Include articles only) and
data/letters/topic_assignments.csv for the letters series.
Outputs PDF to figures/

Usage:
    python make_figures.py

Requirements:
    pip install pandas matplotlib numpy
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

from config import CATALOGUE_CSV, FIGURES_DIR, DATA_DIR

FIGURES_DIR.mkdir(exist_ok=True)

# ── Load editorial/opinion corpus ─────────────────────────────────────────────
df_all = pd.read_csv(CATALOGUE_CSV)
df = df_all[df_all["final_status"].str.contains("Include")].copy()

# ── Load letters corpus ───────────────────────────────────────────────────────
LETTERS_CSV = DATA_DIR / "letters" / "topic_assignments.csv"
df_letters = pd.read_csv(LETTERS_CSV) if LETTERS_CSV.exists() else pd.DataFrame()
if not df_letters.empty:
    df_letters["year"] = pd.to_numeric(df_letters["year"], errors="coerce")
    df_letters = df_letters[df_letters["year"].between(1987, 2026)].copy()
    print(f"[letters] {len(df_letters):,} letters loaded")


def norm_pub(p: str):
    p = str(p)
    if "Sydney Morning Herald" in p or "Sun Herald" in p: return "SMH"
    if "Age, The" in p or "The Age" in p:               return "The Age"
    if "Canberra Times" in p:                            return "Canberra Times"
    if "Guardian" in p:                                  return "Guardian"
    if "Australian" in p:                                return "The Australian"
    return None


df["pub"] = df["publication"].apply(norm_pub)
df = df[df["pub"].notna()].copy()
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df[df["year"].between(1987, 2026)].copy()

# articles_scored.csv is the source of truth — Guardian Australia news articles
# are already set to Excluded-ScopeNews in the catalogue. No additional filtering needed.
print(f"[corpus] {len(df):,} articles loaded from catalogue (final_status includes Include)")

# Consistent palette & order
PUBS   = ["Guardian", "The Age", "SMH", "Canberra Times", "The Australian"]
COLORS = ["#2166ac", "#4dac26", "#d6604d", "#8073ac", "#b35806"]
PUB_COLOR = dict(zip(PUBS, COLORS))

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

# Australian federal elections
ELECTIONS = [1996, 1998, 2001, 2004, 2007, 2010, 2013, 2016, 2019, 2022, 2025]


# ── Figure 2a: Included articles by source ────────────────────────────────────
def fig2a():

    counts = df["pub"].value_counts().reindex(PUBS).fillna(0).astype(int)
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(PUBS, counts.values,
                  color=[PUB_COLOR[p] for p in PUBS],
                  edgecolor="white", linewidth=0.5, width=0.55)
    ax.bar_label(bars, labels=[f"{v:,}" for v in counts.values], padding=4, fontsize=9)
    ax.set_ylabel("Number of articles (analysis corpus)")
    ax.set_xlabel("Publication")
    # ax.set_title("Figure 2a — Analysis corpus articles by source")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, counts.max() * 1.12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2a_articles_by_source.pdf", bbox_inches="tight")
    plt.close()
    print("Fig 2a saved")


# ── Figure 2b: Temporal distribution by source ───────────────────────────────
def fig2b():
    years = sorted(df["year"].dropna().unique().astype(int))
    year_range = list(range(min(years), max(years) + 1))

    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Editorial/opinion lines — one per publication
    for pub, color in zip(PUBS, COLORS):
        sub = df[df["pub"] == pub]
        if len(sub) == 0:
            continue
        yearly = sub.groupby("year").size().reindex(year_range, fill_value=0)
        pct = yearly / len(sub) * 100
        ax.plot(year_range, pct, color=color, linewidth=1.8, label=pub, alpha=0.9)
        ax.fill_between(year_range, pct, alpha=0.08, color=color)

    # Letters line — all Australian titles pooled, dashed
    if not df_letters.empty:
        yearly_l = df_letters.groupby("year").size().reindex(year_range, fill_value=0)
        pct_l = yearly_l / len(df_letters) * 100
        ax.plot(year_range, pct_l, color="#555555", linewidth=1.4,
                linestyle="--", label="Letters (all AU)", alpha=0.85)

    for yr in ELECTIONS:
        if yr in year_range:
            ax.axvline(yr, color="grey", linewidth=0.6, linestyle=":", alpha=0.7)
    ax.text(ELECTIONS[0], ax.get_ylim()[1] * 0.95, "Federal elections →",
            fontsize=7, color="grey", va="top")

    ax.set_xlabel("Year")
    ax.set_ylabel("% of each source's total articles")
    ax.set_xlim(min(year_range), max(year_range))
    ax.set_xticks(range(min(year_range), max(year_range) + 1, 2))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2b_timeline_by_source.pdf", bbox_inches="tight")
    plt.close()
    print("Fig 2b saved")


# ── Figure 2c: Content type composition by source ────────────────────────────
def fig2c():
    CT_MAP = {
        "Editorial":     "Editorial",
        "Columnist":     "Columnist",
        "Opinion/Op-Ed": "Op-Ed / External",
        "Analysis":      "Analysis",
        "Letters" : "Letters",
    }
    # AU news already excluded from df; Letters already excluded from Australian pubs.
    # For Guardian, use section column to correctly label Environment → Analysis,
    # Opinion → Op-Ed / External.
    CT_ORDER  = ["Editorial", "Columnist", "Op-Ed / External", "Analysis", "Letters"]
    CT_COLORS = ["#1b7837", "#762a83", "#e08214", "#2166ac", "#aaaaaa"]
    CT_COLOR  = dict(zip(CT_ORDER, CT_COLORS))

    df_fig = df.copy()
    is_guardian = df_fig["pub"] == "Guardian"
    df_fig["ct_group"] = df_fig["content_type"].map(CT_MAP).fillna("Other")
    if "section" in df_fig.columns:
        df_fig.loc[is_guardian & (df_fig["section"] == "Environment"), "ct_group"] = "Analysis"
        df_fig.loc[is_guardian & (df_fig["section"] == "Opinion"),     "ct_group"] = "Op-Ed / External"

    mat = pd.crosstab(df_fig["pub"], df_fig["ct_group"]).reindex(
        index=PUBS, columns=CT_ORDER, fill_value=0
    )
    mat_pct = mat.div(mat.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(len(PUBS))
    for ct in CT_ORDER:
        vals = mat_pct[ct].values
        if vals.sum() == 0:
            continue
        bars = ax.bar(PUBS, vals, bottom=bottom,
                      color=CT_COLOR[ct], label=ct,
                      edgecolor="white", linewidth=0.4, width=0.6)
        for rect, val, bot in zip(bars, vals, bottom):
            if val > 5:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        bot + val / 2, f"{val:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
        bottom += vals

    ax.set_ylabel("% of included articles")
    ax.set_xlabel("Publication")
    # ax.set_title("Figure 2c — Content type composition by source")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)

    pub_ns = mat.sum(axis=1)
    ax.set_xticks(range(len(PUBS)))
    ax.set_xticklabels([f"{p}\n(n={pub_ns[p]:,})" for p in PUBS], fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2c_content_type_by_source.pdf", bbox_inches="tight")
    plt.close()
    print("Fig 2c saved")


if __name__ == "__main__":
    fig2a()
    fig2b()
    print(f"\nAll figures saved to {FIGURES_DIR}/")
