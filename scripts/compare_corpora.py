"""
compare_corpora.py
──────────────────
Two analyses comparing the Australian broadsheet corpus and the Guardian corpus:

  1. NATIONAL COMPARISON
     Topic-group proportions for each corpus, chi-square test, and a side-by-side
     bar chart using a harmonized taxonomy.

  2. SCIENCE ABSORPTION COMPARISON
     Within the political articles of each corpus, measures how often climate
     science vocabulary appears, tracked by government era.  If science vocabulary
     is increasingly embedded inside political discourse in the Australian papers
     but remains in science/ecology topics in the Guardian, that supports the
     "science absorbed into partisan argument" thesis.

Outputs (written to figures/comparison/):
  corpus_comparison_barchart.pdf/.png
  science_absorption_comparison.pdf/.png
  corpus_comparison_stats.csv

Usage:
  python compare_corpora.py
"""

import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
def chi2_contingency(table):
    """Minimal chi-square test for a 2-row contingency table."""
    import math
    table = np.array(table, dtype=float)
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    total    = table.sum()
    expected = row_sums * col_sums / total
    chi2 = float(((table - expected) ** 2 / expected).sum())
    dof  = (table.shape[0] - 1) * (table.shape[1] - 1)
    # p-value via incomplete gamma (approximation using chi2 CDF)
    # Use a simple regularised incomplete gamma for reporting
    try:
        from math import lgamma, exp, log
        def chi2_sf(x, k):
            """Survival function P(X > x) for chi2(k) — series expansion."""
            if x <= 0:
                return 1.0
            # Use regularized upper incomplete gamma: Q(k/2, x/2)
            a, z = k / 2, x / 2
            # Simple upper tail via log-sum for large values
            term = exp(a * log(z) - z - lgamma(a + 1))
            s = term
            for i in range(1, 200):
                term *= z / (a + i)
                s += term
                if term < 1e-10 * s:
                    break
            return min(1.0, max(0.0, 1 - s))
        p = chi2_sf(chi2, dof)
    except Exception:
        p = float("nan")
    return chi2, p, dof, expected

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO        = Path(__file__).parent
DATA_DIR    = REPO / "data"
FIGURES_DIR = REPO / "figures" / "comparison"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

AU_ASSIGNMENTS  = DATA_DIR / "australian-no-letters" / "topic_assignments_aus.csv"
AU_SUMMARY      = DATA_DIR / "australian-no-letters" / "topic_summary_aus.csv"
AU_SCORED       = DATA_DIR / "articles_scored_australian.csv"

G_ASSIGNMENTS   = DATA_DIR / "guardian" / "topic_assignments_guardian.csv"
G_SUMMARY       = DATA_DIR / "guardian" / "topic_summary_guardian.csv"
G_RELEVANT      = DATA_DIR / "guardian" / "guardian_relevant.csv"

# ── Harmonized taxonomy ────────────────────────────────────────────────────────
# Maps corpus-specific group labels → a shared super-category for comparison.

G_TO_HARMONIZED = {
    "Australian Politics":                   "Domestic Politics & Policy",
    "UK & European Politics":                "Domestic Politics & Policy",
    "North American & International Politics":"International Politics",
    "International Climate Policy":          "International Climate Policy",
    "Climate Science & Scepticism":          "Climate Science",
    "Physical & Ecological Impacts":         "Physical & Ecological Impacts",
    "Energy Transition & Technology":        "Energy & Industry",
    "Activism Culture & Society":            "Culture, Media & Activism",
}

AU_TO_HARMONIZED = {
    "Australian Politics":                       "Domestic Politics & Policy",
    "Carbon Pricing & Domestic Climate Policy":  "Domestic Politics & Policy",
    "Regional & International Politics":         "International Politics",
    "International Climate Policy":              "International Climate Policy",
    "Climate Science":                           "Climate Science",
    "Physical & Ecological Impacts":             "Physical & Ecological Impacts",
    "Energy Transition & Technology":            "Energy & Industry",
    "Culture Media & Society":                   "Culture, Media & Activism",
}

HARMONIZED_ORDER = [
    "Domestic Politics & Policy",
    "International Politics",
    "International Climate Policy",
    "Climate Science",
    "Physical & Ecological Impacts",
    "Energy & Industry",
    "Culture, Media & Activism",
]

from config import HARMONIZED_COLORS, OUTLET_COLORS

# ── Government eras (for science absorption temporal axis) ─────────────────────
ERA_ORDER = ["Pre-Howard", "Howard", "Rudd/Gillard", "Abbott",
             "Turnbull/Morrison", "Albanese"]

# ── Climate science vocabulary ─────────────────────────────────────────────────
# Terms that specifically signal scientific/empirical climate content
# (as opposed to political or policy terms).
SCIENCE_VOCAB = [
    "ipcc", "attribution", "temperature record", "sea level",
    "global temperature", "greenhouse gas", "carbon dioxide", "methane",
    "radiative forcing", "climate sensitivity", "paleoclimate",
    "ocean acidification", "ice sheet", "permafrost", "feedback",
    "tipping point", "extreme weather event", "heat wave", "heatwave",
    "attribution science", "warming trend", "climate model",
    "scientific consensus", "peer.reviewed", "scientific evidence",
    "degrees celsius", "degrees of warming", "pre.industrial",
]
SCIENCE_PATTERN = re.compile(
    r'\b(?:' + '|'.join(re.escape(t) for t in SCIENCE_VOCAB) + r')\b',
    flags=re.IGNORECASE,
)

# ── Political group labels (within each corpus) ────────────────────────────────
AU_POLITICAL_THEMES = {
    "Australian Politics",
    "Carbon Pricing & Domestic Climate Policy",
    "Regional & International Politics",
}
G_POLITICAL_GROUPS = {
    "Australian Politics",
    "UK & European Politics",
    "North American & International Politics",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. NATIONAL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

def load_au_assignments():
    df = pd.read_csv(AU_ASSIGNMENTS, low_memory=False)
    df = df[df["topic_id"] != -1].copy()
    df["harmonized"] = df["group"].map(AU_TO_HARMONIZED)
    return df.dropna(subset=["harmonized"])


def load_g_assignments():
    df = pd.read_csv(G_ASSIGNMENTS, low_memory=False)
    df = df[df["topic_id"] != -1].copy()
    df = df[df["content_type"] != "Letters"].copy()   # exclude letters from all analyses
    df["harmonized"] = df["group"].map(G_TO_HARMONIZED)
    return df.dropna(subset=["harmonized"])


def national_comparison(au_df: pd.DataFrame, g_df: pd.DataFrame):
    print("\n" + "="*60)
    print("  NATIONAL COMPARISON")
    print("="*60)

    # Proportions
    au_pct = (au_df["harmonized"].value_counts() /
              len(au_df) * 100).reindex(HARMONIZED_ORDER, fill_value=0)
    g_pct  = (g_df["harmonized"].value_counts() /
              len(g_df) * 100).reindex(HARMONIZED_ORDER, fill_value=0)

    print("\nHarmonized group proportions (%):")
    comp = pd.DataFrame({"Australian broadsheets": au_pct, "The Guardian": g_pct})
    print(comp.to_string(float_format="{:.1f}".format))

    # Chi-square on raw counts
    au_counts = au_df["harmonized"].value_counts().reindex(HARMONIZED_ORDER, fill_value=0)
    g_counts  = g_df["harmonized"].value_counts().reindex(HARMONIZED_ORDER, fill_value=0)
    contingency = np.array([au_counts.values, g_counts.values])
    chi2, p, dof, _ = chi2_contingency(contingency)
    print(f"\nChi-square test: χ²({dof}) = {chi2:.1f}, p = {p:.2e}")

    # Save stats
    comp["chi2"] = chi2
    comp["p"] = p
    comp.to_csv(FIGURES_DIR / "corpus_comparison_stats.csv")
    print(f"Stats saved → figures/comparison/corpus_comparison_stats.csv")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(HARMONIZED_ORDER))
    width = 0.35

    bars_au = ax.bar(x - width/2, au_pct.values, width,
                     label="Australian broadsheets",
                     color=[HARMONIZED_COLORS[g] for g in HARMONIZED_ORDER],
                     alpha=0.9, edgecolor="white")
    bars_g  = ax.bar(x + width/2, g_pct.values, width,
                     label="The Guardian",
                     color=[HARMONIZED_COLORS[g] for g in HARMONIZED_ORDER],
                     alpha=0.45, edgecolor="white", hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(HARMONIZED_ORDER, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("% of corpus articles")
    ax.set_title("Thematic distribution: Australian broadsheets vs The Guardian",
                 fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    for ext in (".pdf", ".png"):
        stem = FIGURES_DIR / "corpus_comparison_barchart"
        fig.savefig(stem.with_suffix(ext), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Plot saved → figures/comparison/corpus_comparison_barchart.pdf/.png")

    return comp


# ══════════════════════════════════════════════════════════════════════════════
# 2. CROSS-CORPUS KEYWORD CO-OCCURRENCE IN POLITICAL ARTICLES
#
# For each corpus's political articles, measures what % contain keywords from
# each non-political thematic group — using that corpus's own BERTopic keywords.
# Tracks by government era.  Side-by-side heatmaps allow direct comparison of
# which themes bleed into political coverage in each corpus.
# ══════════════════════════════════════════════════════════════════════════════

# Non-political groups to probe, keyed by corpus-specific group name,
# mapped to a harmonized display label for the heatmap rows.
AU_NONPOL_GROUPS = {
    "Climate Science":             "Climate Science",
    "Physical & Ecological Impacts":"Physical & Ecological Impacts",
    "Energy Transition & Technology":"Energy & Industry",
    "International Climate Policy": "International Climate Policy",
    "Culture Media & Society":      "Culture, Media & Activism",
}

G_NONPOL_GROUPS = {
    "Climate Science & Scepticism": "Climate Science",
    "Physical & Ecological Impacts":"Physical & Ecological Impacts",
    "Energy Transition & Technology":"Energy & Industry",
    "International Climate Policy": "International Climate Policy",
    "Activism Culture & Society":   "Culture, Media & Activism",
}

# Row order for heatmap (harmonized labels)
COOC_ROW_ORDER = [
    "Climate Science",
    "Physical & Ecological Impacts",
    "Energy & Industry",
    "International Climate Policy",
    "Culture, Media & Activism",
]


def build_group_pattern(summary_path: Path, assignments: pd.DataFrame,
                        group_name: str, group_col: str,
                        top_n_per_topic: int = 10) -> re.Pattern:
    """
    Build a regex from the top-N BERTopic keywords of all topics in group_name.
    Uses each corpus's own topic summary — corpus-specific vocabulary.
    """
    summary = pd.read_csv(summary_path)
    tids = set(assignments.loc[assignments[group_col] == group_name, "topic_id"].unique())
    kws = []
    for tid in tids:
        row = summary[summary["topic_id"] == tid]
        if row.empty:
            continue
        try:
            top_kws = ast.literal_eval(row.iloc[0]["keywords"])[:top_n_per_topic]
            kws.extend(top_kws)
        except Exception:
            continue
    seen: set = set()
    unique = [k for k in kws if not (k in seen or seen.add(k))]
    print(f"    {group_name}: {len(unique)} keywords — {unique[:8]} …")
    if not unique:
        return None
    return re.compile(
        r'\b(?:' + '|'.join(re.escape(k.lower()) for k in unique) + r')\b',
        flags=re.IGNORECASE,
    )


def cooccurrence_matrix(pol_articles: pd.DataFrame, bodies: pd.DataFrame,
                        merge_keys: list[str],
                        group_patterns: dict[str, re.Pattern]) -> pd.DataFrame:
    """
    For political articles merged with body text, compute % containing each
    group's keywords — by era.  Returns DataFrame: rows=harmonized groups,
    cols=eras, values=%.
    """
    merged = pol_articles.merge(bodies, on=merge_keys, how="left")
    merged = merged.dropna(subset=["body"])
    print(f"    Matched body text: {len(merged):,} / {len(pol_articles):,} political articles")

    rows = {}
    for label, pattern in group_patterns.items():
        if pattern is None:
            continue
        merged[f"_hit_{label}"] = merged["body"].str.lower().apply(
            lambda b, p=pattern: int(bool(p.search(b)))
        )
        era_rates = (merged.groupby("era")[f"_hit_{label}"]
                     .mean()
                     .reindex(ERA_ORDER)
                     .dropna() * 100)
        rows[label] = era_rates

    return pd.DataFrame(rows).T  # shape: groups × eras


def cross_corpus_cooccurrence(au_df: pd.DataFrame, g_df: pd.DataFrame):
    print("\n" + "="*60)
    print("  KEYWORD CO-OCCURRENCE IN POLITICAL ARTICLES")
    print("="*60)

    import csv
    csv.field_size_limit(10_000_000)

    # ── Load body text ─────────────────────────────────────────────────────────
    print("\n  Loading Australian bodies…")
    au_bodies = pd.read_csv(AU_SCORED, usecols=["title", "date", "publication", "body"],
                            low_memory=False)
    au_bodies["date"] = pd.to_datetime(au_bodies["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    print("  Loading Guardian bodies…")
    g_bodies = pd.read_csv(G_RELEVANT, usecols=["title", "date", "body"],
                           low_memory=False)
    g_bodies["date"] = pd.to_datetime(g_bodies["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Normalise dates in assignments
    au_df2 = au_df.copy()
    au_df2["date"] = pd.to_datetime(au_df2["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    g_df2 = g_df.copy()
    g_df2["date"] = pd.to_datetime(g_df2["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # ── Political article subsets ──────────────────────────────────────────────
    au_pol = au_df2[au_df2["group"].isin(AU_POLITICAL_THEMES)].copy()
    g_pol  = g_df2[g_df2["group"].isin(G_POLITICAL_GROUPS)].copy()
    print(f"\n  AU political articles: {len(au_pol):,}  |  G political articles: {len(g_pol):,}")

    # ── Build keyword patterns from each corpus's own BERTopic summaries ───────
    print("\n  Australian group keywords:")
    au_patterns = {
        harmonized: build_group_pattern(AU_SUMMARY, au_df2, corpus_group, "group")
        for corpus_group, harmonized in AU_NONPOL_GROUPS.items()
    }

    print("\n  Guardian group keywords:")
    g_patterns = {
        harmonized: build_group_pattern(G_SUMMARY, g_df2, corpus_group, "group")
        for corpus_group, harmonized in G_NONPOL_GROUPS.items()
    }

    # ── Compute co-occurrence matrices ─────────────────────────────────────────
    print("\n  Scoring Australian political articles…")
    au_cooc = cooccurrence_matrix(au_pol, au_bodies,
                                  merge_keys=["title", "date", "publication"],
                                  group_patterns=au_patterns)

    print("\n  Scoring Guardian political articles…")
    g_cooc = cooccurrence_matrix(g_pol, g_bodies,
                                 merge_keys=["title", "date"],
                                 group_patterns=g_patterns)

    # Reindex rows to canonical order; keep only eras each corpus actually has data for
    au_eras = [e for e in ERA_ORDER if e in au_cooc.columns and au_cooc[e].max() > 0]
    g_eras  = [e for e in ERA_ORDER if e in g_cooc.columns  and g_cooc[e].max()  > 0]
    au_cooc = au_cooc.reindex(index=COOC_ROW_ORDER, columns=au_eras).fillna(0)
    g_cooc  = g_cooc.reindex(index=COOC_ROW_ORDER,  columns=g_eras).fillna(0)

    print("\n  AU co-occurrence (% of political articles):")
    print(au_cooc.round(1).to_string())
    print("\n  Guardian co-occurrence (% of political articles):")
    print(g_cooc.round(1).to_string())

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    au_cooc.round(2).to_csv(FIGURES_DIR / "cooccurrence_au.csv")
    g_cooc.round(2).to_csv(FIGURES_DIR / "cooccurrence_guardian.csv")

    # ── Plot: side-by-side heatmaps ────────────────────────────────────────────
    import matplotlib.colors as mcolors

    # Drop Pre-Howard from AU (very small n → noisy) so both panels share the same eras
    if "Pre-Howard" in au_cooc.columns:
        au_cooc = au_cooc.drop(columns=["Pre-Howard"])

    vmax = max(au_cooc.values.max(), g_cooc.values.max())
    vmax = np.ceil(vmax / 5) * 5  # round up to nearest 5%

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"wspace": 0.08})

    def draw_heatmap(ax, data, title):
        im = ax.imshow(data.values, aspect="auto", cmap="viridis",
                       vmin=0, vmax=vmax)
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(data.index)))
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        # Annotate cells
        for r in range(len(data.index)):
            for c in range(len(data.columns)):
                val = data.values[r, c]
                color = "#333333" if val > vmax * 0.5 else "white"
                ax.text(c, r, f"{val:.0f}%", ha="center", va="center",
                        fontsize=10, color=color)
        return im

    im = draw_heatmap(axes[0], au_cooc,
                      "Australian broadsheets")
    axes[0].set_yticklabels(COOC_ROW_ORDER, fontsize=9)

    draw_heatmap(axes[1], g_cooc,
                 "The Guardian")
    axes[1].set_yticklabels([])

    cbar = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("% of political articles containing group keywords", fontsize=8.5)

    # fig.suptitle(
    #     "Thematic vocabulary co-occurrence in political articles by era",
    #     fontsize=11, fontweight="bold", y=1.02,
    # )

    for ext in (".pdf", ".png"):
        stem = FIGURES_DIR / "keyword_cooccurrence_comparison"
        fig.savefig(stem.with_suffix(ext), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nPlot saved → figures/comparison/keyword_cooccurrence_comparison.pdf/.png")
    return au_cooc, g_cooc


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def temporal_lines_by_outlet(au_df: pd.DataFrame, g_df: pd.DataFrame) -> None:
    """
    Year-by-year thematic trends (3-year rolling mean) showing individual AU
    outlet lines instead of an aggregated broadsheet average, alongside The Guardian.
    Outputs: figures/comparison/temporal_lines_by_outlet.pdf/.png
    """
    FOCUS_GROUPS = [
        "Domestic Politics & Policy",
        "Physical & Ecological Impacts",
        "Climate Science",
        "Energy & Industry",
    ]

    MIN_YEAR, MAX_YEAR, WINDOW = 2001, 2025, 3
    years = np.arange(MIN_YEAR, MAX_YEAR + 1)

    OUTLETS = ["The Australian", "Sydney Morning Herald", "The Age", "Canberra Times"]
    OUTLET_LABELS = {
        "The Australian":        "The Australian",
        "Sydney Morning Herald": "SMH",
        "The Age":               "The Age",
        "Canberra Times":        "Canberra Times",
    }

    PANEL_EVENTS = {
        "Domestic Politics & Policy": [
            (2007, "Rudd\nelected",     "top"),
            (2011, "Carbon\nprice",     "bottom"),
            (2013, "Abbott\nelected",   "top"),
            (2014, "Price\nrepealed",   "bottom"),
            (2022, "Albanese\nelected", "bottom"),
        ],
        "Physical & Ecological Impacts": [
            (2009, "Black\nSaturday",   "top"),
            (2016, "GBR mass\nbleach",  "top"),
            (2019, "Black\nSummer",     "bottom"),
            (2022, "Eastern\nfloods",   "top"),
        ],
        "Climate Science": [
            (2009, "Climate-\ngate",    "top"),
            (2013, "IPCC\nAR5",         "top"),
            (2018, "IPCC\n1.5°C SR",    "top"),
            (2021, "IPCC\nAR6",         "bottom"),
        ],
        "Energy & Industry": [
            (2015, "Turnbull\nPM/COP21", "top"),
            (2016, "SA\nblackout",       "bottom"),
            (2017, "Hazelwood\nclosed",  "top"),
            (2022, "Gas\ncrisis",        "bottom"),
        ],
    }

    def _yearly_share(df, min_yr=MIN_YEAR, max_yr=MAX_YEAR):
        sub = df[(df["year"] >= min_yr) & (df["year"] <= max_yr)].copy()
        ct = sub.groupby(["year", "harmonized"]).size().unstack(fill_value=0)
        share = ct.div(ct.sum(axis=1), axis=0) * 100
        return share.reindex(range(min_yr, max_yr + 1), fill_value=0)

    def _yearly_share_outlet(df, pub, min_yr=MIN_YEAR, max_yr=MAX_YEAR):
        sub = df[(df["publication"] == pub) & (df["year"] >= min_yr) & (df["year"] <= max_yr)].copy()
        if len(sub) == 0:
            return pd.DataFrame(0.0, index=range(min_yr, max_yr + 1), columns=FOCUS_GROUPS)
        ct = sub.groupby(["year", "harmonized"]).size().unstack(fill_value=0)
        share = ct.div(ct.sum(axis=1), axis=0) * 100
        return share.reindex(range(min_yr, max_yr + 1), fill_value=0)

    g_yr = _yearly_share(g_df)
    au_yr = _yearly_share(au_df)
    outlet_yrs = {pub: _yearly_share_outlet(au_df, pub) for pub in OUTLETS}

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharey=False)
    axes = axes.flatten()

    for ax, grp, panel in zip(axes, FOCUS_GROUPS, "ABCD"):
        all_vals = []

        for pub in OUTLETS:
            yr_df = outlet_yrs[pub]
            vals = (yr_df[grp].rolling(WINDOW, center=True, min_periods=1).mean()
                    if grp in yr_df.columns
                    else pd.Series(0.0, index=range(MIN_YEAR, MAX_YEAR + 1)))
            ax.plot(years, vals.values,
                    color=OUTLET_COLORS[pub], linewidth=1.4, alpha=0.85,
                    label=OUTLET_LABELS[pub])
            all_vals.append(vals.values)

        au_mean_vals = (au_yr[grp].rolling(WINDOW, center=True, min_periods=1).mean()
                        if grp in au_yr.columns
                        else pd.Series(0.0, index=range(MIN_YEAR, MAX_YEAR + 1)))
        ax.plot(years, au_mean_vals.values,
                color="#888888", linewidth=2.0, linestyle=":",
                label="AU mean")
        all_vals.append(au_mean_vals.values)

        g_vals = (g_yr[grp].rolling(WINDOW, center=True, min_periods=1).mean()
                  if grp in g_yr.columns
                  else pd.Series(0.0, index=range(MIN_YEAR, MAX_YEAR + 1)))
        g_vals = g_vals.copy()
        g_vals[g_vals.index < 2005] = float("nan")
        ax.plot(years, g_vals.values,
                color=OUTLET_COLORS["The Guardian"], linewidth=2.2,
                linestyle="--", label="The Guardian")
        all_vals.append(g_vals.values)

        ax.set_title(f"({panel})  {grp}", fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Year", fontsize=8.5)
        ax.set_ylabel("Share of annual output (%)", fontsize=8.5)
        ax.set_xlim(MIN_YEAR, MAX_YEAR)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x)}"))
        ax.spines[["top", "right"]].set_visible(False)

        ymax = max(np.nanmax(v) for v in all_vals)
        if ymax == 0:
            ymax = 1
        ax.set_ylim(bottom=0, top=ymax * 1.45)

        for yr, lbl, pos in PANEL_EVENTS.get(grp, []):
            ax.axvline(yr, color="#777777", linewidth=0.8, linestyle=":")
            y_top = ax.get_ylim()[1]
            if pos == "top":
                ax.text(yr + 0.15, y_top * 0.96, lbl,
                        fontsize=5.8, color="#444444", va="top", ha="left", linespacing=1.3)
            else:
                ax.text(yr + 0.15, ymax * 0.06, lbl,
                        fontsize=5.8, color="#444444", va="bottom", ha="left", linespacing=1.3)

    handles, labels = axes[0].get_legend_handles_labels()
    desired_order = [OUTLET_LABELS[p] for p in OUTLETS] + ["AU mean", "The Guardian"]
    order = [labels.index(l) for l in desired_order if l in labels]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=6,
               fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)

    for ext in (".pdf", ".png"):
        stem = FIGURES_DIR / "temporal_lines_by_outlet"
        fig.savefig(stem.with_suffix(ext), bbox_inches="tight", dpi=150)
    plt.close()
    print("Figure saved → figures/comparison/temporal_lines_by_outlet.pdf/.png")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 11,
        "legend.fontsize": 8.5,
    })

    print("Loading corpora…")
    au_df = load_au_assignments()
    g_df  = load_g_assignments()
    print(f"  Australian: {len(au_df):,} articles (excl. outliers & noise)")
    print(f"  Guardian:   {len(g_df):,} articles (excl. outliers)")

    # national_comparison(au_df, g_df)
    # cross_corpus_cooccurrence(au_df, g_df)
    temporal_lines_by_outlet(au_df, g_df)
