"""
temporal_comparison.py
──────────────────────
Temporal analysis figures for the AU vs Guardian comparison.

Figure 1: Stacked bar chart — harmonized group share per era, two panels (AU | Guardian)
Figure 2: Year-by-year line chart — selected harmonized groups, both corpora
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

from config import DATA_DIR, FIGURES_DIR, HARMONIZED_COLORS, OUTLET_COLORS

FIG_DIR = FIGURES_DIR / "comparison"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Fonts ──────────────────────────────────────────────────────────────────────
_available = {f.name for f in fm.fontManager.ttflist}
FONT = "Georgia" if "Georgia" in _available else "DejaVu Serif"
plt.rcParams.update({
    "font.family": FONT,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ── Mappings ───────────────────────────────────────────────────────────────────
AU_TO_H = {
    "Australian Politics":                      "Domestic Politics & Policy",
    "Carbon Pricing & Domestic Climate Policy": "Domestic Politics & Policy",
    "Regional & International Politics":        "International Politics",
    "International Climate Policy":             "International Climate Policy",
    "Climate Science":                          "Climate Science",
    "Physical & Ecological Impacts":            "Physical & Ecological Impacts",
    "Energy Transition & Technology":           "Energy & Industry",
    "Culture Media & Society":                  "Culture, Media & Activism",
}
G_TO_H = {
    "Australian Politics":                        "Domestic Politics & Policy",
    "UK & European Politics":                     "Domestic Politics & Policy",
    "North American & International Politics":    "International Politics",
    "International Climate Policy":               "International Climate Policy",
    "Climate Science & Scepticism":               "Climate Science",
    "Physical & Ecological Impacts":              "Physical & Ecological Impacts",
    "Energy Transition & Technology":             "Energy & Industry",
    "Activism Culture & Society":                 "Culture, Media & Activism",
}

HARMONIZED = [
    "Domestic Politics & Policy",
    "Energy & Industry",
    "Culture, Media & Activism",
    "Climate Science",
    "International Climate Policy",
    "International Politics",
    "Physical & Ecological Impacts",
]

GROUP_COLORS = HARMONIZED_COLORS

ERAS = ["Howard", "Rudd/Gillard", "Abbott", "Turnbull/Morrison", "Albanese"]
ERA_LABELS = ["Howard\n1996–2007", "Rudd/Gillard\n2007–2013",
              "Abbott\n2013–2015", "Turnbull/Morrison\n2015–2022",
              "Albanese\n2022–"]

# ── Load data ──────────────────────────────────────────────────────────────────
au = pd.read_csv(DATA_DIR / "australian-no-letters" / "topic_assignments_aus.csv",
                 low_memory=False)
g  = pd.read_csv(DATA_DIR / "guardian" / "topic_assignments_guardian.csv",
                 low_memory=False)

au["harmonized"] = au["group"].map(AU_TO_H)
g["harmonized"]  = g["group"].map(G_TO_H)

au_era = au[au["era"].isin(ERAS)].copy()
g_era  = g[g["era"].isin(ERAS)].copy()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Stacked bar: harmonized group share per era, AU | Guardian
# ══════════════════════════════════════════════════════════════════════════════
def era_share(df):
    ct = df.groupby(["era","harmonized"]).size().unstack(fill_value=0)
    ct = ct.reindex(ERAS)
    return ct.div(ct.sum(axis=1), axis=0) * 100

au_share = era_share(au_era)
g_share  = era_share(g_era)

fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

for ax, share, title, n_per_era in [
    (axes1[0], au_share, "Australian broadsheets",
     au_era.groupby("era").size().reindex(ERAS)),
    (axes1[1], g_share,  "The Guardian",
     g_era.groupby("era").size().reindex(ERAS)),
]:
    x = np.arange(len(ERAS))
    bottoms = np.zeros(len(ERAS))
    for grp in HARMONIZED:
        vals = share[grp].values if grp in share.columns else np.zeros(len(ERAS))
        ax.bar(x, vals, bottom=bottoms, color=GROUP_COLORS[grp],
               width=0.65, label=grp)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(ERA_LABELS, fontsize=8.5)
    ax.set_ylabel("Share of corpus (%)", fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=22)
    ax.spines[["top", "right"]].set_visible(False)

    # Article-count annotations above each bar
    for i, era in enumerate(ERAS):
        n = int(n_per_era[era]) if era in n_per_era.index else 0
        ax.text(i, 101, f"n={n:,}", ha="center", va="bottom",
                fontsize=7, color="#555555")

# Shared legend below
handles = [plt.Rectangle((0,0),1,1, color=GROUP_COLORS[g]) for g in HARMONIZED]
fig1.legend(handles, HARMONIZED, loc="lower center", ncol=4,
            frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.07))

# fig1.suptitle("Thematic composition by government era", fontsize=13,
#               fontweight="bold", y=1.01)
fig1.tight_layout()
fig1.savefig(str(FIG_DIR / "temporal_era_stacked.pdf"), bbox_inches="tight", dpi=300)
# fig1.savefig(str(FIG_DIR / "temporal_era_stacked.png"), bbox_inches="tight", dpi=150)
plt.close()
print("Figure 1 → temporal_era_stacked.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Year-by-year line chart, selected harmonized groups
# Smoothed 3-year rolling mean; both corpora on same axes, one panel per group
# ══════════════════════════════════════════════════════════════════════════════
FOCUS_GROUPS = [
    "Domestic Politics & Policy",
    "Physical & Ecological Impacts",
    "Climate Science",
    "Energy & Industry",
]

# Restrict to years with enough data in both corpora
MIN_YEAR = 2005
MAX_YEAR = 2025

def yearly_share(df, min_yr=MIN_YEAR, max_yr=MAX_YEAR):
    sub = df[(df["year"] >= min_yr) & (df["year"] <= max_yr)].copy()
    ct = sub.groupby(["year","harmonized"]).size().unstack(fill_value=0)
    yr_totals = ct.sum(axis=1)
    share = ct.div(yr_totals, axis=0) * 100
    return share.reindex(range(min_yr, max_yr+1), fill_value=0)

au_yr = yearly_share(au)
g_yr  = yearly_share(g)

WINDOW = 3
years = np.arange(MIN_YEAR, MAX_YEAR + 1)

fig2, axes2 = plt.subplots(2, 2, figsize=(13, 7), sharey=False)
axes2 = axes2.flatten()

# ── Theme-specific event annotations ──────────────────────────────────────────
# Each entry: (year, label, position)  where position is "top" or "bottom"
PANEL_EVENTS = {
    "Domestic Politics & Policy": [
        (2007, "Rudd\nelected",    "top"),
        (2011, "Carbon\nprice",    "bottom"),
        (2013, "Abbott\nelected",  "top"),
        (2014, "Price\nrepealed",  "bottom"),
        (2019, "Morrison\nelected","top"),
        (2022, "Albanese\nelected","bottom"),
    ],
    "Physical & Ecological Impacts": [
        (2009, "Black\nSaturday",  "top"),
        (2011, "QLD\nfloods",      "top"),
        (2016, "GBR mass\nbleach", "top"),
        (2019, "Black\nSummer",    "bottom"),
        (2022, "Eastern\nfloods",  "top"),
    ],
    "Climate Science": [
        (2009, "Climate-\ngate",    "top"),
        (2009, "COP15\nCopenhagen", "bottom"),
        (2013, "IPCC\nAR5",         "top"),
        (2015, "COP21\nParis",      "bottom"),
        (2018, "IPCC\n1.5°C SR",    "top"),
        (2021, "IPCC\nAR6",         "bottom"),
        (2021, "COP26\nGlasgow",    "top"),
    ],
    "Energy & Industry": [
        (2015, "Turnbull\nPM/COP21", "top"),
        (2016, "SA\nblackout",       "bottom"),
        (2017, "Hazelwood\nclosed",  "top"),
        (2022, "Gas\ncrisis",        "bottom"),
    ],
}

for ax, grp, panel in zip(axes2, FOCUS_GROUPS, "ABCD"):
    au_vals = au_yr[grp].rolling(WINDOW, center=True, min_periods=1).mean() if grp in au_yr.columns else pd.Series(0, index=au_yr.index)
    g_vals  = g_yr[grp].rolling(WINDOW, center=True, min_periods=1).mean()  if grp in g_yr.columns  else pd.Series(0, index=g_yr.index)

    ax.plot(years, au_vals.values, color=OUTLET_COLORS["The Australian"], linewidth=2,
            label="AU broadsheets")
    ax.plot(years, g_vals.values,  color=OUTLET_COLORS["The Guardian"], linewidth=2,
            linestyle="--", label="The Guardian")

    ax.set_title(f"({panel})  {grp}", fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Year", fontsize=8.5)
    ax.set_ylabel("Share of annual output (%)", fontsize=8.5)
    ax.set_xlim(MIN_YEAR, MAX_YEAR)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.spines[["top","right"]].set_visible(False)

    # Theme-specific event lines
    ymax = max(au_vals.max(), g_vals.max()) * 1.0
    ax.set_ylim(bottom=0, top=ymax * 1.35)   # headroom for top labels

    for yr, lbl, pos in PANEL_EVENTS.get(grp, []):
        ax.axvline(yr, color="#777777", linewidth=0.8, linestyle=":")
        if pos == "top":
            ax.text(yr + 0.15, ymax * 1.28, lbl,
                    fontsize=6.0, color="#444444", va="top", ha="left",
                    linespacing=1.3)
        else:
            ax.text(yr + 0.15, ymax * 0.08, lbl,
                    fontsize=6.0, color="#444444", va="bottom", ha="left",
                    linespacing=1.3)

axes2[0].legend(fontsize=8.5, frameon=False)

# fig2.suptitle(f"Year-by-year thematic trends ({WINDOW}-year rolling mean)",
#               fontsize=13, fontweight="bold", y=1.01)
fig2.tight_layout()
fig2.savefig(str(FIG_DIR / "temporal_yearly_lines.pdf"), bbox_inches="tight", dpi=300)
fig2.savefig(str(FIG_DIR / "temporal_yearly_lines.png"), bbox_inches="tight", dpi=150)
plt.close()
print("Figure 2 → temporal_yearly_lines.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Print era-level summary tables for the paper
# ══════════════════════════════════════════════════════════════════════════════
for label, df in [("AU", au_era), ("Guardian", g_era)]:
    sub = df.groupby(["era","harmonized"]).size().unstack(fill_value=0).reindex(ERAS)
    share = (sub.div(sub.sum(axis=1), axis=0) * 100).round(1)
    print(f"\n{label} era shares (harmonized, %):")
    print(share[[h for h in HARMONIZED if h in share.columns]].to_string())

