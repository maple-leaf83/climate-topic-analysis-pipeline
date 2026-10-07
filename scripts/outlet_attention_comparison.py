"""
outlet_attention_comparison.py
──────────────────────────────
Cross-corpus outlet attention analysis.  Produces three figures:

  A.  AU broadsheets only (4 outlets, AU-specific group taxonomy)
      Heatmap + dot plot — representation ratio and binomial z-score.

  B1. All 6 outlets side by side (harmonized taxonomy):
      The Australian, The Age, SMH, Canberra Times,
      Guardian AU edition, Guardian UK edition.

  B2. Same as B1 but Guardian treated as a single combined outlet.

Method (for every outlet × group cell):
  p_obs = observed proportion of outlet's articles in that group
  p_exp = outlet's overall share of the relevant corpus
  SE    = sqrt(p_exp * (1 - p_exp) / n_group)
  z     = (p_obs - p_exp) / SE          (binomial effect size)
  ratio = p_obs / p_exp                 (representation ratio)

Outputs → figures/comparison/
  outlet_attention_au.pdf/.png
  outlet_attention_all6.pdf/.png
  outlet_attention_guardian_combined.pdf/.png

Usage:
    python outlet_attention_comparison.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency
from pathlib import Path

from config import DATA_DIR, FIGURES_DIR

# ── Paths ──────────────────────────────────────────────────────────────────────
AU_ASSIGN = DATA_DIR / "australian-no-letters" / "topic_assignments_aus.csv"
G_ASSIGN  = DATA_DIR / "guardian"              / "topic_assignments_guardian.csv"
OUT_DIR   = DATA_DIR / "australian-no-letters"
FIG_DIR   = FIGURES_DIR / "comparison"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Font ───────────────────────────────────────────────────────────────────────
_available = {f.name for f in fm.fontManager.ttflist}
FONT = "Georgia" if "Georgia" in _available else "DejaVu Serif"
plt.rcParams.update({
    "font.family":     FONT,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ── AU-specific group order (Figure A) ────────────────────────────────────────
AU_GROUP_ORDER = [
    "Australian Politics",
    "Carbon Pricing & Domestic Climate Policy",
    "Regional & International Politics",
    "Climate Science",
    "Physical & Ecological Impacts",
    "Energy Transition & Technology",
    "International Climate Policy",
    "Culture Media & Society",
]

AU_OUTLETS = ["The Australian", "The Age", "Sydney Morning Herald", "Canberra Times"]
from config import OUTLET_COLORS as _OC
AU_OUTLET_COLORS = {k: _OC[k] for k in ["The Australian","The Age","Sydney Morning Herald","Canberra Times"]}

# ── Harmonized group order (Figures B1, B2) ────────────────────────────────────
HARMONIZED_ORDER = [
    "Domestic Politics & Policy",
    "International Politics",
    "International Climate Policy",
    "Climate Science",
    "Physical & Ecological Impacts",
    "Energy & Industry",
    "Culture, Media & Activism",
]

AU_TO_HARMONIZED = {
    "Australian Politics":                      "Domestic Politics & Policy",
    "Carbon Pricing & Domestic Climate Policy": "Domestic Politics & Policy",
    "Regional & International Politics":        "International Politics",
    "International Climate Policy":             "International Climate Policy",
    "Climate Science":                          "Climate Science",
    "Physical & Ecological Impacts":            "Physical & Ecological Impacts",
    "Energy Transition & Technology":           "Energy & Industry",
    "Culture Media & Society":                  "Culture, Media & Activism",
}

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

ALL6_OUTLETS = [
    "The Australian", "The Age", "Sydney Morning Herald", "Canberra Times",
    "The Guardian (AU)", "The Guardian (UK)",
]
ALL6_COLORS = {k: _OC[k] for k in ["The Australian","The Age","Sydney Morning Herald","Canberra Times","The Guardian (AU)","The Guardian (UK)"]}

COMBINED_OUTLETS = [
    "The Australian", "The Age", "Sydney Morning Herald", "Canberra Times",
    "The Guardian",
]
COMBINED_COLORS = {k: _OC[k] for k in COMBINED_OUTLETS}

# ── Guardian-specific group order (pub share matrix) ──────────────────────────
G_GROUP_ORDER = [
    "Australian Politics",
    "UK & European Politics",
    "North American & International Politics",
    "International Climate Policy",
    "Climate Science & Scepticism",
    "Physical & Ecological Impacts",
    "Energy Transition & Technology",
    "Activism Culture & Society",
]

G_OUTLETS = ["The Guardian (UK)", "The Guardian (AU)"]
G_OUTLET_COLORS = {k: _OC[k] for k in ["The Guardian (UK)","The Guardian (AU)"]}


# ══════════════════════════════════════════════════════════════════════════════
# Core statistics
# ══════════════════════════════════════════════════════════════════════════════

def compute_attention(df: pd.DataFrame, group_col: str,
                      group_order: list[str], outlets: list[str],
                      pub_col: str = "publication"):
    """
    Returns (ratio, z, ct) DataFrames for the given group × outlet contingency.
    Rows = groups, cols = outlets.
    """
    df = df[df[group_col].isin(group_order) & df[pub_col].isin(outlets)].copy()
    total = len(df)
    p_exp = (df[pub_col].value_counts() / total).reindex(outlets)

    ct = pd.crosstab(df[group_col], df[pub_col]).reindex(
        index=group_order, columns=outlets, fill_value=0
    )
    n_i = ct.sum(axis=1)

    p_obs = ct.div(n_i, axis=0)
    ratio = p_obs.div(p_exp)

    z = pd.DataFrame(index=ct.index, columns=ct.columns, dtype=float)
    for outlet in outlets:
        pe = p_exp[outlet]
        po = ct[outlet] / n_i
        se = np.sqrt(pe * (1.0 - pe) / n_i)
        z[outlet] = (po - pe) / se

    return ratio.round(4), z.round(3), ct


def print_chi2(ct: pd.DataFrame, label: str):
    chi2, p, dof, expected = chi2_contingency(ct.values)
    exp_min = expected.min()
    print(f"\n[{label}] χ²({dof}) = {chi2:.2f}, p = {p:.2e}, "
          f"min expected = {exp_min:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def draw_heatmap_pair(ratio: pd.DataFrame, z: pd.DataFrame, title: str, out_stem: Path):
    """Side-by-side ratio + z-score heatmap."""
    ratio_vmax = max(abs(ratio.values - 1).max() + 1, 2.5)
    ratio_vmin = max(0.0, 2.0 - ratio_vmax)
    z_vlim = float(np.percentile(np.abs(z.values[np.isfinite(z.values)]), 95))

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(ratio.columns) * 1.4), 5.5),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.06})

    def _draw(ax, data, t, vmin, vcenter, vmax, fmt, cbar_label):
        divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sns.heatmap(data, ax=ax, norm=divnorm, cmap="RdBu_r",
                        annot=True, fmt=fmt, annot_kws={"size": 7.5},
                        linewidths=0.4, linecolor="#dddddd",
                        cbar_kws={"label": cbar_label, "shrink": 0.85, "pad": 0.02})
        ax.set_title(t, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8.5)

    _draw(axes[0], ratio, "Representation ratio  (observed / expected)",
          ratio_vmin, 1.0, ratio_vmax, ".2f", "Ratio")
    _draw(axes[1], z.round(1), "Binomial effect size  (z-score)",
          -z_vlim, 0.0, z_vlim, ".1f", "z")
    axes[1].set_yticklabels([])

    fig.text(0.5, -0.03,
             "Red = over-represented relative to corpus share;  Blue = under-represented.",
             ha="center", fontsize=8, color="#555555", style="italic")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()

    for ext in (".pdf", ".png"):
        fig.savefig(str(out_stem.with_suffix(ext)), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → {out_stem.name}.pdf/.png")


def draw_dotplot(ratio: pd.DataFrame, z: pd.DataFrame,
                 outlet_colors: dict, title: str, out_stem: Path):
    """Faceted dot plot: one panel per outlet."""
    R_HIGH, R_LOW = 1.25, 0.75
    groups = list(ratio.index)
    outlets = list(ratio.columns)
    y_pos = {g: i for i, g in enumerate(groups)}

    def zsqrt(v):
        return float(np.sign(v) * np.sqrt(abs(v)))

    fig, axes = plt.subplots(1, len(outlets),
                             figsize=(3.0 * len(outlets) + 1.5, 6),
                             sharey=True,
                             gridspec_kw={"wspace": 0.08})
    if len(outlets) == 1:
        axes = [axes]

    SQRT_TICKS  = [-9, -6, -3, 0, 3, 6, 9]
    TICK_LABELS = ["-81", "-36", "-9", "0", "9", "36", "81"]
    X_LIM = (-10.5, 10.5)

    for ax, outlet in zip(axes, outlets):
        color = outlet_colors.get(outlet, "#555555")
        for group in groups:
            y     = y_pos[group]
            z_val = float(z.loc[group, outlet])
            x_val = zsqrt(z_val)
            r_val = float(ratio.loc[group, outlet])
            if r_val > R_HIGH:
                ax.scatter(x_val, y, marker=">", s=90, color=color, zorder=3, linewidths=0)
            elif r_val < R_LOW:
                ax.scatter(x_val, y, marker="<", s=90, color=color, zorder=3, linewidths=0)
            else:
                ax.scatter(x_val, y, marker="s", s=60, facecolors="none",
                           edgecolors=color, linewidths=1.2, zorder=3)

        ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
        for yg in range(len(groups)):
            ax.axhline(yg, color="#e0e0e0", linewidth=0.5, zorder=0)
        ax.set_xlim(X_LIM)
        ax.set_ylim(-0.8, len(groups) - 0.2)
        ax.set_xticks(SQRT_TICKS)
        ax.set_xticklabels(TICK_LABELS, fontsize=7.5)
        short = outlet.replace("Sydney Morning Herald", "SMH").replace(" ", "\n")
        ax.set_title(short, fontsize=9, fontweight="bold", color=color, pad=8)
        ax.tick_params(axis="y", length=0)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#cccccc")

    axes[0].set_yticks(range(len(groups)))
    axes[0].set_yticklabels(groups, fontsize=8.5)

    legend_elements = [
        Line2D([0],[0], marker=">", color="w", markerfacecolor="#555",
               markersize=9, label=r"Over-represented  ($r > 1.25$)"),
        Line2D([0],[0], marker="<", color="w", markerfacecolor="#555",
               markersize=9, label=r"Under-represented  ($r < 0.75$)"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="none",
               markeredgecolor="#555", markeredgewidth=1.2,
               markersize=9, label=r"Within expected range"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.tight_layout()

    for ext in (".pdf"):
        fig.savefig(str(out_stem.with_suffix(ext)), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → {out_stem.name}.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Publication share matrix (column-normalised)
# ══════════════════════════════════════════════════════════════════════════════

def pub_share_matrix(df: pd.DataFrame, group_col: str, group_order: list[str],
                     outlets: list[str], outlet_colors: dict,
                     title: str, out_stem: Path, csv_path: Path | None = None):
    """
    Column-normalised heatmap: rows = groups, cols = outlets.
    Each cell = % of that outlet's articles in the group.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    df = df[df[group_col].isin(group_order) & df["publication"].isin(outlets)].copy()
    pub_totals = df["publication"].value_counts().reindex(outlets, fill_value=0)

    rows = []
    for grp in group_order:
        sub = df[df[group_col] == grp]
        counts = sub["publication"].value_counts().reindex(outlets, fill_value=0)
        pct = counts / pub_totals.replace(0, np.nan) * 100
        rows.append({"group": grp, **pct.to_dict()})

    matrix = pd.DataFrame(rows).set_index("group")[outlets]

    fig, ax = plt.subplots(figsize=(max(6, len(outlets) * 2.0), max(5, len(group_order) * 0.7)))
    vmax_m = np.nanmax(matrix.values)
    im = ax.imshow(matrix.values, cmap="Blues", aspect="auto", vmin=0, vmax=vmax_m)

    for i, grp in enumerate(group_order):
        for j, out in enumerate(outlets):
            val = matrix.values[i, j]
            if np.isnan(val) or val < 0.5:
                continue
            text_color = "white" if val > vmax_m * 0.65 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=11, color=text_color)

    ax.set_xticks(range(len(outlets)))
    ax.set_xticklabels(outlets, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(group_order, fontsize=9)
    ax.set_title(title, fontsize=11, pad=10)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.2)
    fig.colorbar(im, cax=cax, label="% of outlet's articles")
    fig.tight_layout()

    for ext in (".pdf"):
        fig.savefig(str(out_stem.with_suffix(ext)), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → {out_stem.name}.pdf")

    if csv_path:
        matrix.round(1).to_csv(csv_path)

    print(matrix.round(1).to_string())
    return matrix


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data…")
    au = pd.read_csv(AU_ASSIGN, low_memory=False)
    au = au[au["topic_id"] != -1].copy()

    g = pd.read_csv(G_ASSIGN, low_memory=False)
    g = g[g["topic_id"] != -1].copy()
    # Remap Guardian publication to edition labels
    g["publication"] = g["publication"].str.strip()

    # ── Build harmonized combined dataframe ───────────────────────────────────
    au_h = au.copy()
    au_h["harmonized"] = au_h["group"].map(AU_TO_HARMONIZED)

    g_h = g.copy()
    g_h["harmonized"] = g_h["group"].map(G_TO_HARMONIZED)

    # For B2: collapse Guardian editions into one outlet
    g_combined = g_h.copy()
    g_combined["publication"] = "The Guardian"

    # Stack AU + Guardian for B1 and B2
    combined_b1 = pd.concat([
        au_h[["harmonized", "publication"]].rename(columns={"harmonized": "group"}),
        g_h[["harmonized", "publication"]].rename(columns={"harmonized": "group"}),
    ], ignore_index=True)

    combined_b2 = pd.concat([
        au_h[["harmonized", "publication"]].rename(columns={"harmonized": "group"}),
        g_combined[["harmonized", "publication"]].rename(columns={"harmonized": "group"}),
    ], ignore_index=True)

    # ── Figure A: AU broadsheets only ─────────────────────────────────────────
    print("\n── Figure A: AU broadsheets (corpus-specific groups) ──")
    au_pubs = au[au["publication"].isin(AU_OUTLETS)]
    ratio_a, z_a, ct_a = compute_attention(
        au, group_col="group", group_order=AU_GROUP_ORDER, outlets=AU_OUTLETS)
    print_chi2(ct_a, "AU broadsheets")
    draw_heatmap_pair(ratio_a, z_a,
                      "Outlet attention by topic group — Australian broadsheets",
                      FIG_DIR / "outlet_attention_au_heatmap")
    draw_dotplot(ratio_a, z_a, AU_OUTLET_COLORS,
                 "Outlet attention by topic group — Australian broadsheets",
                 FIG_DIR / "outlet_attention_au_dotplot")

    # Save AU CSVs
    ct_a.to_csv(OUT_DIR / "outlet_observed_counts.csv")
    ratio_a.to_csv(OUT_DIR / "outlet_representation_ratios.csv")
    z_a.to_csv(OUT_DIR / "outlet_binomial_zscores.csv")

    # ── Figure B1: All 6 outlets, harmonized groups ───────────────────────────
    print("\n── Figure B1: All 6 outlets (harmonized groups) ──")
    ratio_b1, z_b1, ct_b1 = compute_attention(
        combined_b1, group_col="group",
        group_order=HARMONIZED_ORDER, outlets=ALL6_OUTLETS)
    print_chi2(ct_b1, "All 6 outlets")
    draw_heatmap_pair(ratio_b1, z_b1,
                      "Outlet attention by topic group — all outlets (harmonized taxonomy)",
                      FIG_DIR / "outlet_attention_all6_heatmap")
    draw_dotplot(ratio_b1, z_b1, ALL6_COLORS,
                 "Outlet attention by topic group — all outlets (harmonized taxonomy)",
                 FIG_DIR / "outlet_attention_all6_dotplot")
    ct_b1.to_csv(FIG_DIR / "outlet_all6_observed_counts.csv")
    ratio_b1.to_csv(FIG_DIR / "outlet_all6_representation_ratios.csv")
    z_b1.to_csv(FIG_DIR / "outlet_all6_binomial_zscores.csv")

    # ── Figure B2: Guardian as single outlet ──────────────────────────────────
    print("\n── Figure B2: Guardian combined (harmonized groups) ──")
    ratio_b2, z_b2, ct_b2 = compute_attention(
        combined_b2, group_col="group",
        group_order=HARMONIZED_ORDER, outlets=COMBINED_OUTLETS)
    print_chi2(ct_b2, "Guardian combined")
    draw_heatmap_pair(ratio_b2, z_b2,
                      "Outlet attention by topic group — AU broadsheets vs Guardian",
                      FIG_DIR / "outlet_attention_guardian_combined_heatmap")
    draw_dotplot(ratio_b2, z_b2, COMBINED_COLORS,
                 "Outlet attention by topic group — AU broadsheets vs Guardian",
                 FIG_DIR / "outlet_attention_guardian_combined_dotplot")
    ct_b2.to_csv(FIG_DIR / "outlet_combined_observed_counts.csv")
    ratio_b2.to_csv(FIG_DIR / "outlet_combined_representation_ratios.csv")
    z_b2.to_csv(FIG_DIR / "outlet_combined_binomial_zscores.csv")

    # ── Pub share matrix: AU broadsheets + combined Guardian (harmonized) ────────
    print("\n── Pub share matrix: all outlets, harmonized groups (column-normalised) ──")
    pub_share_matrix(
        combined_b2, group_col="group", group_order=HARMONIZED_ORDER,
        outlets=COMBINED_OUTLETS, outlet_colors=COMBINED_COLORS,
        title="% of each outlet's climate articles per thematic group",
        out_stem=FIG_DIR / "pub_share_matrix",
        csv_path=FIG_DIR / "pub_share_matrix.csv",
    )

    print("\nDone.")
