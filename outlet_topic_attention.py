"""
outlet_topic_attention.py
─────────────────────────
Computes whether each outlet devotes significantly more or less attention
to each topic group than its overall corpus share would predict.

Method:
  For every (topic group i, outlet j) cell:
    p_obs_ij  = observed proportion of outlet j's articles within group i
    p_exp_j   = outlet j's overall share of the corpus  (expected baseline)
    SE_ij     = sqrt(p_exp_j * (1 - p_exp_j) / n_i)    (binomial SE)
    z_ij      = (p_obs_ij - p_exp_j) / SE_ij            (effect size)
    ratio_ij  = p_obs_ij / p_exp_j                      (representation ratio)

  z > 0  → outlet over-represented in that group relative to corpus share
  z < 0  → outlet under-represented

Significance threshold: |z| > 2  (≈ p < 0.05, binomial test)

Usage:
    python outlet_topic_attention.py                  # 5 outlets incl. letters
    python outlet_topic_attention.py --exclude-letters

Outputs (written relative to repo root via config.py):
  data/combined/outlet_observed_counts.csv
  data/combined/outlet_representation_ratios.csv
  data/combined/outlet_binomial_zscores.csv
  figures/cluster_analysis/outlet_topic_attention_heatmap.pdf
  figures/cluster_analysis/outlet_topic_attention_dotplot.pdf
"""

import argparse
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.stats import chi2_contingency

from config import DATA_DIR, FIGURES_DIR

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--exclude-letters", action="store_true",
                    help="Exclude Letters to Editor from the analysis")
args = parser.parse_args()

INCLUDE_LETTERS = not args.exclude_letters

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = DATA_DIR / "combined-no-letters" / "topic_assignments.csv"
LETTERS_PATH = DATA_DIR / "letters"             / "topic_assignments.csv"
SUMMARY_PATH = DATA_DIR / "combined-no-letters" / "topic_summary.csv"
FIG_DIR      = FIGURES_DIR / "cluster_analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Outlets ───────────────────────────────────────────────────────────────────
EDITORIAL_OUTLETS = [
    "The Guardian",
    "The Age",
    "The Australian",
    "Sydney Morning Herald",
    "Canberra Times",
]
LETTERS_LABEL = "Letters to Editor"

OUTLETS = EDITORIAL_OUTLETS + ([LETTERS_LABEL] if INCLUDE_LETTERS else [])

OUTLET_COLORS = {
    "The Guardian":           "#1a6e3c",
    "The Age":                "#1f4e79",
    "The Australian":         "#e08214",
    "Sydney Morning Herald":  "#7b2d8b",
    "Canberra Times":         "#c55a11",
    LETTERS_LABEL:            "#8B4513",   # brown
}

NOISE_GROUP = "Noise"

# ── Font ──────────────────────────────────────────────────────────────────────
_available = {f.name for f in fm.fontManager.ttflist}
FONT = "Georgia" if "Georgia" in _available else "DejaVu Serif"
plt.rcParams.update({
    "font.family":     FONT,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

# ── Load editorials ───────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df[df["publication"].isin(EDITORIAL_OUTLETS)].copy()

# ── Load & append letters ─────────────────────────────────────────────────────
if INCLUDE_LETTERS:
    letters = pd.read_csv(LETTERS_PATH)
    # Add group column from topic_summary if not already present
    if "topic_group" not in letters.columns:
        summary = pd.read_csv(SUMMARY_PATH)[["topic_id", "Group"]].rename(
            columns={"Group": "topic_group"}
        )
        letters = letters.merge(summary, on="topic_id", how="left")
    letters["publication"] = LETTERS_LABEL
    df = pd.concat([df, letters], ignore_index=True)
    print(f"Letters appended: {len(letters):,} articles → {LETTERS_LABEL}")

# ── Filter noise ──────────────────────────────────────────────────────────────
df = df[df["topic_group"] != NOISE_GROUP].copy()

total = len(df)
print(f"Articles (noise excluded): {total:,}")
print(f"Outlets:\n{df['publication'].value_counts().to_string()}")
print(f"\nGroups ({df['topic_group'].nunique()}): {sorted(df['topic_group'].unique())}\n")

# ── Corpus-level expected proportions ─────────────────────────────────────────
p_exp = df["publication"].value_counts() / total
print("Expected proportions (corpus share):")
print(p_exp.round(4).to_string())

# ── Contingency table ─────────────────────────────────────────────────────────
ct = pd.crosstab(df["topic_group"], df["publication"])[OUTLETS]
n_i = ct.sum(axis=1)

# ── Representation ratios ─────────────────────────────────────────────────────
p_obs = ct.div(n_i, axis=0)
ratio = p_obs.div(p_exp)

# ── Binomial z-scores ─────────────────────────────────────────────────────────
z = pd.DataFrame(index=ct.index, columns=ct.columns, dtype=float)
for outlet in OUTLETS:
    pe        = p_exp[outlet]
    po        = ct[outlet] / n_i
    se        = np.sqrt(pe * (1.0 - pe) / n_i)
    z[outlet] = (po - pe) / se

# ── Save CSVs ─────────────────────────────────────────────────────────────────
suffix = "" if INCLUDE_LETTERS else "_no_letters"
ct.to_csv(DATA_DIR / "combined-no-letters" / f"outlet_observed_counts{suffix}.csv")
ratio.round(4).to_csv(DATA_DIR / "combined-no-letters" / f"outlet_representation_ratios{suffix}.csv")
z.round(3).to_csv(DATA_DIR / "combined-no-letters" / f"outlet_binomial_zscores{suffix}.csv")
print(f"\nSaved CSVs  (suffix='{suffix}')")

# ── Omnibus chi-square test of independence ────────────────────────────────
chi2_stat, p_val, dof, expected = chi2_contingency(ct)
exp_min  = expected.min()
exp_low  = (expected < 5).sum().sum()
print(f"\n── Pearson chi-square (omnibus) ──")
print(f"  χ²({dof}) = {chi2_stat:.2f},  p = {p_val:.2e}")
print(f"  Min expected cell count: {exp_min:.2f}  |  Cells < 5: {exp_low}")
chi2_path = DATA_DIR / "combined-no-letters" / f"outlet_chi2{suffix}.txt"
chi2_path.write_text(
    f"chi2={chi2_stat:.4f}\ndf={dof}\np={p_val:.4e}\n"
    f"expected_min={exp_min:.4f}\ncells_below_5={int(exp_low)}\n"
)
print(f"  Saved → {chi2_path}")

print("\n── Representation Ratios ──")
print(ratio.round(2).to_string())
print("\n── Binomial z-scores ──")
print(z.round(1).to_string())

# ── Sort rows by Guardian z-score (descending) ───────────────────────────────
row_order  = z["The Guardian"].sort_values(ascending=False).index
z_plot     = z.loc[row_order]
ratio_plot = ratio.loc[row_order]

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Heatmap (ratio + z-score side by side)
# ═══════════════════════════════════════════════════════════════════════════════
ratio_vmax = max(abs(ratio_plot.values - 1).max() + 1, 2.5)
ratio_vmin = max(0.0, 2.0 - ratio_vmax)
z_vlim     = float(np.percentile(np.abs(z_plot.values), 95))

fig, axes = plt.subplots(
    1, 2,
    figsize=(14 if INCLUDE_LETTERS else 12, 5.5),
    gridspec_kw={"width_ratios": [1, 1], "wspace": 0.06},
)

def draw_heatmap(ax, data, title, vmin, vcenter, vmax, fmt, cbar_label):
    divnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.heatmap(
            data, ax=ax, norm=divnorm, cmap="RdBu_r",
            annot=True, fmt=fmt, annot_kws={"size": 8.0},
            linewidths=0.4, linecolor="#dddddd",
            cbar_kws={"label": cbar_label, "shrink": 0.85, "pad": 0.02},
        )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30, labelsize=8.0)
    ax.tick_params(axis="y", rotation=0,  labelsize=8.5)

draw_heatmap(axes[0], ratio_plot,
             title="Representation ratio  (observed / expected)",
             vmin=ratio_vmin, vcenter=1.0, vmax=ratio_vmax,
             fmt=".2f", cbar_label="Ratio")
draw_heatmap(axes[1], z_plot.round(1),
             title="Binomial effect size  (z-score)",
             vmin=-z_vlim, vcenter=0.0, vmax=z_vlim,
             fmt=".1f", cbar_label="z")
axes[1].set_yticklabels([])

fig.text(0.5, -0.03,
         "Rows sorted by Guardian z-score (descending).  "
         "Red = over-represented relative to corpus share;  Blue = under-represented.",
         ha="center", fontsize=8, color="#555555", style="italic")
fig.suptitle("Outlet attention by topic group", fontsize=13, fontweight="bold", y=1.01)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plt.tight_layout()

stem1 = FIG_DIR / f"outlet_topic_attention_heatmap{suffix}"
fig.savefig(str(stem1) + ".pdf", bbox_inches="tight", dpi=300)
plt.close()
print(f"\nHeatmap → {stem1}.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Dot plot (faceted by outlet, sqrt-transformed x-axis)
# ═══════════════════════════════════════════════════════════════════════════════
SIG_THRESHOLD = 2.0

groups   = list(row_order)
n_groups = len(groups)
y_pos    = {g: i for i, g in enumerate(groups)}

def zsqrt(val):
    return float(np.sign(val) * np.sqrt(abs(val)))

n_panels   = len(OUTLETS)
fig_width  = 3.0 * n_panels + 1.5
fig2, axes2 = plt.subplots(
    1, n_panels,
    figsize=(fig_width, 6),
    sharey=True,
    gridspec_kw={"wspace": 0.08},
)
if n_panels == 1:
    axes2 = [axes2]

SQRT_TICKS  = [-9, -6, -3, 0, 3, 6, 9]
TICK_LABELS = ["-81", "-36", "-9", "0", "9", "36", "81"]
X_LIM = (-10.5, 10.5)

for ax, outlet in zip(axes2, OUTLETS):
    color = OUTLET_COLORS[outlet]

    for group in groups:
        y     = y_pos[group]
        z_val = float(z.loc[group, outlet])
        x_val = zsqrt(z_val)

        if z_val > SIG_THRESHOLD:
            ax.scatter(x_val, y, marker=">", s=90, color=color,
                       zorder=3, linewidths=0)
        elif z_val < -SIG_THRESHOLD:
            ax.scatter(x_val, y, marker="<", s=90, color=color,
                       zorder=3, linewidths=0)
        else:
            ax.scatter(x_val, y, marker="s", s=60, facecolors="none",
                       edgecolors=color, linewidths=1.2, zorder=3)

    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    for yg in range(n_groups):
        ax.axhline(yg, color="#e0e0e0", linewidth=0.5, zorder=0)

    sig_x = zsqrt(SIG_THRESHOLD)
    ax.axvspan(-sig_x, sig_x, color="#f5f5f5", zorder=0)

    ax.set_xlim(X_LIM)
    ax.set_ylim(-0.8, n_groups - 0.2)
    ax.set_xticks(SQRT_TICKS)
    ax.set_xticklabels(TICK_LABELS, fontsize=7.5)
    # Wrap long outlet names for panel titles
    title_str = outlet.replace(" ", "\n") if len(outlet) > 12 else outlet
    ax.set_title(title_str, fontsize=9.5, fontweight="bold", color=color, pad=8)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

axes2[0].set_yticks(range(n_groups))
axes2[0].set_yticklabels(groups, fontsize=8.5)
axes2[0].tick_params(axis="y", length=0, pad=4)

fig2.text(0.5, -0.02,
          "Effect size  (z-score, square-root transformed axis)",
          ha="center", fontsize=9, color="#444444")

legend_elements = [
    Line2D([0], [0], marker=">", color="w", markerfacecolor="#555555",
           markersize=9, label="Over-represented  (z > 2)"),
    Line2D([0], [0], marker="<", color="w", markerfacecolor="#555555",
           markersize=9, label="Under-represented  (z < −2)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
           markeredgecolor="#555555", markeredgewidth=1.2,
           markersize=9, label="Not significant  (|z| ≤ 2)"),
]
fig2.legend(handles=legend_elements, loc="lower center", ncol=3,
            frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.1))

fig2.suptitle("Outlet attention by topic group",
              fontsize=13, fontweight="bold", y=1.02)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plt.tight_layout()

stem2 = FIG_DIR / f"outlet_topic_attention_dotplot{suffix}"
fig2.savefig(str(stem2) + ".pdf", bbox_inches="tight", dpi=300)
plt.close()
print(f"Dot plot  → {stem2}.pdf")
print(f"Font used: {FONT}")
