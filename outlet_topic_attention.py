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
    python outlet_topic_attention.py

Outputs (written relative to repo root via config.py):
  data/australian-no-letters/outlet_observed_counts.csv
  data/australian-no-letters/outlet_representation_ratios.csv
  data/australian-no-letters/outlet_binomial_zscores.csv
  figures/cluster_analysis/outlet_topic_attention_heatmap.pdf
  figures/cluster_analysis/outlet_topic_attention_dotplot.pdf
"""

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

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = DATA_DIR / "australian-no-letters" / "topic_assignments.csv"
OUT_DIR   = DATA_DIR / "australian-no-letters"
FIG_DIR   = FIGURES_DIR / "cluster_analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Outlets (canonical names matching publication column in assignments CSV) ───
OUTLETS = ["The Australian", "The Age", "Sydney Morning Herald", "Canberra Times"]

# Seaborn "colorblind" palette (Wong 2011)
OUTLET_COLORS = {
    "The Australian":      "#0173b2",
    "The Age":             "#de8f05",
    "Sydney Morning Herald": "#029e73",
    "Canberra Times":      "#d55e00",
}

NOISE_GROUPS = {"Noise"}

# Preferred display order for rows (matches THEME_ORDER in analyse_clusters.py)
THEME_ORDER = [
    "Political leadership & party dynamics",
    "Carbon pricing & emissions policy",
    "Climate science & physical impacts",
    "Energy policy & transition",
    "Environment & biodiversity",
    "Media, culture & society",
    "International climate diplomacy",
]

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

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df = df[df["publication"].isin(OUTLETS)].copy()
df = df[df["theme"].notna() & ~df["theme"].isin(NOISE_GROUPS)].copy()

total = len(df)
print(f"Articles (noise excluded): {total:,}")
print(f"Outlets:\n{df['publication'].value_counts().to_string()}")
print(f"\nGroups ({df['theme'].nunique()}): {sorted(df['theme'].unique())}\n")

# ── Corpus-level expected proportions ─────────────────────────────────────────
p_exp = df["publication"].value_counts() / total
p_exp = p_exp.reindex(OUTLETS)
print("Expected proportions (corpus share):")
print(p_exp.round(4).to_string())

# ── Contingency table ─────────────────────────────────────────────────────────
ct = pd.crosstab(df["theme"], df["publication"])[OUTLETS]
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
ct.to_csv(OUT_DIR / "outlet_observed_counts.csv")
ratio.round(4).to_csv(OUT_DIR / "outlet_representation_ratios.csv")
z.round(3).to_csv(OUT_DIR / "outlet_binomial_zscores.csv")
print("Saved CSVs")

# ── Omnibus chi-square test of independence ────────────────────────────────
chi2_stat, p_val, dof, expected = chi2_contingency(ct)
exp_min = expected.min()
exp_low = (expected < 5).sum().sum()
print(f"\n── Pearson chi-square (omnibus) ──")
print(f"  χ²({dof}) = {chi2_stat:.2f},  p = {p_val:.2e}")
print(f"  Min expected cell count: {exp_min:.2f}  |  Cells < 5: {exp_low}")
(OUT_DIR / "outlet_chi2.txt").write_text(
    f"chi2={chi2_stat:.4f}\ndf={dof}\np={p_val:.4e}\n"
    f"expected_min={exp_min:.4f}\ncells_below_5={int(exp_low)}\n"
)

print("\n── Representation Ratios ──")
print(ratio.round(2).to_string())
print("\n── Binomial z-scores ──")
print(z.round(1).to_string())

# ── Row order: follow THEME_ORDER, then any remaining groups ─────────────────
row_order  = [g for g in THEME_ORDER if g in z.index] + \
             [g for g in z.index if g not in THEME_ORDER]
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
    figsize=(10, 5),
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
         "Rows in theme-order.  "
         "Red = over-represented relative to corpus share;  Blue = under-represented.",
         ha="center", fontsize=8, color="#555555", style="italic")
fig.suptitle("Outlet attention by topic group", fontsize=13, fontweight="bold", y=1.01)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plt.tight_layout()

fig.savefig(str(FIG_DIR / "outlet_topic_attention_heatmap.pdf"), bbox_inches="tight", dpi=300)
plt.close()
print(f"\nHeatmap → outlet_topic_attention_heatmap.pdf")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Dot plot (faceted by outlet, sqrt-transformed x-axis)
# ═══════════════════════════════════════════════════════════════════════════════
R_HIGH = 1.25
R_LOW  = 0.75

groups   = list(row_order)
n_groups = len(groups)
y_pos    = {g: i for i, g in enumerate(groups)}

def zsqrt(val):
    return float(np.sign(val) * np.sqrt(abs(val)))

n_panels  = len(OUTLETS)
fig_width = 3.0 * n_panels + 1.5
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
        r_val = float(ratio.loc[group, outlet])

        if r_val > R_HIGH:
            ax.scatter(x_val, y, marker=">", s=90, color=color,
                       zorder=3, linewidths=0)
        elif r_val < R_LOW:
            ax.scatter(x_val, y, marker="<", s=90, color=color,
                       zorder=3, linewidths=0)
        else:
            ax.scatter(x_val, y, marker="s", s=60, facecolors="none",
                       edgecolors=color, linewidths=1.2, zorder=3)

    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    for yg in range(n_groups):
        ax.axhline(yg, color="#e0e0e0", linewidth=0.5, zorder=0)
    ax.set_xlim(X_LIM)
    ax.set_ylim(-0.8, n_groups - 0.2)
    ax.set_xticks(SQRT_TICKS)
    ax.set_xticklabels(TICK_LABELS, fontsize=7.5)
    title_str = outlet.replace(" ", "\n") if len(outlet) > 12 else outlet
    ax.set_title(title_str, fontsize=9.5, fontweight="bold", color=color, pad=8)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")

axes2[0].set_yticks(range(n_groups))
axes2[0].set_yticklabels(groups, fontsize=8.5)
axes2[0].tick_params(axis="y", length=0, pad=4)

fig2.text(0.5, -0.02,
          "Effect size  (z-score, square-root transformed axis)  —  "
          "marker shape indicates representation ratio r, not z-score significance",
          ha="center", fontsize=8.5, color="#444444")

legend_elements = [
    Line2D([0], [0], marker=">", color="w", markerfacecolor="#555555",
           markersize=9, label=r"Substantially over-represented  ($r > 1.25$)"),
    Line2D([0], [0], marker="<", color="w", markerfacecolor="#555555",
           markersize=9, label=r"Substantially under-represented  ($r < 0.75$)"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="none",
           markeredgecolor="#555555", markeredgewidth=1.2,
           markersize=9, label=r"Within expected range  ($0.75 \leq r \leq 1.25$)"),
]
fig2.legend(handles=legend_elements, loc="lower center", ncol=3,
            frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.1))

fig2.suptitle("Outlet attention by topic group",
              fontsize=13, fontweight="bold", y=1.02)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plt.tight_layout()

fig2.savefig(str(FIG_DIR / "outlet_topic_attention_dotplot.pdf"), bbox_inches="tight", dpi=300)
plt.close()
print(f"Dot plot  → outlet_topic_attention_dotplot.pdf")
print(f"Font used: {FONT}")
