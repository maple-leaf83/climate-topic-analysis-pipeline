"""
cohesion_analysis.py
────────────────────
Computes cosine similarity between each article's embedding and its assigned
thematic group centroid, for both the Australian broadsheet and Guardian corpora.

Produces:
  figures/cluster_analysis/cohesion_clusters.pdf/.png
    Side-by-side boxplots: one panel per corpus, one box per thematic group.
  data/australian-no-letters/cohesion_scores_aus.csv
  data/guardian/cohesion_scores_guardian.csv

Usage:
    python cohesion_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

from config import DATA_DIR, FIGURES_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
EMB_DIR   = DATA_DIR / "embeddings"
FIG_DIR   = FIGURES_DIR / "cluster_analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)

AU_ASSIGN = DATA_DIR / "australian-no-letters" / "topic_assignments_aus.csv"
G_ASSIGN  = DATA_DIR / "guardian"              / "topic_assignments_guardian.csv"
AU_EMB    = EMB_DIR / "embeddings_cache_australian.npy"
G_EMB     = EMB_DIR / "embeddings_cache_guardian.npy"

# ── Group display order ────────────────────────────────────────────────────────
AU_GROUP_ORDER = [
    "Australian Politics",
    "Carbon Pricing & Domestic Climate Policy",
    "Regional & International Politics",
    "International Climate Policy",
    "Climate Science",
    "Physical & Ecological Impacts",
    "Energy Transition & Technology",
    "Culture Media & Society",
]

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


# ── Core computation ──────────────────────────────────────────────────────────

def compute_cohesion(assignments: pd.DataFrame, embeddings: np.ndarray,
                     group_order: list[str], corpus_label: str) -> pd.DataFrame:
    """
    For each article, compute cosine similarity to the centroid of its group.
    Returns a DataFrame with columns: group, cosine_similarity.
    """
    # L2-normalise so dot product = cosine similarity
    emb = embeddings.astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms

    # Compute group centroids
    centroids = {}
    for grp in group_order:
        idx = assignments.index[assignments["group"] == grp].tolist()
        if not idx:
            continue
        centroids[grp] = emb[idx].mean(axis=0)
        centroids[grp] /= np.linalg.norm(centroids[grp])  # re-normalise centroid

    # For each article, cosine similarity = dot(emb[i], centroid[group])
    rows = []
    for i, row in assignments.iterrows():
        grp = row["group"]
        if grp not in centroids:
            continue
        cos = float(np.dot(emb[i], centroids[grp]))
        rows.append({"group": grp, "cosine_similarity": cos})

    df = pd.DataFrame(rows)
    print(f"\n{corpus_label} cohesion summary:")
    print(df.groupby("group")["cosine_similarity"]
            .agg(["median", "mean", "min", "max"])
            .reindex(group_order)
            .round(3)
            .to_string())
    return df


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_cohesion_single(scores: pd.DataFrame, group_order: list[str],
                         title: str, out_stem: Path):
    """Single-corpus horizontal boxplot, one box per thematic group."""
    data   = [scores.loc[scores["group"] == g, "cosine_similarity"].values
              for g in group_order if g in scores["group"].values]
    labels = [g for g in group_order if g in scores["group"].values]

    fig, ax = plt.subplots(figsize=(7, max(4, len(labels) * 0.55)))

    ax.boxplot(data, vert=False, patch_artist=True,
               flierprops=dict(marker=".", markersize=2,
                               markerfacecolor="#aaaaaa", linestyle="none"),
               medianprops=dict(color="#333333", linewidth=1.5),
               boxprops=dict(facecolor="#c6dbef", linewidth=0.8),
               whiskerprops=dict(linewidth=0.8),
               capprops=dict(linewidth=0.8))

    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Cosine similarity to group centroid", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlim(0.3, 1.02)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    ext = ".pdf"
    fig.savefig(str(out_stem.with_suffix(ext)), bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → {out_stem.name}.pdf")


def plot_cohesion(au_scores: pd.DataFrame, g_scores: pd.DataFrame):
    plot_cohesion_single(
        au_scores, AU_GROUP_ORDER,
        "Australian broadsheets",
        FIG_DIR / "cohesion_clusters_au",
    )
    plot_cohesion_single(
        g_scores, G_GROUP_ORDER,
        "The Guardian",
        FIG_DIR / "cohesion_clusters_guardian",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data…")
    au_df = pd.read_csv(AU_ASSIGN, low_memory=False)
    g_df  = pd.read_csv(G_ASSIGN,  low_memory=False)
    au_emb = np.load(str(AU_EMB))
    g_emb  = np.load(str(G_EMB))

    print(f"AU:  {len(au_df):,} articles, embeddings {au_emb.shape}")
    print(f"G:   {len(g_df):,} articles, embeddings {g_emb.shape}")

    au_scores = compute_cohesion(au_df, au_emb, AU_GROUP_ORDER, "Australian broadsheets")
    g_scores  = compute_cohesion(g_df,  g_emb,  G_GROUP_ORDER,  "Guardian")

    # Save CSVs
    au_scores.to_csv(DATA_DIR / "australian-no-letters" / "cohesion_scores_aus.csv",
                     index=False)
    g_scores.to_csv(DATA_DIR / "guardian" / "cohesion_scores_guardian.csv",
                    index=False)
    print("\nCSVs saved.")

    plot_cohesion(au_scores, g_scores)
