"""
analyse_cohesion.py
-------------------
Generates cohesion_clusters.pdf: boxplot of cosine similarity between each
article's nomic-embed-text-v1 embedding and its assigned thematic group
centroid (n = 9,863 articles across 7 groups).

Outputs
  figures/cohesion/cohesion_clusters.pdf
  figures/cohesion/cohesion_clusters.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
EMB_PATH  = os.path.join(ROOT, "data", "embeddings", "embeddings_cache.npy")
META_PATH = os.path.join(ROOT, "data", "australian-no-letters", "topic_assignments.csv")
FIG_DIR   = os.path.join(ROOT, "figures", "cohesion")
os.makedirs(FIG_DIR, exist_ok=True)

THEME_ORDER = [
    "Political leadership & party dynamics",
    "Carbon pricing & emissions policy",
    "Climate science & physical impacts",
    "Energy policy & transition",
    "Environment & biodiversity",
    "Media, culture & society",
    "International climate diplomacy",
]

SHORT = {
    "Political leadership & party dynamics": "Political\nleadership",
    "Carbon pricing & emissions policy":     "Carbon\npricing",
    "Climate science & physical impacts":    "Climate\nscience",
    "Energy policy & transition":            "Energy\npolicy",
    "Environment & biodiversity":            "Environment",
    "Media, culture & society":              "Media &\nculture",
    "International climate diplomacy":       "Intl.\ndiplomacy",
}

# ── helpers ────────────────────────────────────────────────────────────────────

def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


def centroid(emb: np.ndarray) -> np.ndarray:
    c = emb.mean(axis=0)
    norm = np.linalg.norm(c)
    return c / (norm if norm > 0 else 1.0)


def cos_to_centroid(emb: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row to the group centroid (emb already L2-normed)."""
    c = centroid(emb)
    return emb @ c


# ── load data ──────────────────────────────────────────────────────────────────
print("Loading embeddings …")
emb_raw = np.load(EMB_PATH).astype(np.float32)
meta    = pd.read_csv(META_PATH)

assert len(emb_raw) == len(meta), (
    f"Embedding rows ({len(emb_raw)}) ≠ metadata rows ({len(meta)})"
)
print(f"  {len(emb_raw):,} articles  ×  {emb_raw.shape[1]} dims")

emb = l2_normalize(emb_raw)

# ── boxplot: article cosine-to-centroid by thematic group ──────────────────────
print("Computing cosine similarities …")
data_by_group = []
for theme in THEME_ORDER:
    mask = meta["theme"] == theme
    idx  = meta.index[mask].values
    c2c  = cos_to_centroid(emb[idx])
    data_by_group.append(c2c)
    print(f"  {theme:<45}  n={mask.sum():>4}  median={float(np.median(c2c)):.3f}")

print("Generating boxplot …")
fig, ax = plt.subplots(figsize=(8, 5))

bp = ax.boxplot(data_by_group, patch_artist=True, notch=False,
                medianprops={"color": "black", "lw": 1.5},
                whiskerprops={"lw": 1.0},
                capprops={"lw": 1.0},
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})

group_colors = ["#1b7837", "#4d9221", "#2166ac", "#d6604d", "#01665e", "#8073ac", "#b2182b"]
for patch, color in zip(bp["boxes"], group_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)

ax.set_xticks(range(1, 8))
ax.set_xticklabels([SHORT[t] for t in THEME_ORDER], fontsize=9)
ax.set_ylabel("Cosine similarity to group centroid", fontsize=10)
ax.set_ylim(0.6, 1.01)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cohesion_clusters.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "cohesion_clusters.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Done → figures/cohesion/cohesion_clusters.{pdf,png}")
