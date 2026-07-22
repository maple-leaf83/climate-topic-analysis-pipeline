"""
analyse_cohesion.py
-------------------
Computes cosine-similarity-based cohesion for:
  (a) each BERTopic cluster  (topic_id)
  (b) each of the 7 thematic super-groups  (theme)
  (c) k-means with k=7 — verifies that unsupervised clustering recovers the
      same 7-group structure as the manual thematic groupings

Uses the full 9,863 × 768 embedding matrix, which maps 1-to-1 to
data/australian-no-letters/topic_assignments.csv.

Metrics per group/cluster
  - mean cosine similarity of each article to its centroid  (fast, exact)
  - mean pairwise cosine similarity  (exact for n ≤ 500; sampled otherwise)
  - discriminability: own-centroid cosine minus best-other-centroid cosine

k-means validation
  - Spherical k-means on L2-normalised embeddings (cosine space)
  - k-means++ initialisation, n_init=10 runs, best inertia kept
  - Optimal assignment of k-means clusters → manual groups via greedy
    maximum-overlap matching (equivalent to Hungarian for k=7)
  - Adjusted Rand Index (ARI) and Normalised Mutual Information (NMI)
  - Contingency heatmap

Outputs
  results/cohesion/cohesion_clusters.csv
  results/cohesion/cohesion_groups.csv
  results/cohesion/discriminability_groups.csv
  results/cohesion/kmeans_contingency.csv
  results/cohesion/kmeans_summary.txt
  figures/cohesion/cohesion_clusters.pdf/png
  figures/cohesion/cohesion_groups.pdf/png
  figures/cohesion/cohesion_violin.pdf/png
  figures/cohesion/kmeans_heatmap.pdf/png
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
RES_DIR   = os.path.join(ROOT, "results", "cohesion")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

THEME_ORDER = [
    "Political leadership & party dynamics",
    "Carbon pricing & emissions policy",
    "Climate science & physical impacts",
    "Energy policy & transition",
    "Media, culture & society",
    "Environment & biodiversity",
    "International climate diplomacy",
]

NOISE_TOPICS = {57, 83}   # excluded from cluster-level analysis
NOISE_THEME  = "Noise"    # excluded from k-means comparison
SAMPLE_SIZE  = 500        # max articles for exact pairwise computation per group
K            = 7          # number of k-means clusters
KMEANS_INITS = 10         # independent restarts (best inertia kept)
KMEANS_ITERS = 300        # max iterations per run

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
    return emb @ c          # dot product = cosine since both are unit vectors


def mean_pairwise_cos(emb: np.ndarray, seed: int = 42) -> float:
    """Mean pairwise cosine similarity (upper triangle). Samples if n > SAMPLE_SIZE."""
    n = len(emb)
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    if n > SAMPLE_SIZE:
        idx = rng.choice(n, SAMPLE_SIZE, replace=False)
        emb = emb[idx]
    sim = emb @ emb.T
    upper = sim[np.triu_indices(len(emb), k=1)]
    return float(upper.mean())


# ── k-means helpers ────────────────────────────────────────────────────────────

def spherical_kmeans(emb: np.ndarray, k: int,
                     n_init: int = KMEANS_INITS,
                     max_iter: int = KMEANS_ITERS,
                     seed: int = 42) -> np.ndarray:
    """
    Spherical k-means on L2-normalised embeddings.
    Assignment step: argmax cosine similarity to centroid.
    Update step:     mean of assigned points, re-normalised.
    k-means++ initialisation; best inertia over n_init runs returned.
    """
    rng = np.random.default_rng(seed)
    n   = len(emb)
    best_labels   = None
    best_inertia  = -np.inf

    for run in range(n_init):
        # ── k-means++ initialisation ──────────────────────────────────────────
        first = int(rng.integers(n))
        cents = [emb[first].copy()]
        for _ in range(k - 1):
            # min cosine distance to nearest existing centroid
            sims = np.stack([emb @ c for c in cents], axis=1).max(axis=1)  # (n,)
            dist = (1.0 - sims).clip(0)
            probs = dist / dist.sum()
            idx = int(rng.choice(n, p=probs))
            cents.append(emb[idx].copy())
        cents = np.array(cents, dtype=np.float32)   # (k, d)

        # ── EM loop ───────────────────────────────────────────────────────────
        labels = np.zeros(n, dtype=np.int32)
        for it in range(max_iter):
            # Assign: argmax cosine similarity
            sims_mat = emb @ cents.T          # (n, k)
            new_labels = sims_mat.argmax(axis=1).astype(np.int32)

            if np.array_equal(new_labels, labels) and it > 0:
                break
            labels = new_labels

            # Update: normalised mean
            new_cents = np.zeros_like(cents)
            for j in range(k):
                members = emb[labels == j]
                if len(members) == 0:
                    # reinitialise empty cluster to a random point
                    new_cents[j] = emb[int(rng.integers(n))]
                else:
                    c = members.mean(axis=0)
                    norm = np.linalg.norm(c)
                    new_cents[j] = c / (norm if norm > 0 else 1.0)
            cents = new_cents

        # Inertia = total cosine similarity to assigned centroid (higher = better)
        sims_mat = emb @ cents.T
        inertia  = sims_mat[np.arange(n), labels].sum()
        if inertia > best_inertia:
            best_inertia = inertia
            best_labels  = labels.copy()

        print(f"    run {run+1:2d}/{n_init}  inertia={inertia:.4f}", flush=True)

    print(f"  Best inertia: {best_inertia:.4f}")
    return best_labels


def greedy_max_match(overlap: np.ndarray) -> list:
    """
    Greedy maximum-overlap assignment: iteratively pick the largest cell,
    assign that (k-means row, manual-group col) pair, remove both from
    consideration.  For k=7 this is equivalent to optimal Hungarian matching
    in practice.  Returns list of (km_cluster, manual_group_idx) pairs.
    """
    mat = overlap.astype(float).copy()
    k   = mat.shape[0]
    assignment = []
    used_rows, used_cols = set(), set()
    for _ in range(k):
        # mask already-assigned rows/cols
        tmp = mat.copy()
        tmp[list(used_rows), :] = -1
        tmp[:, list(used_cols)] = -1
        r, c = np.unravel_index(tmp.argmax(), tmp.shape)
        assignment.append((int(r), int(c)))
        used_rows.add(int(r))
        used_cols.add(int(c))
    return assignment


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Adjusted Rand Index, implemented from scratch using numpy."""
    def comb2(n): return n * (n - 1) / 2

    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    # Contingency table
    contingency = np.array([
        [(labels_true == c) & (labels_pred == k) for k in clusters]
        for c in classes
    ], dtype=np.int64).sum(axis=2).T   # shape (n_clusters, n_classes)

    sum_comb_c = sum(comb2(contingency[i, :].sum()) for i in range(len(clusters)))
    sum_comb_k = sum(comb2(contingency[:, j].sum()) for j in range(len(classes)))
    sum_comb   = sum(comb2(n) for n in contingency.ravel())
    n          = len(labels_true)
    total_comb = comb2(n)

    expected = sum_comb_c * sum_comb_k / total_comb
    max_val  = (sum_comb_c + sum_comb_k) / 2

    if max_val == expected:
        return 1.0
    return (sum_comb - expected) / (max_val - expected)


def normalised_mutual_info(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Normalised Mutual Information (arithmetic mean normalisation)."""
    n = len(labels_true)

    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / n
        return -np.sum(probs * np.log(probs + 1e-12))

    classes  = np.unique(labels_true)
    clusters = np.unique(labels_pred)

    mi = 0.0
    for c in classes:
        for k in clusters:
            n_ck = np.sum((labels_true == c) & (labels_pred == k))
            n_c  = np.sum(labels_true == c)
            n_k  = np.sum(labels_pred == k)
            if n_ck == 0:
                continue
            mi += (n_ck / n) * np.log((n * n_ck) / (n_c * n_k))

    h_true = entropy(labels_true)
    h_pred = entropy(labels_pred)
    denom  = (h_true + h_pred) / 2
    return mi / denom if denom > 0 else 0.0


def cohesion_stats(emb: np.ndarray) -> dict:
    """Full cohesion stats for one group/cluster."""
    c2c = cos_to_centroid(emb)
    return {
        "n":                   len(emb),
        "mean_cos_centroid":   float(c2c.mean()),
        "median_cos_centroid": float(np.median(c2c)),
        "sd_cos_centroid":     float(c2c.std()),
        "p10_cos_centroid":    float(np.percentile(c2c, 10)),
        "p90_cos_centroid":    float(np.percentile(c2c, 90)),
        "mean_pairwise_cos":   mean_pairwise_cos(emb),
    }


# ── load data ──────────────────────────────────────────────────────────────────
print("Loading embeddings …")
emb_raw = np.load(EMB_PATH).astype(np.float32)
meta    = pd.read_csv(META_PATH)

assert len(emb_raw) == len(meta), (
    f"Embedding rows ({len(emb_raw)}) ≠ metadata rows ({len(meta)})"
)

print(f"  {len(emb_raw):,} articles  ×  {emb_raw.shape[1]} dims")

# L2-normalise once
emb = l2_normalize(emb_raw)

# ── (A) CLUSTER-LEVEL COHESION ─────────────────────────────────────────────────
print("\nComputing cluster-level cohesion …")
cluster_rows = []

for tid, grp in meta.groupby("topic_id"):
    if int(tid) in NOISE_TOPICS:
        continue
    idx     = grp.index.values
    emb_g   = emb[idx]
    stats   = cohesion_stats(emb_g)
    theme   = grp["theme"].iloc[0]
    stats.update({"topic_id": int(tid), "theme": theme})
    cluster_rows.append(stats)
    if len(cluster_rows) % 20 == 0:
        print(f"  … {len(cluster_rows)} clusters done")

df_clusters = pd.DataFrame(cluster_rows).sort_values("topic_id")
df_clusters.to_csv(os.path.join(RES_DIR, "cohesion_clusters.csv"), index=False)
print(f"  {len(df_clusters)} clusters saved → results/cohesion/cohesion_clusters.csv")

# ── (B) GROUP-LEVEL COHESION ───────────────────────────────────────────────────
print("\nComputing group-level cohesion …")
group_rows = []

for theme in THEME_ORDER:
    mask  = meta["theme"] == theme
    idx   = meta.index[mask].values
    emb_g = emb[idx]
    stats = cohesion_stats(emb_g)
    stats["theme"] = theme
    group_rows.append(stats)
    print(f"  {theme[:45]:<45}  n={stats['n']:>4}  "
          f"mean_cos_centroid={stats['mean_cos_centroid']:.4f}  "
          f"mean_pairwise={stats['mean_pairwise_cos']:.4f}")

df_groups = pd.DataFrame(group_rows)
df_groups.to_csv(os.path.join(RES_DIR, "cohesion_groups.csv"), index=False)
print("  Saved → results/cohesion/cohesion_groups.csv")

# ── (C) DISCRIMINABILITY (group level) ─────────────────────────────────────────
print("\nComputing group-level discriminability …")

# Build one centroid per group
centroids = {}
for theme in THEME_ORDER:
    mask = meta["theme"] == theme
    centroids[theme] = centroid(emb[meta.index[mask].values])

C = np.stack([centroids[t] for t in THEME_ORDER])   # (7, 768)

disc_rows = []
for theme in THEME_ORDER:
    mask  = meta["theme"] == theme
    idx   = meta.index[mask].values
    emb_g = emb[idx]               # (n, 768)

    sims_all  = emb_g @ C.T        # (n, 7) — cosine to every group centroid
    own_idx   = THEME_ORDER.index(theme)
    own_cos   = sims_all[:, own_idx]
    other_cos = np.delete(sims_all, own_idx, axis=1)
    best_other = other_cos.max(axis=1)
    disc = own_cos - best_other

    is_outlier = (meta.loc[idx, "topic_prob"] == 0).values
    for flag, label in [(None, "all"), (False, "direct"), (True, "outlier")]:
        sel = disc if flag is None else disc[is_outlier == flag]
        n   = len(sel)
        disc_rows.append({
            "theme":          theme,
            "subset":         label,
            "n":              n,
            "pct_positive":   float((sel > 0).mean() * 100) if n > 0 else float("nan"),
            "mean_disc":      float(sel.mean())             if n > 0 else float("nan"),
            "median_disc":    float(np.median(sel))         if n > 0 else float("nan"),
        })

df_disc = pd.DataFrame(disc_rows)
df_disc.to_csv(os.path.join(RES_DIR, "discriminability_groups.csv"), index=False)
print("  Saved → results/cohesion/discriminability_groups.csv")

# ── PRINT SUMMARY ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("GROUP-LEVEL COHESION SUMMARY")
print("="*70)
cols = ["theme", "n", "mean_cos_centroid", "sd_cos_centroid", "mean_pairwise_cos"]
print(df_groups[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\nDISCRIMINABILITY (all articles per group)")
disc_all = df_disc[df_disc["subset"] == "all"][["theme","n","pct_positive","mean_disc"]]
print(disc_all.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Overall
all_mask = meta["theme"].isin(THEME_ORDER)
all_idx  = meta.index[all_mask].values
sims_all = emb[all_idx] @ C.T
own_idxs = np.array([THEME_ORDER.index(t) for t in meta.loc[all_idx, "theme"]])
own_cos  = sims_all[np.arange(len(all_idx)), own_idxs]
best_other_global = np.array([
    sims_all[i, [j for j in range(7) if j != own_idxs[i]]].max()
    for i in range(len(all_idx))
])
disc_global = own_cos - best_other_global
print(f"\nOverall discriminability (n={len(disc_global):,}): "
      f"mean={disc_global.mean():.4f}, "
      f"median={np.median(disc_global):.4f}, "
      f"% > 0 = {100*(disc_global>0).mean():.1f}%")

# ── FIGURES ────────────────────────────────────────────────────────────────────
print("\nGenerating figures …")

SHORT = {
    "Political leadership & party dynamics": "Political\nleadership",
    "Carbon pricing & emissions policy":     "Carbon\npricing",
    "Climate science & physical impacts":    "Climate\nscience",
    "Energy policy & transition":            "Energy\npolicy",
    "Media, culture & society":              "Media &\nculture",
    "Environment & biodiversity":            "Environment",
    "International climate diplomacy":       "Intl.\ndiplomacy",
}

# ── Figure 1: group-level bar chart (mean cosine ± SD to centroid) ─────────────
fig, ax = plt.subplots(figsize=(10, 4.5))

labels = [SHORT[t] for t in THEME_ORDER]
means  = [df_groups.loc[df_groups["theme"]==t, "mean_cos_centroid"].values[0] for t in THEME_ORDER]
sds    = [df_groups.loc[df_groups["theme"]==t, "sd_cos_centroid"].values[0]   for t in THEME_ORDER]
ns     = [df_groups.loc[df_groups["theme"]==t, "n"].values[0]                  for t in THEME_ORDER]

bars = ax.bar(range(7), means, yerr=sds, capsize=4,
              color="#4393c3", edgecolor="white", linewidth=0.6,
              error_kw={"elinewidth": 1.2, "ecolor": "#333"}, alpha=0.85)

for i, (m, n) in enumerate(zip(means, ns)):
    ax.text(i, 0.757, f"n={n:,}", ha="center", va="bottom", fontsize=7.5, color="#444")

ax.set_xticks(range(7))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.75, 0.92)
ax.set_ylabel("Mean cosine similarity to group centroid", fontsize=10)
# ax.set_title("Intra-group semantic cohesion (mean ± SD)\n"
#              "Each article's cosine similarity to its thematic group centroid",
#              fontsize=10, fontweight="bold")
ax.axhline(np.mean(means), color="#d6604d", ls="--", lw=1.2, label=f"Grand mean = {np.mean(means):.3f}")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cohesion_groups.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "cohesion_groups.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved figures/cohesion/cohesion_groups.{pdf,png}")

# ── Figure 2: per-group distribution of article cosine-to-centroid (box plots) ─
fig, ax = plt.subplots(figsize=(8, 5))

data_by_group = []
for theme in THEME_ORDER:
    mask = meta["theme"] == theme
    idx  = meta.index[mask].values
    c2c  = cos_to_centroid(emb[idx])
    data_by_group.append(c2c)

bp = ax.boxplot(data_by_group, patch_artist=True, notch=False,
                medianprops={"color": "black", "lw": 1.5},
                whiskerprops={"lw": 1.0},
                capprops={"lw": 1.0},
                flierprops={"marker": ".", "markersize": 2, "alpha": 0.3})

group_colors = ["#1b7837","#4d9221","#2166ac","#d6604d","#8073ac","#01665e","#b2182b"]
for patch, color in zip(bp["boxes"], group_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.65)

ax.set_xticks(range(1, 8))
ax.set_xticklabels([SHORT[t] for t in THEME_ORDER], fontsize=9)
ax.set_ylabel("Cosine similarity to group centroid", fontsize=10)
# ax.set_title("Distribution of article-to-centroid cosine similarity by thematic group",
#              fontsize=10, fontweight="bold")
ax.set_ylim(0.6, 1.01)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cohesion_clusters.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "cohesion_clusters.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved figures/cohesion/cohesion_clusters.{pdf,png}")

# ── Figure 3: outlier vs direct violin ────────────────────────────────────────
fig, axes = plt.subplots(1, 7, figsize=(14, 4.5), sharey=True)

for ax, theme in zip(axes, THEME_ORDER):
    mask    = meta["theme"] == theme
    idx     = meta.index[mask].values
    c2c     = cos_to_centroid(emb[idx])
    is_out  = (meta.loc[idx, "topic_prob"] == 0).values

    for j, (sel, color, label) in enumerate([
            (c2c[~is_out], "#2166ac", "Direct"),
            (c2c[is_out],  "#d6604d", "Outlier")]):
        if len(sel) > 1:
            parts = ax.violinplot([sel], positions=[j], widths=0.7,
                                  showmedians=True, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.65)
            parts["cmedians"].set_color("black")
            parts["cmedians"].set_linewidth(1.5)

    ax.set_title(SHORT[theme].replace("\n", " "), fontsize=8, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Direct", "Outlier"], fontsize=7, rotation=30)
    ax.set_ylim(0.6, 1.01)
    if ax == axes[0]:
        ax.set_ylabel("Cosine similarity to group centroid", fontsize=9)

fig.suptitle("Cosine similarity to group centroid: HDBSCAN direct vs reassigned outlier articles",
             fontsize=10, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cohesion_violin.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "cohesion_violin.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved figures/cohesion/cohesion_violin.{pdf,png}")

# ── (D) K-MEANS VALIDATION ─────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"K-MEANS VALIDATION  (k={K}, spherical, {KMEANS_INITS} restarts)")
print("="*70)

# Exclude noise articles (105 articles with theme == "Noise")
nonnoise_mask = meta["theme"].isin(THEME_ORDER)
nonnoise_idx  = meta.index[nonnoise_mask].values
emb_nn        = emb[nonnoise_idx]            # (9758, 768)
themes_nn     = meta.loc[nonnoise_idx, "theme"].values

print(f"  Non-noise articles: {len(emb_nn):,}")
print(f"  Running spherical k-means …")

km_labels = spherical_kmeans(emb_nn, k=K)

# ── Contingency table: km_labels (rows) × manual group (cols) ─────────────────
# Encode manual groups as integers in THEME_ORDER order
theme_to_int = {t: i for i, t in enumerate(THEME_ORDER)}
manual_labels = np.array([theme_to_int[t] for t in themes_nn])

contingency = np.zeros((K, K), dtype=int)
for km, man in zip(km_labels, manual_labels):
    contingency[km, man] += 1

# ── Optimal assignment: k-means cluster → manual group ────────────────────────
assignment = greedy_max_match(contingency)
km_to_theme = {km: THEME_ORDER[man] for km, man in assignment}

print("\n  Optimal assignment (k-means cluster → manual group):")
print(f"  {'KM cluster':>12}  {'Manual group':<45}  {'Overlap %':>10}  {'n KM':>6}  {'n Manual':>8}")
for km, man in sorted(assignment):
    n_km  = contingency[km, :].sum()
    n_man = contingency[:, man].sum()
    pct   = 100 * contingency[km, man] / n_km if n_km > 0 else 0
    print(f"  {km:>12}  {THEME_ORDER[man]:<45}  {pct:>9.1f}%  {n_km:>6}  {n_man:>8}")

# ── ARI and NMI ───────────────────────────────────────────────────────────────
# Remap km_labels via assignment so cluster indices align with manual group indices
km_remapped = np.array([theme_to_int[km_to_theme[km]] for km in km_labels])

ari = adjusted_rand_index(manual_labels, km_remapped)
nmi = normalised_mutual_info(manual_labels, km_remapped)

# Per-group precision and recall under the optimal assignment
print(f"\n  Adjusted Rand Index (ARI) = {ari:.4f}")
print(f"  Normalised Mutual Info (NMI) = {nmi:.4f}")

print("\n  Per-group precision / recall:")
print(f"  {'Group':<45}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}")
f1_scores = []
for km, man in sorted(assignment, key=lambda x: x[1]):
    tp  = contingency[km, man]
    fp  = contingency[km, :].sum() - tp
    fn  = contingency[:, man].sum() - tp
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1  = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0
    f1_scores.append(f1)
    print(f"  {THEME_ORDER[man]:<45}  {pre:>10.3f}  {rec:>8.3f}  {f1:>6.3f}")
print(f"  {'Macro-avg F1':<45}  {'':>10}  {'':>8}  {np.mean(f1_scores):>6.3f}")

# ── Save contingency table ─────────────────────────────────────────────────────
# Reorder rows so km cluster 0 = Political leadership etc (via assignment)
ordered_km = [km for km, _ in sorted(assignment, key=lambda x: x[1])]
df_cont = pd.DataFrame(
    contingency[ordered_km, :],
    index=[f"KM→{THEME_ORDER[man][:25]}" for _, man in sorted(assignment, key=lambda x: x[1])],
    columns=[t[:25] for t in THEME_ORDER]
)
df_cont.to_csv(os.path.join(RES_DIR, "kmeans_contingency.csv"))
print(f"\n  Contingency table saved → results/cohesion/kmeans_contingency.csv")

# Save summary text
summary_lines = [
    f"K-means validation (k={K}, spherical k-means on L2-normalised nomic-embed-text-v1 embeddings)",
    f"Non-noise articles: {len(emb_nn):,}",
    f"ARI  = {ari:.4f}",
    f"NMI  = {nmi:.4f}",
    f"Macro-avg F1 = {np.mean(f1_scores):.4f}",
    "",
    "Per-group precision / recall / F1:",
]
for (km, man), f1 in zip(sorted(assignment, key=lambda x: x[1]), f1_scores):
    tp  = contingency[km, man]
    fp  = contingency[km, :].sum() - tp
    fn  = contingency[:, man].sum() - tp
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    summary_lines.append(f"  {THEME_ORDER[man]:<45}  P={pre:.3f}  R={rec:.3f}  F1={f1:.3f}")

with open(os.path.join(RES_DIR, "kmeans_summary.txt"), "w") as f:
    f.write("\n".join(summary_lines))
print("  Summary saved → results/cohesion/kmeans_summary.txt")

# ── Figure 4: k-means contingency heatmap ─────────────────────────────────────
# Show row-normalised overlap (% of k-means cluster going to each manual group)
SHORT_WRAP = {
    "Political leadership & party dynamics": "Political\nleadership",
    "Carbon pricing & emissions policy":     "Carbon\npricing",
    "Climate science & physical impacts":    "Climate\nscience",
    "Energy policy & transition":            "Energy\npolicy",
    "Media, culture & society":              "Media &\nculture",
    "Environment & biodiversity":            "Environment",
    "International climate diplomacy":       "Intl.\ndiplomacy",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Panel A — row-normalised heatmap (% of k-means cluster in each manual group)
ax = axes[0]
cont_ordered = contingency[ordered_km, :]             # reorder rows
row_sums = cont_ordered.sum(axis=1, keepdims=True)
cont_pct = 100 * cont_ordered / np.where(row_sums == 0, 1, row_sums)

im = ax.imshow(cont_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
plt.colorbar(im, ax=ax, label="% of k-means cluster")

row_labels = [SHORT_WRAP[THEME_ORDER[man]].replace("\n", " ")
              for _, man in sorted(assignment, key=lambda x: x[1])]
col_labels = [SHORT_WRAP[t].replace("\n", " ") for t in THEME_ORDER]

ax.set_xticks(range(K))
ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
ax.set_yticks(range(K))
ax.set_yticklabels([f"KM {i+1}: {l}" for i, l in enumerate(row_labels)], fontsize=8)
ax.set_xlabel("Manual thematic group", fontsize=9)
ax.set_ylabel("k-means cluster (labelled by best-matching manual group)", fontsize=9)
ax.set_title(f"A  k-means (k=7) vs manual groups\n(row-normalised; % of k-means cluster)",
             fontsize=10, fontweight="bold", loc="left")

# Annotate cells
for i in range(K):
    for j in range(K):
        val = cont_pct[i, j]
        color = "white" if val > 55 else "black"
        ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                fontsize=7, color=color, fontweight="bold" if i == j else "normal")

# Diagonal box to highlight the matched cells
for idx, (km, man) in enumerate(sorted(assignment, key=lambda x: x[1])):
    ax.add_patch(plt.Rectangle(
        (man - 0.5, idx - 0.5), 1, 1,
        fill=False, edgecolor="#d6604d", linewidth=2.5
    ))

# Panel B — per-group precision, recall, F1 bars
ax2 = axes[1]
group_labels_short = [SHORT_WRAP[THEME_ORDER[man]].replace("\n"," ")
                      for _, man in sorted(assignment, key=lambda x: x[1])]
pre_vals, rec_vals = [], []
for km, man in sorted(assignment, key=lambda x: x[1]):
    tp  = contingency[km, man]
    fp  = contingency[km, :].sum() - tp
    fn  = contingency[:, man].sum() - tp
    pre_vals.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
    rec_vals.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)

x     = np.arange(K)
width = 0.28
bars1 = ax2.bar(x - width, pre_vals, width, label="Precision", color="#2166ac", alpha=0.8)
bars2 = ax2.bar(x,          rec_vals, width, label="Recall",    color="#4dac26", alpha=0.8)
bars3 = ax2.bar(x + width,  f1_scores, width, label="F1",       color="#d6604d", alpha=0.8)

ax2.set_xticks(x)
ax2.set_xticklabels(group_labels_short, rotation=35, ha="right", fontsize=8)
ax2.set_ylim(0, 1.05)
ax2.set_ylabel("Score", fontsize=10)
ax2.set_title(f"B  Per-group precision / recall / F1\n"
              f"ARI = {ari:.3f}   NMI = {nmi:.3f}   Macro-F1 = {np.mean(f1_scores):.3f}",
              fontsize=10, fontweight="bold", loc="left")
ax2.axhline(np.mean(f1_scores), color="#d6604d", ls="--", lw=1.2,
            label=f"Macro-avg F1 = {np.mean(f1_scores):.3f}")
ax2.legend(fontsize=8, loc="lower left")
ax2.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(FIG_DIR, "kmeans_heatmap.pdf"), bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "kmeans_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Saved figures/cohesion/kmeans_heatmap.{pdf,png}")

print("\nDone.")
