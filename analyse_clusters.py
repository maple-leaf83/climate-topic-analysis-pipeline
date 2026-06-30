"""
analyse_clusters.py
───────────────────
Exploratory analysis of BERTopic cluster assignments.

Five output modes (all saved to figures/cluster_analysis/):

  1. overview   — era heatmap + publication stacked bar for ALL topics
  2. timeline   — per-topic article volume by year (all 68 or a subset)
  3. cluster N  — deep-dive on a single topic: timeline, pub breakdown, era×source
                  breakdown, and a sample article list
  4. group G    — same deep-dive for one of the 11 named thematic groups
  5. bubbles    — corpus-level packed bubble chart (all topics) + per-cluster
                  keyword bubble charts

Usage examples
──────────────
  python analyse_clusters.py                      # all figures
  python analyse_clusters.py --mode timeline      # timeline for all topics
  python analyse_clusters.py --mode cluster --id 0          # T00 deep-dive
  python analyse_clusters.py --mode cluster --id 9 42       # T09 + T42
  python analyse_clusters.py --mode group --name "Australian domestic politics"
  python analyse_clusters.py --mode group --name energy      # substring match

Requirements
────────────
  pip install pandas matplotlib numpy --break-system-packages
"""

import argparse
import ast
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO        = Path(__file__).parent
DATA_DIR    = REPO / "data" / "combined-no-letters"
FIGURES_DIR = REPO / "figures" / "cluster_analysis"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ASSIGNMENTS_CSV = DATA_DIR / "topic_assignments.csv"
SUMMARY_CSV     = DATA_DIR / "topic_summary.csv"
LETTERS_CSV     = REPO / "data" / "letters" / "topic_assignments.csv"

# ── Style ──────────────────────────────────────────────────────────────────────
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

ELECTIONS = [1996, 1998, 2001, 2004, 2007, 2010, 2013, 2016, 2019, 2022, 2025]

ERA_ORDER = ["Pre-Howard", "Howard", "Rudd/Gillard", "Abbott", "Turnbull/Morrison", "Albanese"]
ERA_COLORS = ["#aaaaaa", "#4393c3", "#74c476", "#fd8d3c", "#9e9ac8", "#f768a1"]
ERA_COLOR  = dict(zip(ERA_ORDER, ERA_COLORS))

# Canonical publication normalisation
def norm_pub(p: str) -> str:
    p = str(p)
    if "Guardian" in p:                                    return "Guardian"
    if "Sydney Morning Herald" in p or "Sun Herald" in p: return "SMH"
    if "Age, The" in p or "The Age" in p:                 return "The Age"
    if "Canberra Times" in p:                              return "Canberra Times"
    if "Australian" in p:                                  return "The Australian"
    return "Other"

PUB_ORDER  = ["Guardian", "The Age", "SMH", "Canberra Times", "The Australian", "Letters", "Other"]
PUB_COLORS = ["#2166ac", "#4dac26", "#d6604d", "#8073ac", "#e08214", "#e6ab02", "#aaaaaa"]
PUB_COLOR  = dict(zip(PUB_ORDER, PUB_COLORS))

# ── Thematic group assignments ─────────────────────────────────────────────────
# Populated at runtime from the topic_group column in topic_assignments.csv.
# Do not edit this dict — update topic_group_mapping.csv instead.
GROUPS: dict[str, list[int]] = {}
NOISE_GROUPS: set[str] = {"Noise"}


def build_groups(df: pd.DataFrame) -> dict[str, list[int]]:
    """
    Derive the group → [topic_id, …] mapping from the 'topic_group' column
    in the assignments dataframe. Noise rows are excluded.
    Raises ValueError if the column is absent (run_bertopic + update mapping first).
    """
    if "topic_group" not in df.columns:
        raise ValueError(
            "'topic_group' column not found in topic_assignments.csv.\n"
            "Run the topic-group mapping update script before plotting."
        )
    active = df[
        df["topic_group"].notna() & ~df["topic_group"].isin(NOISE_GROUPS)
    ]
    return {
        grp: sorted(active[active["topic_group"] == grp]["topic_id"].unique().tolist())
        for grp in active["topic_group"].unique()
    }

# ── Data loading ───────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(ASSIGNMENTS_CSV)
    df["pub"]      = df["publication"].apply(norm_pub)
    df["year"]     = pd.to_numeric(df["year"], errors="coerce")
    df["doc_type"] = "editorial"
    df = df[df["year"].between(1987, 2026)].copy()

    # Merge letters if the transform has been run
    if LETTERS_CSV.exists():
        letters = pd.read_csv(LETTERS_CSV)
        letters["pub"]      = letters["publication"].apply(norm_pub)
        letters["year"]     = pd.to_numeric(letters["year"], errors="coerce")
        letters["doc_type"] = "letter"
        letters = letters[letters["year"].between(1987, 2026)].copy()
        df = pd.concat([df, letters], ignore_index=True)
        # Treat all letters as a single "Letters" publication category
        df.loc[df["doc_type"] == "letter", "pub"] = "Letters"
        print(f"  Loaded {len(letters):,} letters (data/letters/topic_assignments.csv)")
    else:
        print("  [info] No letters assignments found — run with --transform-only to add them")

    # Normalise era labels
    era_map = {
        "Pre-Howard": "Pre-Howard",
        "Howard": "Howard",
        "Rudd": "Rudd/Gillard", "Gillard": "Rudd/Gillard", "Rudd/Gillard": "Rudd/Gillard",
        "Abbott": "Abbott",
        "Turnbull": "Turnbull/Morrison", "Morrison": "Turnbull/Morrison",
        "Turnbull/Morrison": "Turnbull/Morrison",
        "Albanese": "Albanese",
    }
    df["era_norm"] = df["era"].map(era_map).fillna(df["era"])

    summary = pd.read_csv(SUMMARY_CSV)
    summary["topic_id"] = summary["topic_id"].astype(int)
    return df, summary


def short_label(summary: pd.DataFrame, tid: int, max_words: int = 6) -> str:
    """Return a compact human-readable label: 'T07 — china energy emissions…'"""
    row = summary[summary["topic_id"] == tid]
    if row.empty:
        return f"T{tid:02d}"
    label = row.iloc[0]["label"]
    # label format: "7_china energy_emissions china_..."
    parts = label.split("_")[1:]          # drop numeric prefix
    words = " / ".join(parts[:3])
    return f"T{tid:02d} — {words}"


# ── Helper: election lines ─────────────────────────────────────────────────────
def add_elections(ax, year_range):
    for yr in ELECTIONS:
        if yr in year_range:
            ax.axvline(yr, color="grey", linewidth=0.5, linestyle=":", alpha=0.6)


# ── Helper: era bands + labels below x-axis ───────────────────────────────────
ERA_SPANS = [
    ("Pre-Howard",       1987, 1996),
    ("Howard",           1996, 2007),
    ("Rudd/Gillard",     2007, 2013),
    ("Abbott",           2013, 2015),
    ("Turnbull/Morrison",2015, 2022),
    ("Albanese",         2022, 2026),
]

def add_era_bands(ax, year_range):
    """Light background shading per era + era name as text just above the plot."""
    yr_min, yr_max = min(year_range), max(year_range)
    # Blended transform: x in data coords, y in axes fraction
    trans = ax.get_xaxis_transform()
    for i, (era, start, end) in enumerate(ERA_SPANS):
        vis_start = max(start, yr_min)
        vis_end   = min(end, yr_max)
        if vis_start >= vis_end:
            continue
        color = ERA_COLORS[i % len(ERA_COLORS)]
        ax.axvspan(vis_start, vis_end, alpha=0.06, color=color, linewidth=0)
        mid = (vis_start + vis_end) / 2
        ax.text(mid, 1.01, era, transform=trans,
                ha="center", va="bottom", fontsize=7, color="#444444",
                style="italic", clip_on=False)


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: OVERVIEW — era heatmap + pub stacked bar for all 67 topics
# ══════════════════════════════════════════════════════════════════════════════
def plot_overview(df: pd.DataFrame, summary: pd.DataFrame):
    print("Generating overview figures…")

    # ── 1a. Era heatmap ───────────────────────────────────────────────────────
    era_counts = (
        df.groupby(["topic_id", "era_norm"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ERA_ORDER, fill_value=0)
    )
    era_pct = era_counts.div(era_counts.sum(axis=1), axis=0) * 100

    # Sort topics by group order then article count within group
    ordered_tids = []
    for grp, tids in GROUPS.items():
        ordered_tids.extend(sorted(tids, key=lambda t: -df[df["topic_id"] == t].shape[0]))
    era_pct = era_pct.reindex(ordered_tids, fill_value=0)

    labels = [short_label(summary, t) for t in ordered_tids]

    fig, ax = plt.subplots(figsize=(12, 14))
    im = ax.imshow(era_pct.values.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_xticks(range(len(ordered_tids)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(ERA_ORDER)))
    ax.set_yticklabels(ERA_ORDER)
    ax.set_title("Figure — Topic × Era heatmap (% of topic's articles per era)", pad=12)
    plt.colorbar(im, ax=ax, label="% of topic articles", shrink=0.5)

    # Draw group separators
    pos = 0
    for grp, tids in GROUPS.items():
        ax.axvline(pos - 0.5, color="white", linewidth=1.5)
        ax.text(pos + len(tids) / 2 - 0.5, -1.8, grp, ha="center", va="top",
                fontsize=6, rotation=30, color="#333333", transform=ax.transData)
        pos += len(tids)

    fig.tight_layout()
    out = FIGURES_DIR / "overview_era_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")

    # ── 1b. Publication stacked bar for all topics ─────────────────────────────
    pub_counts = (
        df.groupby(["topic_id", "pub"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=PUB_ORDER, fill_value=0)
    )
    pub_pct = pub_counts.div(pub_counts.sum(axis=1), axis=0) * 100
    pub_pct = pub_pct.reindex(ordered_tids, fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    bottom = np.zeros(len(ordered_tids))
    x = np.arange(len(ordered_tids))
    for pub in PUB_ORDER:
        vals = pub_pct[pub].values
        ax.bar(x, vals, bottom=bottom, color=PUB_COLOR[pub], label=pub,
               edgecolor="none", width=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    ax.set_ylabel("% of topic articles")
    ax.set_title("Figure — Publication share per topic (all 68 clusters)")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 105)

    # Group separators
    pos = 0
    for grp, tids in GROUPS.items():
        ax.axvline(pos - 0.5, color="#cccccc", linewidth=1.0)
        pos += len(tids)

    fig.tight_layout()
    out = FIGURES_DIR / "overview_pub_share.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: TIMELINE — article volume by year for all (or selected) topics
# ══════════════════════════════════════════════════════════════════════════════
def plot_timeline_all(df: pd.DataFrame, summary: pd.DataFrame, topic_ids=None):
    """One figure per thematic group, each line = one topic within that group."""
    if topic_ids is None:
        groups_to_plot = GROUPS
    else:
        groups_to_plot = {"Selected topics": topic_ids}

    year_range = list(range(1987, 2027))
    cmap = plt.cm.tab20

    for grp_name, tids in groups_to_plot.items():
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = [cmap(i / max(len(tids), 1)) for i in range(len(tids))]

        for tid, color in zip(tids, colors):
            sub = df[df["topic_id"] == tid]
            if sub.empty:
                continue
            yearly = sub.groupby("year").size().reindex(year_range, fill_value=0)
            lbl = short_label(summary, tid)
            ax.plot(year_range, yearly.values, linewidth=1.5, color=color,
                    label=lbl, alpha=0.85)
            ax.fill_between(year_range, yearly.values, alpha=0.06, color=color)

        add_elections(ax, year_range)
        ax.set_xlabel("Year")
        ax.set_ylabel("Articles per year")
        ax.set_xlim(1987, 2026)
        ax.set_title(f"Timeline — {grp_name}")
        ax.legend(frameon=False, fontsize=7.5, loc="upper left",
                  bbox_to_anchor=(1.01, 1))
        fig.tight_layout()

        safe_name = grp_name.lower().replace(" ", "_").replace("&", "and").replace("/", "-")
        out = FIGURES_DIR / f"timeline_{safe_name}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 3 / 4: CLUSTER DEEP-DIVE (one or more topic IDs, or a group)
# ══════════════════════════════════════════════════════════════════════════════
def plot_cluster_deepdive(df: pd.DataFrame, summary: pd.DataFrame,
                          topic_ids: list[int], label: str):
    """Three separate figures for the given topic ID(s) treated as a unit:
       Panel A — cluster share of each publication's coverage over time
       Panel B — publication share (horizontal bar)
       Panel C — articles by era and source (grouped bar)
    """
    sub = df[df["topic_id"].isin(topic_ids)].copy()
    if sub.empty:
        print(f"  No articles found for topics {topic_ids}")
        return

    safe       = label.lower().replace(" ", "_").replace("&", "and").replace("/", "-")[:60]
    year_range = list(range(int(sub["year"].min()), 2027))
    title_sfx  = f"{label}  (n = {len(sub):,})"

    # ── Panel A: % of each publication's total corpus in this cluster ────────
    df_ed = df[df["doc_type"] == "editorial"] if "doc_type" in df.columns else df
    fig, ax = plt.subplots(figsize=(9, 4))
    for pub in [p for p in PUB_ORDER if p != "Letters"]:
        psub = (sub[(sub["pub"] == pub) & (sub["doc_type"] == "editorial")]
                if "doc_type" in sub.columns else sub[sub["pub"] == pub])
        if psub.empty:
            continue
        yearly_n = psub.groupby("year").size().reindex(year_range, fill_value=0)
        total_n  = df_ed[df_ed["pub"] == pub].groupby("year").size().reindex(year_range, fill_value=0)
        pct = yearly_n.div(total_n.replace(0, np.nan)) * 100
        ax.plot(year_range, pct.values, color=PUB_COLOR[pub],
                linewidth=1.6, label=pub, alpha=0.9)
        ax.fill_between(year_range, pct.fillna(0).values, alpha=0.07, color=PUB_COLOR[pub])
    add_elections(ax, year_range)
    add_era_bands(ax, year_range)
    ax.set_xlabel("Year")
    ax.set_ylabel("% of publication's annual articles")
    ax.set_title(f"Cluster share of each publication's coverage — {title_sfx}", pad=28)
    ax.set_xlim(min(year_range), 2026)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    out = FIGURES_DIR / f"deepdive_{safe}_A.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")

    # ── Panel B: publication share ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    pub_order_b = [p for p in PUB_ORDER if p != "Other"]
    pub_counts  = sub["pub"].value_counts().reindex(pub_order_b, fill_value=0)
    pub_pct     = pub_counts / pub_counts.sum() * 100
    nonzero     = pub_pct[pub_pct > 0]
    bars = ax.barh(nonzero.index[::-1], nonzero.values[::-1],
                   color=[PUB_COLOR[p] for p in nonzero.index[::-1]],
                   edgecolor="white")
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in nonzero.values[::-1]],
                 padding=4, fontsize=8)
    ax.set_xlabel("% of cluster articles")
    ax.set_title(f"Publication share — {title_sfx}")
    ax.set_xlim(0, nonzero.max() * 1.18 if not nonzero.empty else 1)
    fig.tight_layout()
    out = FIGURES_DIR / f"deepdive_{safe}_B.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")

    # ── Panel C: era × publication breakdown (article counts) ────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    era_pub = (
        sub.groupby(["era_norm", "pub"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=ERA_ORDER, columns=PUB_ORDER, fill_value=0)
    )
    era_pub     = era_pub.loc[:, era_pub.sum() > 0]
    active_pubs = list(era_pub.columns)

    n_eras      = len(ERA_ORDER)
    n_pubs      = len(active_pubs)
    group_width = 0.7
    bar_w       = group_width / max(n_pubs, 1)
    offsets     = np.linspace(-group_width / 2 + bar_w / 2,
                              group_width / 2 - bar_w / 2, n_pubs)
    x           = np.arange(n_eras)
    max_val     = 0
    for i, pub in enumerate(active_pubs):
        vals = era_pub[pub].values
        bars = ax.bar(x + offsets[i], vals, width=bar_w * 0.9,
                      color=PUB_COLOR.get(pub, "#aaaaaa"), label=pub,
                      edgecolor="white", linewidth=0.4)
        for rect, v in zip(bars, vals):
            if v > 0:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + max(vals.max() * 0.01, 1),
                        str(int(v)), ha="center", va="bottom",
                        fontsize=7, color="#333333")
        max_val = max(max_val, vals.max())
    ax.set_xticks(x)
    ax.set_xticklabels(ERA_ORDER, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("Number of articles")
    ax.set_title(f"Articles by era and source — {title_sfx}")
    ax.set_ylim(0, max_val * 1.18)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    out = FIGURES_DIR / f"deepdive_{safe}_C.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")

    # ── Bonus: print top-20 articles by year (most recent) ───────────────────
    print(f"\n  Sample articles (20 most recent):")
    sample = sub[["year", "pub", "title"]].sort_values("year", ascending=False).head(20)
    for _, row in sample.iterrows():
        print(f"    {int(row['year'])}  [{row['pub'][:15]:15s}]  {row['title'][:80]}")


# ══════════════════════════════════════════════════════════════════════════════
# MODE 5a: CORPUS BUBBLE CHART — all topics as packed circles
# ══════════════════════════════════════════════════════════════════════════════
GROUP_COLORS = {
    "Australian Politics & Policy":             "#e41a1c",
    "Nature, Ecosystems & Food Systems":        "#17becf",
    "Energy Transition & Technology":           "#984ea3",
    "US & UK Politics":                         "#f768a1",
    "Fossil Fuels, Divestment & Carbon Markets":"#ff7f00",
    "Climate Science":                          "#4daf4a",
    "International Negotiations & Geopolitics": "#377eb8",
    "Activism & Social Movements":              "#e6ab02",
}


def _pack_circles(radii: np.ndarray) -> np.ndarray:
    """
    Greedy circle packing. Place circles one at a time on an Archimedean spiral,
    accepting the first position with no overlaps. Returns (N, 2) array of centres.
    """
    n = len(radii)
    centres = np.zeros((n, 2))
    if n == 0:
        return centres
    # Place largest first at origin
    order = np.argsort(radii)[::-1]
    placed = []

    for idx in order:
        r = radii[idx]
        if not placed:
            centres[idx] = [0, 0]
            placed.append(idx)
            continue
        # Spiral outward until a gap is found
        step = 0.0
        found = False
        while not found:
            step += 0.05
            angle = step * 2.3          # golden-ratio-ish angle increment
            cx = step * np.cos(angle) * radii[order[0]]
            cy = step * np.sin(angle) * radii[order[0]]
            ok = True
            for j in placed:
                dist = np.hypot(cx - centres[j, 0], cy - centres[j, 1])
                if dist < r + radii[j] - 1e-6:
                    ok = False
                    break
            if ok:
                centres[idx] = [cx, cy]
                placed.append(idx)
                found = True
    return centres


def plot_corpus_bubbles(df: pd.DataFrame, summary: pd.DataFrame):
    """Packed bubble chart: one circle per display entry, sized by article count."""
    print("  Generating corpus bubble chart…")

    # Build display entries (one per topic)
    entries = []
    for grp, tids in GROUPS.items():
        for tid in tids:
            sub = df[df["topic_id"] == tid]
            n = len(sub)
            row = summary[summary["topic_id"] == tid]
            kws = ast.literal_eval(row.iloc[0]["keywords"]) if not row.empty else []
            kw = kws[0] if kws else f"T{tid:02d}"
            label = f"T{tid:02d}"
            entries.append({"group": grp, "tid": tid, "n": n,
                            "kw": kw, "label": label})

    ns      = np.array([e["n"] for e in entries], dtype=float)
    radii   = np.sqrt(ns / np.pi) * 0.55   # scale so largest ≈ readable
    centres = _pack_circles(radii)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_aspect("equal")
    ax.axis("off")

    for e, r, (cx, cy) in zip(entries, radii, centres):
        color = GROUP_COLORS.get(e["group"], "#cccccc")
        circle = plt.Circle((cx, cy), r, color=color, alpha=0.82, linewidth=0.5,
                            edgecolor="white")
        ax.add_patch(circle)
        # Label: short keyword + article count
        fs_kw = max(5.0, min(10.0, r * 1.1))
        fs_n  = max(4.5, min(8.5,  r * 0.85))
        ax.text(cx, cy + r * 0.18, e["kw"], ha="center", va="center",
                fontsize=fs_kw, fontweight="bold", color="white",
                wrap=False, clip_on=True)
        ax.text(cx, cy - r * 0.28, f"n={e['n']:,}", ha="center", va="center",
                fontsize=fs_n, color="white", alpha=0.9, clip_on=True)

    # Legend for groups
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, label=g, alpha=0.85)
               for g, c in GROUP_COLORS.items()]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, frameon=False, fontsize=8)

    ax.autoscale_view()
    ax.set_title("Corpus topic map — bubble size ∝ article count",
                 fontsize=13, pad=16)
    fig.tight_layout()
    out = FIGURES_DIR / "corpus_bubble_chart.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"    Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _get_kw_weights(summary: pd.DataFrame, topic_ids: list[int],
                    top_n: int = 30) -> list[tuple[str, float]]:
    """
    Aggregate keyword weights across multiple topics.
    Weight = inverse rank (rank-1 = 1.0, rank-10 = 0.1).
    Keywords appearing in multiple topics get their weights summed.
    Returns sorted list of (keyword, weight) pairs, capped at top_n.
    """
    kw_weights: dict[str, float] = {}
    for tid in topic_ids:
        row = summary[summary["topic_id"] == tid]
        if row.empty:
            continue
        kws = ast.literal_eval(row.iloc[0]["keywords"])
        for rank, kw in enumerate(kws):
            w = (len(kws) - rank) / len(kws)
            kw_weights[kw] = kw_weights.get(kw, 0) + w
    items = sorted(kw_weights.items(), key=lambda x: -x[1])[:top_n]
    return items


# ══════════════════════════════════════════════════════════════════════════════
# MODE 5c: WORDCLOUD keyword charts (requires: pip install wordcloud)
# ══════════════════════════════════════════════════════════════════════════════
def plot_keyword_wordcloud(summary: pd.DataFrame, topic_ids: list[int],
                           label: str, group_color: str = "#2166ac"):
    """
    Word cloud of top keywords for a cluster/group.
    Requires the `wordcloud` package (pip install wordcloud).
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("    [skip] wordcloud not installed — run: pip install wordcloud")
        return

    items = _get_kw_weights(summary, topic_ids, top_n=40)
    if not items:
        return

    # WordCloud expects a {word: frequency} dict
    freq = {w: float(v) for w, v in items}

    # Parse hex color to RGB for colormap
    base = tuple(int(group_color[i:i+2], 16) for i in (1, 3, 5))

    def _color_func(word, font_size, position, orientation,
                    random_state=None, **kwargs):
        # Vary lightness around the group colour
        r = int(np.clip(base[0] + random_state.randint(-30, 30), 0, 255))
        g = int(np.clip(base[1] + random_state.randint(-30, 30), 0, 255))
        b = int(np.clip(base[2] + random_state.randint(-30, 30), 0, 255))
        return f"rgb({r},{g},{b})"

    wc = WordCloud(
        width=1100, height=650,
        background_color="white",
        max_words=40,
        prefer_horizontal=0.85,
        color_func=_color_func,
        margin=6,
        font_step=1,
        min_font_size=9,
        max_font_size=90,
        collocations=False,
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Keywords: {label}", fontsize=12, pad=10, fontweight="bold")
    fig.tight_layout(pad=0.5)

    safe = label.lower().replace(" ", "_").replace("&", "and").replace("/", "-")[:55]
    out  = FIGURES_DIR / f"wordcloud_{safe}.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    plt.close()
    print(f"    Saved {out.name}")


def plot_all_keyword_wordclouds(summary: pd.DataFrame):
    """Word cloud for every thematic group (requires wordcloud package)."""
    print("  Generating keyword word clouds…")
    for grp_name, tids in GROUPS.items():
        color = GROUP_COLORS.get(grp_name, "#555555")
        plot_keyword_wordcloud(summary, tids, grp_name, group_color=color)


def plot_all_keyword_bubbles(summary: pd.DataFrame):
    """Convenience wrapper: generate word clouds for all groups."""
    plot_all_keyword_wordclouds(summary)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Analyse BERTopic cluster assignments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--mode", choices=["overview", "timeline", "cluster", "group", "bubbles", "all"],
                   default="all",
                   help="Analysis mode (default: all)")
    p.add_argument("--id", type=int, nargs="+", metavar="N",
                   help="Topic ID(s) for --mode cluster")
    p.add_argument("--name", type=str, metavar="NAME",
                   help="Group name (or substring) for --mode group")
    p.add_argument("--list-groups", action="store_true",
                   help="List available thematic groups and exit")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading data…")
    df, summary = load_data()
    print(f"  {len(df):,} articles, {df['topic_id'].nunique()} topics\n")

    # Build group mapping from the topic_group column (avoids hardcoding)
    GROUPS.update(build_groups(df))

    if args.list_groups:
        print("Available thematic groups:")
        for name, tids in GROUPS.items():
            print(f"  '{name}'  →  topics {tids}")
        sys.exit(0)

    mode = args.mode

    if mode in ("overview", "all"):
        plot_overview(df, summary)

    if mode in ("timeline", "all"):
        print("Generating per-group timeline figures…")
        plot_timeline_all(df, summary)

    if mode == "cluster":
        if not args.id:
            print("Error: --mode cluster requires --id N [N ...]")
            sys.exit(1)
        for tid in args.id:
            lbl = short_label(summary, tid)
            print(f"Deep-dive: {lbl}")
            plot_cluster_deepdive(df, summary, [tid], lbl)

    if mode == "group":
        if not args.name:
            print("Error: --mode group requires --name 'Group name'")
            sys.exit(1)
        matches = {k: v for k, v in GROUPS.items()
                   if args.name.lower() in k.lower()}
        if not matches:
            print(f"No group matching '{args.name}'. Use --list-groups to see options.")
            sys.exit(1)
        for grp_name, tids in matches.items():
            print(f"Deep-dive for group: {grp_name}")
            plot_cluster_deepdive(df, summary, tids, grp_name)

    if mode in ("bubbles", "all"):
        print("Generating bubble charts…")
        plot_corpus_bubbles(df, summary)
        plot_all_keyword_bubbles(summary)

    if mode == "all":
        print(f"\nGenerating deep-dives for all {len(GROUPS)} thematic groups…")
        for grp_name, tids in GROUPS.items():
            print(f"  {grp_name}")
            plot_cluster_deepdive(df, summary, tids, grp_name)

    print(f"\nDone. All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
