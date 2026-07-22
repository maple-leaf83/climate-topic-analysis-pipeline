"""
analyse_clusters.py
───────────────────
Generates all figures for the BERTopic cluster analysis.

Outputs (saved to figures/cluster_analysis/):
  - deepdive_<group>_A.pdf  — per-publication % share by year (Panel A)
  - deepdive_<group>_C.pdf  — article counts by era and publication (Panel C)
  - pub_share_matrix.pdf    — column-normalised outlet × group heatmap
  - pol_lead_keyword_cooccurrence.pdf — keyword co-occurrence within Political Leadership by era

Usage
─────
  python analyse_clusters.py

Requirements
────────────
  pip install pandas matplotlib numpy --break-system-packages
"""

import ast
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO        = Path(__file__).parent
DATA_DIR    = REPO / "data" / "australian-no-letters"
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

ERA_ORDER = ["Howard", "Rudd/Gillard", "Abbott", "Turnbull/Morrison", "Albanese"]
ERA_COLORS = ["#4393c3", "#74c476", "#fd8d3c", "#9e9ac8", "#f768a1"]
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

# Australian corpus only — Guardian and letters excluded from analysis
PUB_ORDER  = ["The Australian", "The Age", "SMH", "Canberra Times", "Other"]
# Seaborn "colorblind" palette (Wong 2011) — accessible to most colour-vision deficiencies
PUB_COLORS = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#949494"]
PUB_COLOR  = dict(zip(PUB_ORDER, PUB_COLORS))

# ── Thematic group assignments ─────────────────────────────────────────────────
# Populated at runtime from the 'theme' column in topic_assignments.csv.
# The column is added by the theme-mapping step in the analysis pipeline.
GROUPS: dict[str, list[int]] = {}
NOISE_GROUPS: set[str] = {"Noise", "UK-specific (Guardian only)", "Outlier (unassigned)"}

# Preferred display order for themes (used in heatmaps + bar charts)
THEME_ORDER = [
    "Political leadership & party dynamics",
    "Carbon pricing & emissions policy",
    "Climate science & physical impacts",
    "Energy policy & transition",
    "Environment & biodiversity",
    "Media, culture & society",
    "International climate diplomacy",
]

# Minimum articles a publication must have in a given year for its Panel A
# line to be drawn in colour.  Years below this threshold are drawn in grey.
MIN_PUB_YEAR_N: int = 15


def build_groups(df: pd.DataFrame) -> dict[str, list[int]]:
    """
    Derive the group → [topic_id, …] mapping from the 'theme' column
    in the assignments dataframe. Noise/outlier rows are excluded.
    Falls back to 'topic_group' column for backward compatibility.
    """
    col = "theme" if "theme" in df.columns else "topic_group"
    if col not in df.columns:
        raise ValueError(
            "Neither 'theme' nor 'topic_group' column found in topic_assignments.csv.\n"
            "Ensure the theme-mapping step has been run."
        )
    active = df[df[col].notna() & ~df[col].isin(NOISE_GROUPS)]
    groups = {
        grp: sorted(active[active[col] == grp]["topic_id"].unique().tolist())
        for grp in active[col].unique()
    }
    # Sort by THEME_ORDER where possible
    ordered = {k: groups[k] for k in THEME_ORDER if k in groups}
    ordered.update({k: v for k, v in groups.items() if k not in ordered})
    return ordered

# ── Data loading ───────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(ASSIGNMENTS_CSV)
    df["pub"]      = df["publication"].apply(norm_pub)
    df["year"]     = pd.to_numeric(df["year"], errors="coerce")
    df["doc_type"] = "editorial"
    df = df[df["year"].between(1987, 2026)].copy()


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
# CLUSTER DEEP-DIVE (Panel A + Panel C for each group)
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
        reliable = total_n >= MIN_PUB_YEAR_N
        # Full line in grey (unreliable baseline visible everywhere)
        ax.plot(year_range, pct.values, color="#bbbbbb", linewidth=1.0, alpha=0.7)
        # Overplot reliable segments in publication colour
        pct_reliable = pct.where(reliable)
        ax.plot(year_range, pct_reliable.values, color=PUB_COLOR[pub],
                linewidth=1.6, label=pub, alpha=0.9)
    add_elections(ax, year_range)
    add_era_bands(ax, year_range)
    ax.set_xlabel("Year")
    ax.set_ylabel("% of publication's annual articles")
    ax.set_title(title_sfx, pad=28)
    ax.set_xlim(1996, 2026)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    out = FIGURES_DIR / f"deepdive_{safe}_A.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")

    # # ── Panel B: publication share ────────────────────────────────────────────
    # fig, ax = plt.subplots(figsize=(6, 4))
    # pub_order_b = [p for p in PUB_ORDER if p != "Other"]
    # pub_counts  = sub["pub"].value_counts().reindex(pub_order_b, fill_value=0)
    # pub_pct     = pub_counts / pub_counts.sum() * 100
    # nonzero     = pub_pct[pub_pct > 0]
    # bars = ax.barh(nonzero.index[::-1], nonzero.values[::-1],
    #                color=[PUB_COLOR[p] for p in nonzero.index[::-1]],
    #                edgecolor="white")
    # ax.bar_label(bars, labels=[f"{v:.1f}%" for v in nonzero.values[::-1]],
    #              padding=4, fontsize=8)
    # ax.set_xlabel("% of cluster articles")
    # ax.set_title(f"Publication share — {title_sfx}")
    # ax.set_xlim(0, nonzero.max() * 1.18 if not nonzero.empty else 1)
    # fig.tight_layout()
    # out = FIGURES_DIR / f"deepdive_{safe}_B.pdf"
    # fig.savefig(out, bbox_inches="tight")
    # plt.close()
    # print(f"  Saved {out.name}")

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
    ax.set_title(title_sfx)
    ax.set_ylim(0, max_val * 1.18)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    out = FIGURES_DIR / f"deepdive_{safe}_C.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")

    # ── Panel D: era × publication — % of each outlet's era total ───────────
    # Denominator: total editorial articles per outlet per era (not just this group)
    df_ed_era = df[df["doc_type"] == "editorial"] if "doc_type" in df.columns else df
    era_total = (
        df_ed_era.groupby(["era_norm", "pub"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=ERA_ORDER, columns=PUB_ORDER, fill_value=0)
    )
    # Only keep pubs that appear in this group
    era_total_grp = era_total[active_pubs]
    # Column-normalised %: group articles / total outlet×era articles
    era_pct = era_pub.div(era_total_grp.replace(0, np.nan)) * 100

    fig, ax = plt.subplots(figsize=(9, 4))
    max_val_d = 0
    for i, pub in enumerate(active_pubs):
        vals = era_pct[pub].values
        bars = ax.bar(x + offsets[i], vals, width=bar_w * 0.9,
                      color=PUB_COLOR.get(pub, "#aaaaaa"), label=pub,
                      edgecolor="white", linewidth=0.4)
        for rect, v in zip(bars, vals):
            if np.isnan(v) or v < 0.5:
                continue
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + max(np.nanmax(vals) * 0.01, 0.3),
                    f"{v:.0f}%", ha="center", va="bottom",
                    fontsize=7, color="#333333")
        max_val_d = max(max_val_d, np.nanmax(vals) if not np.all(np.isnan(vals)) else 0)
    ax.set_xticks(x)
    ax.set_xticklabels(ERA_ORDER, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("% of outlet's articles in era")
    ax.set_title(title_sfx)
    ax.set_ylim(0, max_val_d * 1.18)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    out = FIGURES_DIR / f"deepdive_{safe}_D.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out.name}")

    # ── Bonus: print top-20 articles by year (most recent) ───────────────────
    print(f"\n  Sample articles (20 most recent):")
    sample = sub[["year", "pub", "title"]].sort_values("year", ascending=False).head(20)
    for _, row in sample.iterrows():
        print(f"    {int(row['year'])}  [{row['pub'][:15]:15s}]  {row['title'][:80]}")


# # ══════════════════════════════════════════════════════════════════════════════
# # MODE 5a: CORPUS BUBBLE CHART — all topics as packed circles
# # ══════════════════════════════════════════════════════════════════════════════
# GROUP_COLORS = {
#     "Australian Politics & Policy":             "#e41a1c",
#     "Nature, Ecosystems & Food Systems":        "#17becf",
#     "Energy Transition & Technology":           "#984ea3",
#     "US & UK Politics":                         "#f768a1",
#     "Fossil Fuels, Divestment & Carbon Markets":"#ff7f00",
#     "Climate Science":                          "#4daf4a",
#     "International Negotiations & Geopolitics": "#377eb8",
#     "Activism & Social Movements":              "#e6ab02",
# }


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


# def plot_corpus_bubbles(df: pd.DataFrame, summary: pd.DataFrame):
#     """Packed bubble chart: one circle per display entry, sized by article count."""
#     print("  Generating corpus bubble chart…")
#
#     # Build display entries (one per topic)
#     entries = []
#     for grp, tids in GROUPS.items():
#         for tid in tids:
#             sub = df[df["topic_id"] == tid]
#             n = len(sub)
#             row = summary[summary["topic_id"] == tid]
#             kws = ast.literal_eval(row.iloc[0]["keywords"]) if not row.empty else []
#             kw = kws[0] if kws else f"T{tid:02d}"
#             label = f"T{tid:02d}"
#             entries.append({"group": grp, "tid": tid, "n": n,
#                             "kw": kw, "label": label})
#
#     ns      = np.array([e["n"] for e in entries], dtype=float)
#     radii   = np.sqrt(ns / np.pi) * 0.55   # scale so largest ≈ readable
#     centres = _pack_circles(radii)
#
#     fig, ax = plt.subplots(figsize=(14, 14))
#     ax.set_aspect("equal")
#     ax.axis("off")
#
#     for e, r, (cx, cy) in zip(entries, radii, centres):
#         color = GROUP_COLORS.get(e["group"], "#cccccc")
#         circle = plt.Circle((cx, cy), r, color=color, alpha=0.82, linewidth=0.5,
#                             edgecolor="white")
#         ax.add_patch(circle)
#         # Label: short keyword + article count
#         fs_kw = max(5.0, min(10.0, r * 1.1))
#         fs_n  = max(4.5, min(8.5,  r * 0.85))
#         ax.text(cx, cy + r * 0.18, e["kw"], ha="center", va="center",
#                 fontsize=fs_kw, fontweight="bold", color="white",
#                 wrap=False, clip_on=True)
#         ax.text(cx, cy - r * 0.28, f"n={e['n']:,}", ha="center", va="center",
#                 fontsize=fs_n, color="white", alpha=0.9, clip_on=True)
#
#     # Legend for groups
#     from matplotlib.patches import Patch
#     handles = [Patch(facecolor=c, label=g, alpha=0.85)
#                for g, c in GROUP_COLORS.items()]
#     ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
#               ncol=3, frameon=False, fontsize=8)
#
#     ax.autoscale_view()
#     ax.set_title("Corpus topic map — bubble size ∝ article count",
#                  fontsize=13, pad=16)
#     fig.tight_layout()
#     out = FIGURES_DIR / "corpus_bubble_chart.pdf"
#     fig.savefig(out, bbox_inches="tight")
#     plt.close()
#     print(f"    Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLICATION SHARE MATRIX
# ══════════════════════════════════════════════════════════════════════════════
def plot_pub_share_matrix(df: pd.DataFrame):
    """
    Heatmap: rows = thematic groups, columns = publications.
    Each cell = % of that publication's articles in the group
    (column-normalised by each outlet's total article count).
    """
    print("  Generating publication share matrix…")
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    pubs = [p for p in PUB_ORDER if p not in ("Other", "Letters")]

    # Total articles per publication (denominator for column normalisation)
    # Editorial articles only — letters excluded
    ed_df = df[df["doc_type"] == "editorial"] if "doc_type" in df.columns else df
    pub_totals = ed_df["pub"].value_counts().reindex(pubs, fill_value=0)

    # Build group × publication table (column-normalised %, editorial only)
    rows = []
    for grp, tids in GROUPS.items():
        sub = ed_df[ed_df["topic_id"].isin(tids)]
        counts = sub["pub"].value_counts().reindex(pubs, fill_value=0)
        pct = counts / pub_totals.replace(0, np.nan) * 100
        rows.append({"group": grp, **pct.to_dict()})

    matrix = pd.DataFrame(rows).set_index("group")[pubs]

    n_groups = len(matrix)
    n_pubs   = len(pubs)

    fig, ax = plt.subplots(figsize=(max(7, n_pubs * 1.4), max(5, n_groups * 0.65)))

    im = ax.imshow(matrix.values, cmap="Blues", aspect="auto", vmin=0)

    # Annotate cells
    vmax = np.nanmax(matrix.values)
    for i in range(n_groups):
        for j in range(n_pubs):
            val = matrix.values[i, j]
            if np.isnan(val) or val < 0.5:
                continue
            text_color = "white" if val > vmax * 0.65 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color=text_color)

    ax.set_xticks(range(n_pubs))
    ax.set_xticklabels(pubs, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(matrix.index, fontsize=9)

    ax.set_title("% of each publication's articles per thematic group",
                 fontsize=11, pad=10)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.2)
    fig.colorbar(im, cax=cax, label="% of publication's articles")

    fig.tight_layout()
    out = FIGURES_DIR / "pub_share_matrix.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"    Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# POLITICAL LEADERSHIP KEYWORD CO-OCCURRENCE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def plot_t00_keyword_cooccurrence(df: pd.DataFrame, summary: pd.DataFrame):
    """
    Within all articles in the Political leadership & party dynamics group, measure
    how often keywords from each other thematic group appear in article bodies.

    For each Political Leadership article, scores binary presence (1/0) of any
    keyword from each other group, then aggregates by political era.  Shows which
    themes co-occur most with political leadership coverage, and how that has
    shifted over time.
    """
    import re
    import csv
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    print("  Loading article bodies for Political Leadership keyword co-occurrence…")

    # ── Load body text from articles_scored.csv ───────────────────────────────
    SCORED_CSV = REPO / "data" / "articles_scored.csv"
    csv.field_size_limit(10_000_000)
    bodies = pd.read_csv(
        SCORED_CSV,
        usecols=["title", "year", "body"],
        dtype={"year": "Int64"},
    )

    # ── Restrict to Political Leadership articles, join body text ─────────────
    pol_lead_tids = GROUPS.get("Political leadership & party dynamics", [])
    t00 = df[df["topic_id"].isin(pol_lead_tids)].copy()
    t00["year"] = t00["year"].astype("Int64")
    t00 = t00.merge(bodies, on=["title", "year"], how="left")
    t00 = t00.dropna(subset=["body"])
    print(f"  Political Leadership articles with body text: {len(t00):,} / {len(df[df['topic_id'].isin(pol_lead_tids)]):,}")

    # ── Build keyword pattern per group ──────────────────────────────────────
    # Exclude Political Leadership (circular) and noise/UK-specific groups
    skip = NOISE_GROUPS | {"Political leadership & party dynamics"}

    group_patterns: dict[str, re.Pattern] = {}
    for grp, tids in GROUPS.items():
        if grp in skip:
            continue
        kws: list[str] = []
        for tid in tids:
            row = summary[summary["topic_id"] == tid]
            if row.empty:
                continue
            kws.extend(ast.literal_eval(row.iloc[0]["keywords"])[:5])  # top 5 per topic
        if not kws:
            continue
        # Deduplicate preserving order
        seen: set[str] = set()
        unique = [k for k in kws if not (k in seen or seen.add(k))]  # type: ignore[func-returns-value]
        # Compile a single alternation pattern — much faster than looping
        pattern = re.compile(
            r'\b(?:' + '|'.join(re.escape(k.lower()) for k in unique) + r')\b'
        )
        group_patterns[grp] = pattern

    # ── Score each article against each group's keywords ─────────────────────
    print(f"  Scoring {len(t00):,} Political Leadership articles against {len(group_patterns)} groups…")
    bodies_lower = t00["body"].str.lower().fillna("")

    for grp, pattern in group_patterns.items():
        t00[f"grp_{grp}"] = bodies_lower.apply(
            lambda b, p=pattern: int(bool(p.search(b)))
        )

    # ── Aggregate by era ──────────────────────────────────────────────────────
    grp_cols = [f"grp_{g}" for g in group_patterns]
    era_rates = (
        t00.groupby("era_norm")[grp_cols].mean() * 100
    ).reindex(ERA_ORDER).dropna(how="all")
    era_rates.columns = list(group_patterns.keys())

    # Reorder groups to match pub_share_matrix row order (GROUPS key order)
    grp_order = [g for g in GROUPS if g not in skip]
    era_rates = era_rates.reindex(columns=grp_order)

    # Transpose: rows = groups, columns = eras  (mirrors pub_share_matrix layout)
    cooc = era_rates.T  # shape: n_grps × n_eras

    # ── Plot heatmap ──────────────────────────────────────────────────────────
    n_grps = len(cooc)
    n_eras = len(cooc.columns)

    fig, ax = plt.subplots(figsize=(max(6, n_eras * 1.4), max(5, n_grps * 0.65)))

    im = ax.imshow(cooc.values, cmap="YlOrRd", aspect="auto", vmin=0)

    vmax = float(np.nanmax(cooc.values))
    for i in range(n_grps):
        for j in range(n_eras):
            val = cooc.values[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val > vmax * 0.72 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=8.5, color=text_color)

    ax.set_xticks(range(n_eras))
    ax.set_xticklabels(cooc.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_grps))
    ax.set_yticklabels(cooc.index, fontsize=9)
    ax.set_title(
        "Keyword co-occurrence by era within the Political Leadership & Party Dynamics group\n",
        fontsize=11, pad=10,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.2)
    fig.colorbar(im, cax=cax, label="% of Political Leadership articles")

    fig.tight_layout()
    out = FIGURES_DIR / "pol_lead_keyword_cooccurrence.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"    Saved {out.name}")

    # ── Print summary ─────────────────────────────────────────────────────────
    overall = cooc.mean(axis=1)  # mean across eras
    print("\n  Overall keyword co-occurrence rates (mean across eras, % of Political Leadership articles):")
    for grp in overall.sort_values(ascending=False).index:
        print(f"    {grp:50s}  {overall[grp]:.1f}%")


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
# def plot_keyword_wordcloud(summary: pd.DataFrame, topic_ids: list[int],
#                            label: str, group_color: str = "#2166ac"):
#     """
#     Word cloud of top keywords for a cluster/group.
#     Requires the `wordcloud` package (pip install wordcloud).
#     """
#     try:
#         from wordcloud import WordCloud
#     except ImportError:
#         print("    [skip] wordcloud not installed — run: pip install wordcloud")
#         return
#
#     items = _get_kw_weights(summary, topic_ids, top_n=40)
#     if not items:
#         return
#
#     # WordCloud expects a {word: frequency} dict
#     freq = {w: float(v) for w, v in items}
#
#     # Parse hex color to RGB for colormap
#     base = tuple(int(group_color[i:i+2], 16) for i in (1, 3, 5))
#
#     def _color_func(word, font_size, position, orientation,
#                     random_state=None, **kwargs):
#         # Vary lightness around the group colour
#         r = int(np.clip(base[0] + random_state.randint(-30, 30), 0, 255))
#         g = int(np.clip(base[1] + random_state.randint(-30, 30), 0, 255))
#         b = int(np.clip(base[2] + random_state.randint(-30, 30), 0, 255))
#         return f"rgb({r},{g},{b})"
#
#     wc = WordCloud(
#         width=1100, height=650,
#         background_color="white",
#         max_words=40,
#         prefer_horizontal=0.85,
#         color_func=_color_func,
#         margin=6,
#         font_step=1,
#         min_font_size=9,
#         max_font_size=90,
#         collocations=False,
#     ).generate_from_frequencies(freq)
#
#     fig, ax = plt.subplots(figsize=(11, 6.5))
#     ax.imshow(wc, interpolation="bilinear")
#     ax.axis("off")
#     ax.set_title(f"Keywords: {label}", fontsize=12, pad=10, fontweight="bold")
#     fig.tight_layout(pad=0.5)
#
#     safe = label.lower().replace(" ", "_").replace("&", "and").replace("/", "-")[:55]
#     out  = FIGURES_DIR / f"wordcloud_{safe}.pdf"
#     fig.savefig(out, bbox_inches="tight", dpi=180)
#     plt.close()
#     print(f"    Saved {out.name}")


# def plot_all_keyword_wordclouds(summary: pd.DataFrame):
#     """Word cloud for every thematic group (requires wordcloud package)."""
#     print("  Generating keyword word clouds…")
#     for grp_name, tids in GROUPS.items():
#         color = GROUP_COLORS.get(grp_name, "#555555")
#         plot_keyword_wordcloud(summary, tids, grp_name, group_color=color)
#
#
# def plot_all_keyword_bubbles(summary: pd.DataFrame):
#     """Convenience wrapper: generate word clouds for all groups."""
#     plot_all_keyword_wordclouds(summary)


# ══════════════════════════════════════════════════════════════════════════════
# PUB SHARE BY ERA  — column-normalised heatmap, one panel per era
# ══════════════════════════════════════════════════════════════════════════════
def plot_pub_share_by_era(df: pd.DataFrame):
    """
    For each of the five government eras, produce a column-normalised heatmap:
      rows  = thematic groups (in THEME_ORDER)
      cols  = publications
      cells = % of that outlet's articles in that era falling in the group

    All five eras are laid out as a single figure with shared row/col labels.
    """
    print("  Generating pub share by era (column-normalised)…")
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    pubs     = [p for p in PUB_ORDER if p not in ("Letters", "Other")]
    themes   = [g for g in THEME_ORDER if g in GROUPS]
    eras     = [e for e in ERA_ORDER if e in df["era_norm"].unique()]
    n_eras   = len(eras)
    n_themes = len(themes)
    n_pubs   = len(pubs)

    fig, axes = plt.subplots(
        1, n_eras,
        figsize=(n_eras * (n_pubs * 1.1 + 0.6), n_themes * 0.65 + 1.8),
        sharey=True,
    )
    if n_eras == 1:
        axes = [axes]

    vmax_global = 0.0
    matrices = {}
    for era in eras:
        era_df = df[(df["era_norm"] == era) & (df["doc_type"] == "editorial")]
        pub_totals = era_df["pub"].value_counts().reindex(pubs, fill_value=0)
        rows = []
        for grp in themes:
            tids  = GROUPS.get(grp, [])
            sub   = era_df[era_df["topic_id"].isin(tids)]
            cnts  = sub["pub"].value_counts().reindex(pubs, fill_value=0)
            pct   = cnts / pub_totals.replace(0, np.nan) * 100
            rows.append(pct.values)
        mat = np.array(rows, dtype=float)
        matrices[era] = mat
        vmax_global = max(vmax_global, float(np.nanmax(mat)) if mat.size else 0)

    for ax, era in zip(axes, eras):
        mat = matrices[era]
        im  = ax.imshow(mat, cmap="Blues", aspect="auto",
                        vmin=0, vmax=vmax_global)
        ax.set_xticks(range(n_pubs))
        ax.set_xticklabels(pubs, rotation=35, ha="right", fontsize=8)
        ax.set_title(era, fontsize=9, fontweight="bold", pad=6)

        # Annotate cells
        for i in range(n_themes):
            for j in range(n_pubs):
                val = mat[i, j]
                if np.isnan(val) or val < 1.0:
                    continue
                text_color = "white" if val > vmax_global * 0.65 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7.5, color=text_color)

    # y-axis labels on the leftmost panel only
    axes[0].set_yticks(range(n_themes))
    axes[0].set_yticklabels(themes, fontsize=8.5)

    fig.suptitle(
        "% of each publication's articles per theme, by government era\n"
        "(column-normalised within era)",
        fontsize=11, y=1.01,
    )

    # Shared colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(
            cmap="Blues",
            norm=plt.Normalize(vmin=0, vmax=vmax_global),
        ),
        ax=axes, shrink=0.6, pad=0.02, label="% of outlet's articles",
    )

    fig.tight_layout()
    out = FIGURES_DIR / "pub_share_by_era.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"    Saved {out.name}")

    # ── Also print the numbers ────────────────────────────────────────────────
    print("\n  Column-normalised % by era:")
    for era in eras:
        pub_totals = df[(df["era_norm"] == era) & (df["doc_type"] == "editorial")]["pub"]\
            .value_counts().reindex(pubs, fill_value=0)
        print(f"\n  {era}  (totals: {dict(zip(pubs, pub_totals.values))})")
        for i, grp in enumerate(themes):
            row_str = "  ".join(
                f"{pubs[j]}:{matrices[era][i,j]:.0f}%" for j in range(n_pubs)
                if not np.isnan(matrices[era][i, j])
            )
            print(f"    {grp[:45]:45s}  {row_str}")


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED PANEL D — all themes on one page, % of outlet era total
# ══════════════════════════════════════════════════════════════════════════════
def plot_all_panel_d(df: pd.DataFrame):
    """
    Single-page figure: 7 panels (one per thematic group) showing the
    percentage of each publication's total articles in each era that fall
    within the group.  Layout: 4-up top row, 3-up bottom row (slot 8 empty).
    Single shared legend sits below both rows via ultraplot fig.legend(loc='b').
    """
    import ultraplot as uplt

    print("  Generating combined Panel D (% era share, all groups)…")

    df_ed = df[df["doc_type"] == "editorial"] if "doc_type" in df.columns else df
    pubs  = [p for p in PUB_ORDER if p not in ("Other", "Letters")]

    # Denominator: total editorial articles per outlet per era
    era_total = (
        df_ed.groupby(["era_norm", "pub"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=ERA_ORDER, columns=pubs, fill_value=0)
    )

    # Build per-group percentage tables
    theme_names = [g for g in THEME_ORDER if g in GROUPS]
    group_data: dict[str, pd.DataFrame] = {}
    for grp in theme_names:
        tids = GROUPS[grp]
        sub  = df_ed[df_ed["topic_id"].isin(tids)]
        raw  = (
            sub.groupby(["era_norm", "pub"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=ERA_ORDER, columns=pubs, fill_value=0)
        )
        group_data[grp] = raw.div(era_total.replace(0, np.nan)) * 100

    # Short era labels to avoid crowding
    era_labels = ["Howard", "Rudd/\nGillard", "Abbott", "Turnb./\nMorrison", "Albanese"]

    # Short theme labels for panel titles
    short_titles = {
        "Political leadership & party dynamics": "Political leadership",
        "Carbon pricing & emissions policy":     "Carbon pricing",
        "Climate science & physical impacts":    "Climate science",
        "Energy policy & transition":            "Energy policy",
        "Environment & biodiversity":            "Environment & biodiversity",
        "Media, culture & society":              "Media, culture & society",
        "International climate diplomacy":       "International diplomacy",
    }

    # ── ultraplot layout: 4 rows × 2 cols, slot 8 empty (0) ─────────────────
    array = [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 0],   # 0 = hidden empty cell
    ]
    fig, axs = uplt.subplots(
        array=array,
        figsize=(11, 18),
        hspace=2.4,
        wspace=2.4,
        sharey=False,   # each group has its own y-scale
    )

    n_eras  = len(ERA_ORDER)
    n_pubs  = len(pubs)
    group_w = 0.72
    bar_w   = group_w / n_pubs
    offsets = np.linspace(-group_w / 2 + bar_w / 2,
                           group_w / 2 - bar_w / 2, n_pubs)
    x       = np.arange(n_eras)

    # Left column panels get the ylabel
    LEFT_COL = {0, 2, 4, 6}

    legend_handles: list = []
    legend_labels:  list = []

    for idx, (ax, grp) in enumerate(zip(axs, theme_names)):
        data    = group_data[grp]
        max_val = 0

        for i, pub in enumerate(pubs):
            vals = data[pub].values
            bars = ax.bar(
                x + offsets[i], vals,
                width=bar_w * 0.88,
                color=PUB_COLOR.get(pub, "#aaaaaa"),
                edgecolor="white",
                linewidth=0.3,
                label=pub,
            )
            if idx == 0:
                legend_handles.append(bars[0])
                legend_labels.append(pub)

            vmax_pub = float(np.nanmax(vals)) if not np.all(np.isnan(vals)) else 0
            max_val  = max(max_val, vmax_pub)

            for rect, v in zip(bars, vals):
                if np.isnan(v) or v < 1.5:
                    continue
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + max(max_val * 0.015, 0.4),
                    f"{v:.0f}",
                    ha="center", va="bottom",
                    fontsize=8, color="#333333",
                )

        ax.format(
            title=short_titles.get(grp, grp),
            titleloc="c",
            titlesize=12,
            titleweight="bold",
            ylabel="% of outlet's articles in era" if idx in LEFT_COL else "",
            ylabelsize=11,
            xticks=x,
            xticklabels=era_labels,
            xticklabelsize=10,
            yticklabelsize=10,
            ylim=(0, max(max_val * 1.22, 1)),
            grid=False,
            abc='A', abcloc='ul'
        )

    # ── Shared legend centred below the figure ────────────────────────────────
    fig.legend(
        legend_handles, legend_labels,
        loc="b",
        ncols=n_pubs,
        frame=False,
        fontsize=12,
        handlelength=1.6,
        handletextpad=0.6,
        columnspacing=1.5,
    )

    fig.format(
        suptitle="% of each outlet's articles per era devoted to each thematic group",
        suptitlesize=13,
    )

    out = FIGURES_DIR / "panel_d_all_themes.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"    Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED PANEL D (6-theme variant) — 3×2, excludes International diplomacy
# ══════════════════════════════════════════════════════════════════════════════
def plot_panel_d_6themes(df: pd.DataFrame):
    """
    2×3 version of plot_all_panel_d, excluding 'International climate diplomacy'
    (smallest group).  Same ultraplot settings as the 4×2 version.
    """
    import ultraplot as uplt

    print("  Generating 6-theme Panel D (3×2, no international diplomacy)…")

    EXCLUDE = {"International climate diplomacy"}

    df_ed = df[df["doc_type"] == "editorial"] if "doc_type" in df.columns else df
    pubs  = [p for p in PUB_ORDER if p not in ("Other", "Letters")]

    # Denominator: total editorial articles per outlet per era
    era_total = (
        df_ed.groupby(["era_norm", "pub"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=ERA_ORDER, columns=pubs, fill_value=0)
    )

    # Build per-group percentage tables (6 themes only)
    theme_names = [g for g in THEME_ORDER if g in GROUPS and g not in EXCLUDE]
    group_data: dict[str, pd.DataFrame] = {}
    for grp in theme_names:
        tids = GROUPS[grp]
        sub  = df_ed[df_ed["topic_id"].isin(tids)]
        raw  = (
            sub.groupby(["era_norm", "pub"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=ERA_ORDER, columns=pubs, fill_value=0)
        )
        group_data[grp] = raw.div(era_total.replace(0, np.nan)) * 100

    era_labels = ["Howard", "Rudd/\nGillard", "Abbott", "Turnb./\nMorrison", "Albanese"]

    short_titles = {
        "Political leadership & party dynamics": "Political leadership",
        "Carbon pricing & emissions policy":     "Carbon pricing",
        "Climate science & physical impacts":    "Climate science",
        "Energy policy & transition":            "Energy policy",
        "Environment & biodiversity":            "Environment & biodiversity",
        "Media, culture & society":              "Media, culture & society",
    }

    # ── 2×3 layout — all 6 cells filled, no empty slot ───────────────────────
    array = [
        [1, 2, 3],
        [4, 5, 6],
    ]
    fig, axs = uplt.subplots(
        array=array,
        figsize=(16, 9),
        hspace=2.4,
        wspace=2.4,
        sharey=False,
    )

    n_eras  = len(ERA_ORDER)
    n_pubs  = len(pubs)
    group_w = 0.72
    bar_w   = group_w / n_pubs
    offsets = np.linspace(-group_w / 2 + bar_w / 2,
                           group_w / 2 - bar_w / 2, n_pubs)
    x       = np.arange(n_eras)

    LEFT_COL = {0, 3}   # left-column indices in a 2×3 grid

    legend_handles: list = []
    legend_labels:  list = []

    for idx, (ax, grp) in enumerate(zip(axs, theme_names)):
        data    = group_data[grp]
        max_val = 0

        for i, pub in enumerate(pubs):
            vals = data[pub].values
            bars = ax.bar(
                x + offsets[i], vals,
                width=bar_w * 0.88,
                color=PUB_COLOR.get(pub, "#aaaaaa"),
                edgecolor="white",
                linewidth=0.3,
                label=pub,
            )
            if idx == 0:
                legend_handles.append(bars[0])
                legend_labels.append(pub)

            vmax_pub = float(np.nanmax(vals)) if not np.all(np.isnan(vals)) else 0
            max_val  = max(max_val, vmax_pub)

            for rect, v in zip(bars, vals):
                if np.isnan(v) or v < 1.5:
                    continue
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + max(max_val * 0.015, 0.4),
                    f"{v:.0f}",
                    ha="center", va="bottom",
                    fontsize=8, color="#333333",
                )

        ax.format(
            title=short_titles.get(grp, grp),
            titleloc="c",
            titlesize=12,
            titleweight="bold",
            ylabel="% of outlet's articles in era" if idx in LEFT_COL else "",
            ylabelsize=11,
            xticks=x,
            xticklabels=era_labels,
            xticklabelsize=10,
            yticklabelsize=10,
            ylim=(0, max(max_val * 1.22, 1)),
            grid=False,
            abc="A", abcloc="ul",
        )

    # ── Shared legend centred below the figure ────────────────────────────────
    fig.legend(
        legend_handles, legend_labels,
        loc="b",
        ncols=n_pubs,
        frame=False,
        fontsize=12,
        handlelength=1.6,
        handletextpad=0.6,
        columnspacing=1.5,
    )

    fig.format(
        suptitle="% of each outlet's articles per era devoted to each thematic group",
        suptitlesize=13,
    )

    out = FIGURES_DIR / "panel_d_6themes.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"    Saved {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("Loading data…")
    df, summary = load_data()
    print(f"  {len(df):,} articles, {df['topic_id'].nunique()} topics\n")

    # Ensure doc_type column exists (australian-no-letters has no letters rows)
    if "doc_type" not in df.columns:
        df["doc_type"] = "editorial"

    GROUPS.update(build_groups(df))
    print(f"  Themes loaded: {list(GROUPS.keys())}\n")

    print("Generating publication share matrix (column-normalised)…")
    plot_pub_share_matrix(df)

    print("Generating pub share by era…")
    plot_pub_share_by_era(df)

    print("Generating Political Leadership keyword co-occurrence analysis…")
    plot_t00_keyword_cooccurrence(df, summary)

    # print(f"\nGenerating deep-dives for all {len(GROUPS)} thematic groups…")
    # for grp_name, tids in GROUPS.items():
    #     print(f"  Deep-dive: {grp_name}")
    #     plot_cluster_deepdive(df, summary, tids, grp_name)

    # print("\nGenerating combined Panel D (all themes, % era share)…")
    # plot_all_panel_d(df)

    print("\nGenerating combined Panel D (6-theme 3×2 variant)…")
    plot_panel_d_6themes(df)

    print(f"\nDone. All figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
