"""
report_topics.py
Combines Guardian and Australian topic assignments into a unified report:
  - data/topic_combined.csv          : all articles, unified topic labels
  - data/topic_alignment.csv         : cross-corpus Jaccard alignment
  - figures/fig4_topic_table.pdf     : 4-column topic summary table
  - figures/fig5_top10_by_pub.pdf    : horizontal bar chart, top-10 topics by publication

Usage:
    python report_topics.py
"""

import os, ast, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO = Path(__file__).parent
DATA  = _REPO / "data"
FIGS  = _REPO / "figures"
FIGS.mkdir(exist_ok=True)
COMBINED   = DATA / "combined"
COMBINED.mkdir(exist_ok=True)

AU_ASSIGN  = DATA / "australian-no-letters" / "topic_assignments.csv"
AU_SUMMARY = DATA / "australian-no-letters" / "topic_summary_aus.csv"
G_ASSIGN   = DATA / "guardian"             / "topic_assignments.csv"
G_SUMMARY  = DATA / "guardian"             / "topic_summary.csv"
G_CSV      = _REPO.parent / "guardian_articles.csv"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

PUB_COLOURS = {
    "Guardian":      "#4C72B0",
    "The Age":       "#DD8452",
    "SMH":           "#55A868",
    "Canberra Times":"#C44E52",
}

# ── Step 1: Load & normalise ──────────────────────────────────────────────────

def norm_au_pub(p):
    p = str(p)
    if "Sydney Morning Herald" in p: return "SMH"
    if "Age, The" in p or "The Age" in p: return "The Age"
    if "Canberra Times" in p: return "Canberra Times"
    return None

def load_australian():
    df = pd.read_csv(AU_ASSIGN)
    df["pub"] = df["publication"].apply(norm_au_pub)
    df = df[df["pub"].notna()]
    df["corpus"] = "Australian"
    df["topic_uid"] = "AU_" + df["topic_id"].astype(str)
    return df

def load_guardian():
    df = pd.read_csv(G_ASSIGN)
    # Merge section from original CSV
    g_csv = pd.read_csv(G_CSV, usecols=["title","date","section"], low_memory=False)
    df = df.merge(g_csv, on=["title","date"], how="left")
    df["pub"] = "Guardian"
    df["corpus"] = "Guardian"
    df["topic_uid"] = "G_" + df["topic_id"].astype(str)
    return df

# ── Step 2: Build unified labels ──────────────────────────────────────────────

def clean_label(raw: str) -> str:
    """Strip BERTopic auto-label prefix (e.g. '0_word1_word2_...' → 'word1 word2 ...')"""
    parts = str(raw).split("_", 1)
    if len(parts) == 2:
        return parts[1].replace("_", " ").strip()
    return raw.strip()

def load_summaries():
    au = pd.read_csv(AU_SUMMARY)
    au = au[au["topic_id"] != -1].copy()
    au["topic_uid"]   = "AU_" + au["topic_id"].astype(str)
    au["clean_label"] = au["label"].apply(clean_label)
    au["corpus"]      = "Australian"

    g = pd.read_csv(G_SUMMARY)
    g = g[g["topic_id"] != -1].copy()
    g["topic_uid"]   = "G_" + g["topic_id"].astype(str)
    g["clean_label"] = g["label"].apply(clean_label)
    g["corpus"]      = "Guardian"

    return au, g

# ── Step 3: Cross-corpus alignment ────────────────────────────────────────────

def kw_set(rep_str):
    try:
        items = ast.literal_eval(rep_str)
        return {str(w).lower().strip() for w in items}
    except Exception:
        return {w.strip().strip("'\"[]").lower()
                for w in str(rep_str).split(",")}

def align_topics(au_sum, g_sum):
    rows = []
    for _, g_row in g_sum.iterrows():
        g_kw = kw_set(g_row["keywords"])
        best_j, best_uid, best_label = 0.0, None, ""
        for _, au_row in au_sum.iterrows():
            au_kw = kw_set(au_row["keywords"])
            if not g_kw or not au_kw:
                continue
            j = len(g_kw & au_kw) / len(g_kw | au_kw)
            if j > best_j:
                best_j, best_uid = j, au_row["topic_uid"]
                best_label = au_row["clean_label"]
        rows.append({
            "g_topic_uid":    g_row["topic_uid"],
            "g_label":        g_row["clean_label"],
            "g_count":        g_row["count"],
            "g_keywords":     g_row["keywords"],
            "au_topic_uid":   best_uid,
            "au_label":       best_label,
            "jaccard":        round(best_j, 4),
            "match_strength": ("strong"   if best_j >= 0.30 else
                               "moderate" if best_j >= 0.15 else "weak"),
        })
    df = pd.DataFrame(rows).sort_values("jaccard", ascending=False)
    df.to_csv(COMBINED / "topic_alignment.csv", index=False)
    strong   = (df["match_strength"] == "strong").sum()
    moderate = (df["match_strength"] == "moderate").sum()
    print(f"[align] Strong (J≥0.30): {strong}  Moderate (J≥0.15): {moderate}")
    return df

# ── Step 4: Combined assignment table ─────────────────────────────────────────

def build_combined(df_au, df_g, au_sum, g_sum):
    label_map = {}
    for _, r in au_sum.iterrows():
        label_map[r["topic_uid"]] = r["clean_label"]
    for _, r in g_sum.iterrows():
        label_map[r["topic_uid"]] = r["clean_label"]

    combined = pd.concat([
        df_au[["title","pub","corpus","year","era","topic_uid","content_type"]],
        df_g[["title","pub","corpus","year","era","topic_uid","content_type"]],
    ], ignore_index=True)
    combined["topic_label"] = combined["topic_uid"].map(label_map)
    combined.to_csv(COMBINED / "topic_combined.csv", index=False)
    print(f"[combined] {len(combined):,} articles → data/combined/topic_combined.csv")
    return combined, label_map

# ── Step 5: 4-column topic table (PDF) ───────────────────────────────────────

def fig_topic_table(au_sum, g_sum, combined):
    """4-column table: Australian topics (label, n) | Guardian topics (label, n)"""

    # Count articles per topic_uid
    counts = combined.groupby("topic_uid").size().to_dict()

    au_rows = au_sum.sort_values("count", ascending=False)[["topic_uid","clean_label","count"]].values.tolist()
    g_rows  = g_sum.sort_values("count",  ascending=False)[["topic_uid","clean_label","count"]].values.tolist()

    # Pad to equal length
    max_len = max(len(au_rows), len(g_rows))
    au_rows += [("", "", "")] * (max_len - len(au_rows))
    g_rows  += [("", "", "")] * (max_len - len(g_rows))

    # Split into two halves for 4-column layout
    half = (max_len + 1) // 2
    au_a, au_b = au_rows[:half], au_rows[half:]
    g_a,  g_b  = g_rows[:half],  g_rows[half:]

    # Combine into rows: [AU_label, AU_n, G_label, G_n] × 2
    def make_row(au, g):
        au_l = au[1][:32] if au[1] else ""
        au_n = str(au[2]) if au[2] else ""
        g_l  = g[1][:32]  if g[1]  else ""
        g_n  = str(g[2])  if g[2]  else ""
        return [au_l, au_n, g_l, g_n]

    table_rows = [make_row(au_a[i] if i<len(au_a) else ("","",""),
                           g_a[i]  if i<len(g_a)  else ("","",""))
                  for i in range(half)]
    table_rows += [make_row(au_b[i] if i<len(au_b) else ("","",""),
                            g_b[i]  if i<len(g_b)  else ("","",""))
                   for i in range(len(au_b))]

    col_labels  = ["Australian topic", "n", "Guardian topic", "n"]
    col_widths  = [0.36, 0.06, 0.36, 0.06]
    row_h       = 0.22
    fig_h       = max(6, len(table_rows) * row_h + 1.2)

    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        colWidths=col_widths,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("lightgrey")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f7f7f7")
        else:
            cell.set_facecolor("white")
        cell.set_height(row_h / fig_h)

    ax.set_title("Topic summary: Australian papers and Guardian corpus",
                 fontsize=10, pad=10, loc="left")
    fig.tight_layout()
    out = FIGS / "fig4_topic_table.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[fig] Topic table → {out}")

# ── Step 6: Top-10 horizontal bar chart by publication ───────────────────────

def fig_top10_by_pub(combined, label_map):
    """Horizontal stacked bar: top-10 topics by total article count, split by pub."""

    pubs = ["Guardian", "The Age", "SMH", "Canberra Times"]

    # Count per topic per pub
    ct = combined.groupby(["topic_uid","pub"]).size().unstack(fill_value=0)
    for p in pubs:
        if p not in ct.columns:
            ct[p] = 0
    ct = ct[pubs]
    ct["total"] = ct.sum(axis=1)
    ct = ct.sort_values("total", ascending=False).head(10)
    ct = ct.sort_values("total", ascending=True)  # ascending for horiz chart

    labels = [label_map.get(uid, uid)[:40] for uid in ct.index]
    totals = ct["total"].values

    fig, ax = plt.subplots(figsize=(10, 6))
    lefts = np.zeros(len(ct))
    for pub in pubs:
        vals = ct[pub].values
        bars = ax.barh(range(len(ct)), vals, left=lefts,
                       color=PUB_COLOURS[pub], label=pub, height=0.65)
        # Label segments > 2% of total
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v / totals[i] > 0.04:
                ax.text(l + v/2, i, f"{v:,}", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")
        lefts += vals

    ax.set_yticks(range(len(ct)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Number of articles")
    ax.set_title("Top 10 topics by article count — contribution by publication",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    fig.tight_layout()
    out = FIGS / "fig5_top10_by_pub.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[fig] Top-10 bar chart → {out}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("[load] Reading topic assignments …")
    df_au   = load_australian()
    df_g    = load_guardian()
    au_sum, g_sum = load_summaries()

    print("[align] Cross-corpus topic alignment …")
    align_topics(au_sum, g_sum)

    print("[combine] Building unified assignment table …")
    combined, label_map = build_combined(df_au, df_g, au_sum, g_sum)

    print("[fig] Generating topic table …")
    fig_topic_table(au_sum, g_sum, combined)

    print("[fig] Generating top-10 bar chart …")
    fig_top10_by_pub(combined, label_map)

    print("\n[done] All outputs written.")
    print(f"  data/combined/topic_alignment.csv")
    print(f"  data/combined/topic_combined.csv")
    print(f"  figures/fig4_topic_table.pdf")
    print(f"  figures/fig5_top10_by_pub.pdf")

if __name__ == "__main__":
    main()
