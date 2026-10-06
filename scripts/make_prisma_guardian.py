"""
make_prisma_guardian.py
Generates a PRISMA-adapted corpus flow diagram for the Guardian corpus.
Shows API retrieval → tone filtering → relevance screening → included corpus.

All counts are derived dynamically from data/guardian/guardian_articles_scored.csv
and data/guardian/guardian_relevant.csv.

Outputs figures/fig_prisma_guardian.pdf/.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

REPO        = Path(__file__).parent
DATA_DIR    = REPO / "data" / "guardian"
FIGURES_DIR = REPO / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_prisma_stats() -> dict:
    scored   = pd.read_csv(DATA_DIR / "guardian_articles_scored.csv", low_memory=False)
    relevant = pd.read_csv(DATA_DIR / "guardian_relevant.csv",         low_memory=False)

    n_fetched        = len(scored)
    n_no_body        = (scored["final_status"] == "Excluded-NoBody").sum()
    n_not_relevant   = (scored["final_status"] == "Excluded-NotRelevant").sum()
    n_excl_screening = n_no_body + n_not_relevant

    # Analysis corpus: relevant, letters excluded
    n_relevant       = (scored["relevance"] == "Include").sum()
    n_letters        = (relevant["content_type"] == "Letters").sum()
    n_analysis       = n_relevant - n_letters   # used for BERTopic

    # Year range (relevant corpus)
    rel = scored[scored["relevance"] == "Include"].copy()
    rel["year"] = pd.to_numeric(rel["year"], errors="coerce")
    year_min = int(rel["year"].dropna().min())
    year_max = int(rel["year"].dropna().max())

    # Edition breakdown of analysis corpus
    analysis = rel[rel["content_type"] != "Letters"]
    ed_counts = analysis.groupby("edition").size().reindex(["UK", "AU"], fill_value=0)

    # Content-type breakdown of analysis corpus
    ct_counts = analysis.groupby("content_type").size().sort_values(ascending=False)

    print("── PRISMA statistics (Guardian corpus) ────────────────────────")
    print(f"  Total fetched (post tone-filter):  {n_fetched:,}")
    print(f"  Excluded — no body text:           {n_no_body:,}")
    print(f"  Excluded — not relevant:           {n_not_relevant:,}")
    print(f"  Total excluded at screening:       {n_excl_screening:,}")
    print(f"  Relevant (incl. letters):          {n_relevant:,}")
    print(f"  Letters excluded from analysis:    {n_letters:,}")
    print(f"  Final analysis corpus:             {n_analysis:,}")
    print(f"  Year range:                        {year_min}–{year_max}")
    print(f"  By edition:  UK={ed_counts['UK']:,}  AU={ed_counts['AU']:,}")
    print(f"  By content type:")
    for ct, n in ct_counts.items():
        print(f"    {ct}: {n:,}")
    print()

    return dict(
        n_fetched=n_fetched,
        n_no_body=n_no_body,
        n_not_relevant=n_not_relevant,
        n_excl_screening=n_excl_screening,
        n_relevant=n_relevant,
        n_letters=n_letters,
        n_analysis=n_analysis,
        year_min=year_min,
        year_max=year_max,
        ed_counts=ed_counts,
        ct_counts=ct_counts,
    )


s = get_prisma_stats()

# ── Colour scheme (Guardian palette) ──────────────────────────────────────────
C_BLUE   = '#005689'   # Guardian blue
C_GREEN  = '#1b7837'
C_RED    = '#b2182b'
C_GREY   = '#666666'
C_LBLUE  = '#d0e8f5'
C_LGREEN = '#d9f0d3'
C_LRED   = '#fddbc7'
C_LGREY  = '#eeeeee'
C_LTEAL  = '#d4eef2'

# ── Figure setup (normalised to match AU PRISMA) ───────────────────────────────
fig, ax = plt.subplots(figsize=(8.5, 7.0))
ax.set_xlim(0, 11)
ax.set_ylim(4.6, 12.2)
ax.axis('off')

Y_IDENT  = 11.4
Y_DEDUP  = 10.1
Y_SCREEN =  8.8
Y_INCL   =  7.5
Y_HBAR   =  6.5
Y_PUB    =  5.7
Y_CT     =  4.8    # content-type row (unused)

BH = 0.75
BW = 5.5


def box(ax, x, y, w, h, text, facecolor='#f7f7f7', edgecolor='#333333',
        fontsize=9, bold=False):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1",
                          facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=1.2, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            fontweight='bold' if bold else 'normal', zorder=3,
            multialignment='center', transform=ax.transData)


def arrow(ax, x1, y1, x2, y2, color='#555555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=1.5, connectionstyle='arc3,rad=0'))


def side_excl(ax, x, y, w, h, text):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=C_LRED, edgecolor=C_RED,
                          linewidth=1.0, zorder=2)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center', fontsize=9.0,
            color='black', zorder=3, multialignment='center',
            transform=ax.transData)


# ── Phase labels (left margin) ─────────────────────────────────────────────────
phase_defs = [
    ('Identification', Y_IDENT,  '#dce9f5'),
    ('Tone\nFiltering', Y_DEDUP,  '#eeeeee'),
    ('Screening',       Y_SCREEN, '#fff3cd'),
]
for label, yc, fc in phase_defs:
    rect = FancyBboxPatch((0.05, yc - 0.45), 1.6, 0.90,
                          boxstyle="round,pad=0.1",
                          facecolor=fc, edgecolor='#aaaaaa',
                          linewidth=0.8, zorder=1)
    ax.add_patch(rect)
    ax.text(0.69, yc, label, ha='center', va='center', fontsize=9.5,
            fontweight='bold', color='#333333', multialignment='center',
            transform=ax.transData)

# Included phase label — spans Included box down through edition boxes
incl_top    = Y_INCL + BH / 2
incl_bottom = Y_PUB - 0.31          # 0.31 = half of edition box height (0.62)
rect = FancyBboxPatch((0.05, incl_bottom - 0.1), 1.6,
                      incl_top - incl_bottom + 0.2,
                      boxstyle="round,pad=0.1",
                      facecolor='#d9f0d3', edgecolor='#aaaaaa',
                      linewidth=0.8, zorder=1)
ax.add_patch(rect)
ax.text(0.69, (incl_top+incl_bottom)/2, 'Included',
        ha='center', va='center', fontsize=9.5,
        fontweight='bold', color='#333333', multialignment='center',
        transform=ax.transData)

# ═══════════════════════════════════════════════════════════════════════════════
# IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
box(ax, 5.5, Y_IDENT, BW, 1.0,
    f'Guardian Content API — full-text retrieval\n'
    f'Sections: commentisfree, environment, australia-news\n'
    f'n = {s["n_fetched"]:,} records retrieved',
    facecolor=C_LBLUE, edgecolor=C_BLUE, fontsize=8.5)

arrow(ax, 5.5, Y_IDENT - 0.5, 5.5, Y_DEDUP + BH / 2 + 0.06)

# ═══════════════════════════════════════════════════════════════════════════════
# TONE FILTERING
# ═══════════════════════════════════════════════════════════════════════════════
box(ax, 5.5, Y_DEDUP, BW, BH,
    'Tone-tag filter applied at fetch\n'
    'Retained: opinion, analysis, features, editorials, letters\n'
    'Excluded at source: tone/news (straight reporting)',
    facecolor=C_LGREY, edgecolor='#555555', fontsize=8.5)

arrow(ax, 5.5, Y_DEDUP - BH / 2, 5.5, Y_SCREEN + BH / 2 + 0.06)

# ═══════════════════════════════════════════════════════════════════════════════
# SCREENING
# ═══════════════════════════════════════════════════════════════════════════════
box(ax, 5.5, Y_SCREEN, BW, BH,
    'Relevance criterion applied\n'
    '(keyword frequency in title and body text)',
    facecolor='#fff9e6', edgecolor='#e0a000', fontsize=8.5)

excl_text = (f'Excluded\nn = {s["n_excl_screening"]:,}'
             #f'(not relevant: {s["n_not_relevant"]:,};\n'
             # f' no body: {s["n_no_body"]:,})'
             )
side_excl(ax, 9.6, Y_SCREEN, 2.2, 0.80, excl_text)
ax.annotate('', xy=(8.5, Y_SCREEN), xytext=(8.25, Y_SCREEN),
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.3))

arrow(ax, 5.5, Y_SCREEN - BH / 2, 5.5, Y_INCL + BH / 2 + 0.06)

# ═══════════════════════════════════════════════════════════════════════════════
# INCLUDED
# ═══════════════════════════════════════════════════════════════════════════════
incl_text = (f'Final analysis corpus:\n N = {s["n_analysis"]:,} articles  ({s["year_min"]}–{s["year_max"]})'
            )
box(ax, 5.5, Y_INCL, BW, BH,
    incl_text,
    facecolor=C_LGREEN, edgecolor=C_GREEN, fontsize=10, bold=True)

arrow(ax, 5.5, Y_INCL - BH / 2, 5.5, Y_HBAR + 0.06)

# ── Edition breakdown ──────────────────────────────────────────────────────────
ed_labels  = [f'Guardian (UK)\nn = {s["ed_counts"]["UK"]:,}',
              f'Guardian (AU)\nn = {s["ed_counts"]["AU"]:,}']
ed_xs      = [3.5, 7.5]
ax.plot([3.5, 7.5], [Y_HBAR, Y_HBAR], color='#555555', lw=1.5, zorder=2)
for xc, label in zip(ed_xs, ed_labels):
    arrow(ax, xc, Y_HBAR, xc, Y_PUB + 0.31 + 0.06)
    box(ax, xc, Y_PUB, 2.6, 0.62, label,
        facecolor=C_LGREEN, edgecolor=C_GREEN, fontsize=10.0)

# ── Content-type breakdown ─────────────────────────────────────────────────────
ct_items  = list(s["ct_counts"].items())
n_ct      = len(ct_items)
x_start, x_end = 1.5, 9.5
spacing = (x_end - x_start) / (n_ct - 1) if n_ct > 1 else 0
ct_xs   = [x_start + i * spacing for i in range(n_ct)]
box_w   = min(1.6, spacing * 0.80)

# ax.text(5.5, Y_PUB - 0.55, 'by content type:', ha='center', va='center',
#         fontsize=7.5, color='#555555', transform=ax.transData)

CT_SHORT = {
    "Opinion/Op-Ed": "Opinion/\nOp-Ed",
    "Feature":       "Feature",
    "Analysis":      "Analysis",
    "Editorial":     "Editorial",
    "Letters":       "Letters\n(excluded)",
}
# for xc, (ct, n) in zip(ct_xs, ct_items):
#     short = CT_SHORT.get(ct, ct)
#     fc = '#fddbc7' if ct == "Letters" else C_LTEAL
#     ec = C_RED     if ct == "Letters" else '#2166ac'
#     box(ax, xc, Y_CT, box_w, 0.62,
#         f'{short}\nn = {n:,}',
#         facecolor=fc, edgecolor=ec, fontsize=7.5)

fig.tight_layout(pad=0.5)
for ext in (".pdf", ".png"):
    out = FIGURES_DIR / f"fig_prisma_guardian{ext}"
    fig.savefig(str(out), bbox_inches='tight', dpi=200)
    print(f"Saved → {out}")
plt.close()
