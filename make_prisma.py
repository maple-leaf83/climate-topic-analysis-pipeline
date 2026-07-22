"""
make_prisma.py
Generates a PRISMA-adapted corpus flow diagram for "A Climate of Opinion".
Shows the Australian domestic corpus only (NewsBank sources).
Guardian data was collected separately and is reserved for a future framing analysis.

All counts are derived dynamically from data/articles_scored.csv.

Outputs figures/fig1_prisma.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

from config import DATA_DIR, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_prisma_stats() -> dict:
    """
    Derive all PRISMA flow counts from articles_scored.csv,
    restricted to the four Australian NewsBank publications.

    Returns a dict with keys:
        n_newsbank_raw      — total Australian articles in scored CSV (post-dedup, pre-screening)
        n_excl_relevance    — excluded by relevance criterion
        n_after_relevance   — passing relevance screening
        n_included          — final included articles (editorial + letters)
        n_editorial         — included editorials/opinion (analysis corpus)
        n_letters           — included letters
        pub_counts          — pd.Series: included count by publication (desc)
        pub_editorial       — pd.Series: editorial-only count by publication (desc)
        year_min, year_max  — temporal range of included corpus
    """
    scored = pd.read_csv(DATA_DIR / "articles_scored.csv")

    # Restrict to Australian NewsBank publications only; exclude letters throughout
    au = scored[scored["publication"] != "The Guardian"].copy()
    au = au[au["year"].fillna(0).astype(int) >= 1987]
    au = au[au["content_type"] != "Letters"].copy()

    n_newsbank_raw    = len(au)
    n_editorial       = (au["final_status"] == "Included-Editorial").sum()
    n_excl_relevance  = n_newsbank_raw - n_editorial   # all non-editorial records
    n_after_relevance = n_editorial
    n_included        = n_editorial
    n_letters         = 0

    # Per-publication breakdown (all included: editorial + letters)
    inc          = au[au["final_status"].str.startswith("Included", na=False)]
    pub_counts   = inc.groupby("publication").size().sort_values(ascending=False)

    # Editorial-only breakdown (the analysis corpus)
    ed_only      = au[au["final_status"] == "Included-Editorial"]
    pub_editorial = ed_only.groupby("publication").size().sort_values(ascending=False)

    year_min = int(au["year"].dropna().astype(int).min())
    year_max = int(au["year"].dropna().astype(int).max())

    stats = dict(
        n_newsbank_raw=n_newsbank_raw,
        n_excl_relevance=n_excl_relevance,
        n_after_relevance=n_after_relevance,
        n_included=n_included,
        n_editorial=n_editorial,
        n_letters=n_letters,
        pub_counts=pub_counts,
        pub_editorial=pub_editorial,
        year_min=year_min,
        year_max=year_max,
    )

    print("── PRISMA statistics (Australian corpus only) ─────────────────")
    print(f"  NewsBank raw (post-dedup):    {n_newsbank_raw:,}")
    print(f"  Excl. not relevant:           {n_excl_relevance:,}")
    print(f"  After relevance screening:    {n_after_relevance:,}")
    print(f"  Final included:               {n_included:,}")
    print(f"    Editorial/opinion:          {n_editorial:,}")
    print(f"    Letters:                    {n_letters:,}")
    print("  By publication (all included):")
    for pub, n in pub_counts.items():
        print(f"    {pub}: {n:,}")
    print()

    return stats


# ── Load stats ─────────────────────────────────────────────────────────────────
s = get_prisma_stats()
n_newsbank_raw    = s["n_newsbank_raw"]
n_excl_relevance  = s["n_excl_relevance"]
n_after_relevance = s["n_after_relevance"]
n_included        = s["n_included"]
n_editorial       = s["n_editorial"]
n_letters         = s["n_letters"]
pub_counts        = s["pub_counts"]
pub_editorial     = s["pub_editorial"]
year_min          = s["year_min"]
year_max          = s["year_max"]

# ── Colour scheme ──────────────────────────────────────────────────────────────
C_BLUE   = '#2166ac'
C_GREEN  = '#1b7837'
C_RED    = '#b2182b'
C_GREY   = '#666666'
C_LBLUE  = '#d1e5f0'
C_LGREEN = '#d9f0d3'
C_LRED   = '#fddbc7'
C_LGREY  = '#eeeeee'

# ── Figure setup ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7.5))
ax.set_xlim(0, 11)
ax.set_ylim(4.8, 12.2)
ax.axis('off')

# ── Y positions (top to bottom) ────────────────────────────────────────────────
Y_IDENT  = 11.3   # Identification box centre
Y_DEDUP  =  9.9   # Deduplication box centre
Y_SCREEN =  8.6   # Screening box centre
Y_FINAL  =  7.3   # Final corpus box centre
Y_HBAR   =  6.3   # Horizontal connector
Y_PUB    =  5.6   # Per-publication boxes centre

BH = 0.72   # standard box height
BW = 5.5    # main box width


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
    ax.text(x, y, text, ha='center', va='center', fontsize=8.0,
            color='black', zorder=3, multialignment='center',
            transform=ax.transData)


# ── Phase labels (left margin) ─────────────────────────────────────────────────
for label, yc, fc in [
    ('Identification', Y_IDENT,  '#dce9f5'),
    ('Deduplication',  Y_DEDUP,  '#eeeeee'),
    ('Screening',      Y_SCREEN, '#fff3cd'),
]:
    rect = FancyBboxPatch((0.05, yc - 0.62), 1.28, 1.24,
                          boxstyle="round,pad=0.1",
                          facecolor=fc, edgecolor='#aaaaaa',
                          linewidth=0.8, zorder=1)
    ax.add_patch(rect)
    ax.text(0.69, yc, label, ha='center', va='center', fontsize=7.5,
            fontweight='bold', color='#333333', multialignment='center',
            transform=ax.transData)

# Included phase label — spans final corpus box down to per-pub boxes
incl_top    = Y_FINAL + BH / 2
incl_bottom = Y_PUB   - 0.31          # 0.31 = half of pub box height (0.62)
rect = FancyBboxPatch((0.05, incl_bottom - 0.1), 1.28,
                      incl_top - incl_bottom + 0.2,
                      boxstyle="round,pad=0.1",
                      facecolor='#d9f0d3', edgecolor='#aaaaaa',
                      linewidth=0.8, zorder=1)
ax.add_patch(rect)
ax.text(0.69, (incl_top + incl_bottom) / 2, 'Included',
        ha='center', va='center', fontsize=7.5,
        fontweight='bold', color='#333333', multialignment='center',
        transform=ax.transData)

# ═══════════════════════════════════════════════════════════════════════════════
# IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
box(ax, 5.5, Y_IDENT, BW, 1.0,
    f'NewsBank Australia — PDF export\n'
    f'The Australian, The Age, Sydney Morning Herald, Canberra Times\n'
    f'n = {n_newsbank_raw:,} records identified (post cross-folder deduplication)',
    facecolor=C_LBLUE, edgecolor=C_BLUE, fontsize=9)

arrow(ax, 5.5, Y_IDENT - 0.5, 5.5, Y_DEDUP + BH / 2 + 0.06)

# ═══════════════════════════════════════════════════════════════════════════════
# DEDUPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
box(ax, 5.5, Y_DEDUP, BW, BH,
    'Cross-folder duplicates removed (title × date × publication)\n'
    f'Pre-screening corpus: N = {n_newsbank_raw:,} articles',
    facecolor=C_LGREY, edgecolor='#555555', fontsize=8.5)

arrow(ax, 5.5, Y_DEDUP - BH / 2, 5.5, Y_SCREEN + BH / 2 + 0.06)

# ═══════════════════════════════════════════════════════════════════════════════
# SCREENING
# ═══════════════════════════════════════════════════════════════════════════════
box(ax, 5.5, Y_SCREEN, BW, BH,
    'Relevance criterion applied\n'
    '(keyword frequency in title and body text)',
    facecolor='#fff9e6', edgecolor='#e0a000', fontsize=8.5)

side_excl(ax, 9.6, Y_SCREEN, 2.2, 0.65,
          f'Not relevant\nn = {n_excl_relevance:,}')
ax.annotate('', xy=(8.5, Y_SCREEN), xytext=(8.25, Y_SCREEN),
            arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.3))

arrow(ax, 5.5, Y_SCREEN - BH / 2, 5.5, Y_FINAL + BH / 2 + 0.06)

# ═══════════════════════════════════════════════════════════════════════════════
# INCLUDED
# ═══════════════════════════════════════════════════════════════════════════════
box(ax, 5.5, Y_FINAL, BW, BH,
    f'Final included corpus: N = {n_editorial:,} \neditorial/opinion articles'
    f'  ({year_min}–{year_max})',
    facecolor=C_LGREEN, edgecolor=C_GREEN, fontsize=9, bold=True)

arrow(ax, 5.5, Y_FINAL - BH / 2, 5.5, Y_HBAR + 0.06)

# ── Per-publication breakdown ──────────────────────────────────────────────────
pubs   = list(pub_editorial.index)
counts = list(pub_editorial.values)
n_pubs = len(pubs)
x_start, x_end = 1.5, 9.1
spacing = (x_end - x_start) / (n_pubs - 1) if n_pubs > 1 else 0
xs      = [x_start + i * spacing for i in range(n_pubs)]
box_w   = min(1.7, spacing * 0.82)

PUB_SHORT = {
    "The Australian":        "The Australian",
    "The Age":               "The Age",
    "Sydney Morning Herald": "SMH",
    "Canberra Times":        "Canberra Times",
}

ax.plot([x_start, x_end], [Y_HBAR, Y_HBAR], color='#555555', lw=1.5, zorder=2)

for xc, pub, n in zip(xs, pubs, counts):
    arrow(ax, xc, Y_HBAR, xc, Y_PUB + 0.31 + 0.06)
    label = PUB_SHORT.get(pub, pub)
    box(ax, xc, Y_PUB, box_w, 0.62,
        f'{label}\nn = {n:,}',
        facecolor=C_LGREEN, edgecolor=C_GREEN, fontsize=8.0)

fig.tight_layout(pad=0.5)
out_path = FIGURES_DIR / "fig1_prisma.pdf"
fig.savefig(str(out_path), bbox_inches='tight', dpi=200)
plt.close()
print(f"\nPRISMA figure saved → {out_path}")
