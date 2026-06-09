"""
make_prisma.py
Generates a PRISMA-adapted corpus flow diagram for "A Climate of Opinion".
All counts are derived dynamically from:
  data/articles_scored.csv   — post-dedup, scored corpus
  ../guardian_articles.csv   — raw Guardian API output (for pre-dedup count)
  data/newsbank_raw.csv      — raw NewsBank output (for pre-dedup count, if present)

Outputs figures/fig1_prisma.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

from config import DATA_DIR, FIGURES_DIR, GUARDIAN_CSV

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_prisma_stats() -> dict:
    """
    Derive all PRISMA flow counts from articles_scored.csv.

    Returns a dict with keys:
        n_guardian_raw      — raw Guardian API articles (from guardian_articles.csv)
        n_newsbank_raw      — raw NewsBank articles (from newsbank_raw.csv if present)
        n_prescreening      — post-dedup total entering screening
        n_excl_relevance    — excluded by relevance criterion
        n_after_relevance   — passing relevance screening
        n_excl_scope        — excluded as straight news (scope restriction)
        n_included          — final included articles
        n_editorial         — included editorials/opinion
        n_letters           — included letters
        pub_counts          — pd.Series: included count by publication (desc)
        newsbank_pubs       — list of non-Guardian publication names
        year_min, year_max  — temporal range of included corpus
    """
    scored = pd.read_csv(DATA_DIR / "articles_scored.csv")

    # Raw pre-dedup counts from source files
    n_guardian_raw = 0
    if GUARDIAN_CSV.exists():
        try:
            n_guardian_raw = len(pd.read_csv(GUARDIAN_CSV, usecols=["title"]))
        except Exception:
            n_guardian_raw = (scored["publication"] == "The Guardian").sum()

    nb_raw_path = DATA_DIR / "newsbank_raw.csv"
    if nb_raw_path.exists():
        try:
            n_newsbank_raw = len(pd.read_csv(nb_raw_path, usecols=["title"]))
        except Exception:
            n_newsbank_raw = (scored["publication"] != "The Guardian").sum()
    else:
        n_newsbank_raw = (scored["publication"] != "The Guardian").sum()

    # Screening counts
    n_prescreening    = len(scored)
    n_excl_relevance  = (scored["relevance"] == "Exclude").sum()
    n_after_relevance = n_prescreening - n_excl_relevance
    n_excl_scope      = (scored["final_status"] == "Excluded-ScopeNews").sum()
    n_included        = scored["final_status"].str.startswith("Included", na=False).sum()
    n_editorial       = (scored["final_status"] == "Included-Editorial").sum()
    n_letters         = (scored["final_status"] == "Included-Letter").sum()

    # Per-publication breakdown (included only)
    inc        = scored[scored["final_status"].str.startswith("Included", na=False)]
    pub_counts = inc.groupby("publication").size().sort_values(ascending=False)

    nb_pubs   = sorted(scored[scored["publication"] != "The Guardian"]["publication"].unique())
    year_min  = int(scored["year"].dropna().astype(int).min())
    year_max  = int(scored["year"].dropna().astype(int).max())

    stats = dict(
        n_guardian_raw=n_guardian_raw,
        n_newsbank_raw=n_newsbank_raw,
        n_prescreening=n_prescreening,
        n_excl_relevance=n_excl_relevance,
        n_after_relevance=n_after_relevance,
        n_excl_scope=n_excl_scope,
        n_included=n_included,
        n_editorial=n_editorial,
        n_letters=n_letters,
        pub_counts=pub_counts,
        newsbank_pubs=nb_pubs,
        year_min=year_min,
        year_max=year_max,
    )

    print("── PRISMA statistics ──────────────────────────────────────────")
    print(f"  Guardian (raw):       {n_guardian_raw:,}")
    print(f"  NewsBank (raw):       {n_newsbank_raw:,}")
    print(f"  Pre-screening total:  {n_prescreening:,}")
    print(f"  Excl. not relevant:   {n_excl_relevance:,}")
    print(f"  After relevance:      {n_after_relevance:,}")
    print(f"  Excl. scope (news):   {n_excl_scope:,}")
    print(f"  Final included:       {n_included:,}")
    print(f"    Editorial/opinion:  {n_editorial:,}")
    print(f"    Letters:            {n_letters:,}")
    print("  By publication:")
    for pub, n in pub_counts.items():
        print(f"    {pub}: {n:,}")
    print()

    return stats


# ── Load stats ─────────────────────────────────────────────────────────────────
s = get_prisma_stats()
n_guardian_raw    = s["n_guardian_raw"]
n_newsbank_raw    = s["n_newsbank_raw"]
n_prescreening    = s["n_prescreening"]
n_excl_relevance  = s["n_excl_relevance"]
n_after_relevance = s["n_after_relevance"]
n_excl_scope      = s["n_excl_scope"]
n_included        = s["n_included"]
pub_counts        = s["pub_counts"]
nb_pubs           = s["newsbank_pubs"]
year_min          = s["year_min"]
year_max          = s["year_max"]

# ── Colour scheme ──────────────────────────────────────────────────────────────
C_BLUE   = '#2166ac'
C_GREEN  = '#1b7837'
C_RED    = '#b2182b'
C_LBLUE  = '#d1e5f0'
C_LGREEN = '#d9f0d3'
C_LRED   = '#fddbc7'

# ── Figure setup ───────────────────────────────────────────────────────────────
n_pubs = len(pub_counts)
fig_w  = max(7.5, 1.8 * n_pubs + 1.5)
fig, ax = plt.subplots(figsize=(fig_w, 8))
ax.set_xlim(0, 10)
ax.set_ylim(2, 13.5)
ax.axis('off')


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
phases = [
    ('Identification',       12.4, '#dce9f5'),
    ('Combination\n& Dedup', 10.2, '#dce9f5'),
    ('Screening',             7.5, '#fff3cd'),
    ('Included',              3.8, '#d9f0d3'),
]
for label, yc, fc in phases:
    rect = FancyBboxPatch((0.05, yc - 0.78), 1.28, 1.56,
                          boxstyle="round,pad=0.1",
                          facecolor=fc, edgecolor='#aaaaaa',
                          linewidth=0.8, zorder=1)
    ax.add_patch(rect)
    ax.text(0.69, yc, label, ha='center', va='center', fontsize=7.5,
            fontweight='bold', color='#333333', multialignment='center',
            transform=ax.transData)

# ═══════════════════════════════════════════════════════════════════════════════
# IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
box(ax, 3.2, 12.8, 3.5, 0.95,
    f'Guardian Open Platform API\n3 queries × 3 sections\nn = {n_guardian_raw:,} articles',
    facecolor=C_LBLUE, edgecolor=C_BLUE, fontsize=8.5)

nb_pub_str = ", ".join(nb_pubs) if len(nb_pubs) <= 3 else f"{len(nb_pubs)} publications"
box(ax, 7.3, 12.8, 3.5, 0.95,
    f'NewsBank Australia (PDF export)\n{nb_pub_str}\nn = {n_newsbank_raw:,} articles',
    facecolor=C_LBLUE, edgecolor=C_BLUE, fontsize=8.5)

arrow(ax, 3.2, 12.32, 5.05, 10.68)
arrow(ax, 7.3, 12.32, 5.55, 10.68)

# ═══════════════════════════════════════════════════════════════════════════════
# COMBINATION & DEDUP
# ═══════════════════════════════════════════════════════════════════════════════
box(ax, 5.3, 10.28, 4.5, 0.82,
    'Records combined and deduplicated\n(by title × date × publication)',
    facecolor='#eeeeee', edgecolor='#555555', fontsize=8.5)

arrow(ax, 5.3, 9.87, 5.3, 9.40)

box(ax, 5.3, 9.02, 4.5, 0.78,
    f'Pre-screening corpus\nN = {n_prescreening:,} articles',
    facecolor='#e8e8e8', edgecolor='#333333', fontsize=9, bold=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SCREENING
# ═══════════════════════════════════════════════════════════════════════════════
arrow(ax, 5.3, 8.63, 5.3, 8.08)

box(ax, 5.3, 7.70, 4.5, 0.78,
    'Relevance criterion applied\n(keyword frequency in title and body)',
    facecolor='#fff9e6', edgecolor='#e0a000', fontsize=8.5)

side_excl(ax, 8.75, 7.70, 2.65, 0.72,
          f'Not relevant\nn = {n_excl_relevance:,}')
ax.annotate('', xy=(7.55, 7.70), xytext=(8.42, 7.70),
            arrowprops=dict(arrowstyle='<-', color=C_RED, lw=1.3))

arrow(ax, 5.3, 7.31, 5.3, 6.78)

box(ax, 5.3, 6.42, 4.5, 0.72,
    'Scope restriction: opinion and analysis only\n(Guardian Australia news excluded)',
    facecolor='#fff9e6', edgecolor='#e0a000', fontsize=8.5)

side_excl(ax, 8.75, 6.42, 2.65, 0.72,
          f'Straight news reporting\nn = {n_excl_scope:,}')
ax.annotate('', xy=(7.55, 6.42), xytext=(8.42, 6.42),
            arrowprops=dict(arrowstyle='<-', color=C_RED, lw=1.3))

arrow(ax, 5.3, 6.06, 5.3, 5.48)

# ═══════════════════════════════════════════════════════════════════════════════
# INCLUDED
# ═══════════════════════════════════════════════════════════════════════════════
yr_min = year_min
yr_max = year_max

box(ax, 5.3, 5.12, 4.5, 0.72,
    f'Articles assessed for full inclusion\nn = {n_after_relevance:,}',
    facecolor='#fff9e6', edgecolor='#e0a000', fontsize=8.5)

arrow(ax, 5.3, 4.76, 5.3, 4.25)

box(ax, 5.3, 3.88, 4.5, 0.75,
    f'Final analysis corpus\nN = {n_included:,} articles ({yr_min}–{yr_max})',
    facecolor=C_LGREEN, edgecolor=C_GREEN, fontsize=9, bold=True)

# ── Per-publication breakdown ──────────────────────────────────────────────────
pubs      = list(pub_counts.index)
counts    = list(pub_counts.values)
n_pubs    = len(pubs)
# Keep boxes well within [0, 10] by anchoring to [1.2, 8.8]
x_start, x_end = 1.2, 8.8
spacing   = (x_end - x_start) / (n_pubs - 1) if n_pubs > 1 else 0
xs        = [x_start + i * spacing for i in range(n_pubs)] if n_pubs > 1 else [5.3]
box_w     = min(1.6, spacing * 0.82)  # never wider than spacing × 0.82, max 1.6

# Short publication labels
PUB_SHORT = {
    "The Guardian":           "The Guardian",
    "The Age":                "The Age",
    "Sydney Morning Herald":  "SMH",
    "Canberra Times":         "Canberra Times",
    "The Australian":         "The Australian",
}

for xc, pub, n in zip(xs, pubs, counts):
    arrow(ax, xc, 3.50, xc, 2.92)
    label = PUB_SHORT.get(pub, pub)
    box(ax, xc, 2.62, box_w, 0.68,
        f'{label}\nn = {n:,}',
        facecolor=C_LGREEN, edgecolor=C_GREEN, fontsize=8.0)

fig.tight_layout(pad=0.5)
out_path = FIGURES_DIR / "fig1_prisma.pdf"
fig.savefig(str(out_path), bbox_inches='tight', dpi=200)
plt.close()
print(f"\nPRISMA figure saved → {out_path}")
