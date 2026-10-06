"""
config.py — Central configuration for the climate opinion corpus pipeline.
Edit paths here; all other scripts import from this module.
"""

from pathlib import Path

# ── Colorblind-safe palette (Okabe-Ito / Wong 2011) ───────────────────────────
# Use these constants in ALL plotting scripts so figures are consistent.
OKABE_ITO = {
    "blue":          "#0072B2",
    "orange":        "#E69F00",
    "bluish_green":  "#009E73",
    "sky_blue":      "#56B4E9",
    "vermillion":    "#D55E00",
    "reddish_purple":"#CC79A7",
    "yellow":        "#F0E442",
    "black":         "#000000",
}

# Harmonized thematic groups → colour
HARMONIZED_COLORS = {
    "Domestic Politics & Policy":    "#0072B2",   # blue
    "International Politics":        "#56B4E9",   # sky blue
    "International Climate Policy":  "#009E73",   # bluish green
    "Climate Science":               "#F0E442",   # yellow
    "Physical & Ecological Impacts": "#D55E00",   # vermillion
    "Energy & Industry":             "#E69F00",   # orange
    "Culture, Media & Activism":     "#CC79A7",   # reddish purple
}

# Outlet colours — 4 AU broadsheets + Guardian variants
OUTLET_COLORS = {
    "The Australian":        "#0072B2",   # blue
    "The Age":               "#E69F00",   # orange
    "Sydney Morning Herald": "#009E73",   # bluish green
    "Canberra Times":        "#D55E00",   # vermillion
    "The Guardian":          "#CC79A7",   # reddish purple (combined)
    "The Guardian (AU)":     "#CC79A7",   # reddish purple
    "The Guardian (UK)":     "#56B4E9",   # sky blue
}

# ── Root paths ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent   # repo root (scripts/ → repo root)
DATA_DIR    = ROOT / "data"                  # pipeline data directory
FIGURES_DIR = ROOT / "figures"               # output figures directory
MODELS_DIR  = ROOT / "models"                # local model cache (avoids ~/.cache)
ST_MODEL_DIR = MODELS_DIR / "all-MiniLM-L6-v2"  # sentence transformer cache

# ── Input corpora (CSV files; article body text not included — see README) ──────
CATALOGUE_CSV         = DATA_DIR / "articles_scored_australian.csv"
GUARDIAN_CATALOGUE_CSV = DATA_DIR / "guardian" / "guardian_articles_scored.csv"

# ── Output files (written into repo/data/ by the pipeline) ────────────────────
EXCEL_OUT       = DATA_DIR / "article_catalogue_review.xlsx"

# ── Relevance scoring vocabulary ───────────────────────────────────────────────
CLIMATE_TERMS = [
    "climate change", "global warming", "climate emergency", "greenhouse",
    "carbon emission", "carbon dioxide", "co2", "net zero", "net-zero",
    "carbon tax", "carbon price", "carbon pricing", "carbon trading",
    "emissions trading", "greenhouse gas",
    "renewable energy", "fossil fuel", "coal", "natural gas", "sea level",
    "arctic", "antarctic", "glacier", "drought", "bushfire", "wildfire",
    "flood", "extreme weather", "ipcc", "paris agreement", "kyoto",
    "decarboni", "clean energy", "solar", "wind energy", "climate action",
    "climate policy", "climate science", "climate denial", "climate sceptic",
    "climate skeptic", "adapt", "mitigation", "cop26", "cop27", "cop28",
    "climate crisis", "carbon neutral", "zero emission",
]

# ── Core climate identifiers (used as primary inclusion triggers) ──────────────
# These phrases unambiguously identify climate-change discourse across four
# dimensions: (1) canonical terms, (2) contemporary equivalents,
# (3) scientific mechanism, (4) policy mechanisms and targets.
# An article matching any of these ≥ CC_CORE_THRESHOLD times is included
# regardless of whether it uses the phrase "climate change" or "global warming".
CORE_CLIMATE_PHRASES = [
    # Canonical identifiers
    "climate change", "global warming",
    # Contemporary equivalents
    "climate emergency", "climate crisis",
    # Scientific mechanism
    "greenhouse gas", "ipcc",
    # Policy mechanisms (Australian legislative context: carbon price 2012–2014,
    # ETS proposals 2009–2010, net-zero commitments 2021–present)
    "carbon tax", "carbon price", "carbon pricing",
    "emissions trading", "carbon trading",
    # International frameworks and targets
    "paris agreement", "net zero", "net-zero",
]

# ── Inclusion criterion ────────────────────────────────────────────────────────
# An article is included if ANY of the following hold:
#   (a) Any CORE_CLIMATE_PHRASE appears >= CC_CORE_THRESHOLD times (combined)
#   (b) Any CORE_CLIMATE_PHRASE appears in the title
#   (c) Any CORE_CLIMATE_PHRASE appears >= 1 time AND
#       climate_mentions >= CLIMATE_MENTIONS_THRESHOLD
CC_CORE_THRESHOLD          = 3   # formerly CC_GW_THRESHOLD (value unchanged)
CLIMATE_MENTIONS_THRESHOLD = 3

# ── CT columnist names (for auto-classification) ───────────────────────────────
CT_COLUMNISTS = {
    "jack waterford", "john hewson", "crispin hull", "ebony bennett",
    "nicholas stuart", "michelle grattan", "john warhurst", "mark kenny",
    "adam triggs",
}

# ── The Australian columnist names (for auto-classification of Inquirer folders) ──
THE_AUSTRALIAN_COLUMNISTS = {
    "paul kelly", "chris kenny", "janet albrechtsen", "greg sheridan",
    "dennis shanahan", "gerard henderson", "bjorn lomborg", "peter van onselen",
    "troy bramston", "nick cater", "tom dusevic", "graham lloyd",
    "judith sloan", "adam creighton", "christopher pearson", "piers akerman",
    "james jeffrey", "gemma tognini", "rowan callick",
}
