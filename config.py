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
ROOT        = Path(__file__).parent          # repo root
DATA_DIR    = ROOT / "data"                  # output data directory
FIGURES_DIR = ROOT / "figures"               # output figures directory
MODELS_DIR  = ROOT / "models"                # local model cache (avoids ~/.cache)
ST_MODEL_DIR = MODELS_DIR / "all-MiniLM-L6-v2"  # sentence transformer cache

# ── Input: NewsBank PDF folders ────────────────────────────────────────────────
# Set NEWSBANK_ROOT to the directory containing your downloaded PDF folders.
NEWSBANK_ROOT = ROOT.parent / "Data"   # PDF folders live inside Data/

NEWSBANK_FOLDERS = {
    # SMH
    "SMH_HeraldsView":       {"publication": "Sydney Morning Herald", "content_type": "Editorial"},
    "SMH-PoliticalEditor":   {"publication": "Sydney Morning Herald", "content_type": "Columnist"},
    "SMH_PeterHartcher":     {"publication": "Sydney Morning Herald", "content_type": "Columnist"},
    "SMH_RossGittins":       {"publication": "Sydney Morning Herald", "content_type": "Columnist"},
    "SMH_Sheehan_Farrelly":  {"publication": "Sydney Morning Herald", "content_type": "Columnist"},
    "SMH_Devine":            {"publication": "Sydney Morning Herald", "content_type": "Columnist"},
    "SMH_analysis":          {"publication": "Sydney Morning Herald", "content_type": "Analysis"},
    "SMH_letters":           {"publication": "Sydney Morning Herald", "content_type": "Letters"},
    "SMH_opinion":           {"publication": "Sydney Morning Herald", "content_type": "Opinion/Op-Ed"},
    "SMH_NewsReview":        {"publication": "Sydney Morning Herald", "content_type": "Opinion/Op-Ed"},
    "SMH_1987_1990":         {"publication": "Sydney Morning Herald", "content_type": "Analysis"},
    "SMH_Editor":            {"publication": "Sydney Morning Herald", "content_type": "Editorial"},
    # The Age
    "TheAge_Editor":                 {"publication": "The Age", "content_type": "Editorial"},
    "TheAge_PolEditor":              {"publication": "The Age", "content_type": "Columnist"},
    "TheAge_Davdison_Grattan_Ross":  {"publication": "The Age", "content_type": "Columnist"},
    "TheAge_Analysis":               {"publication": "The Age", "content_type": "Analysis"},
    "TheAge_Letter_Insight":         {"publication": "The Age", "content_type": "Letters"},
    # Canberra Times
    "CT_Editorial":          {"publication": "Canberra Times", "content_type": "Editorial"},
    "CT_LTEditor":           {"publication": "Canberra Times", "content_type": "Letters"},
    "CT_Letters":            {"publication": "Canberra Times", "content_type": "Letters"},
    "CT_Letters_97-2007":    {"publication": "Canberra Times", "content_type": "Letters"},
    "CT_Opinion":            {"publication": "Canberra Times", "content_type": None},   # auto-classified
    "CT_opinion_analysis":   {"publication": "Canberra Times", "content_type": None},   # auto-classified
    # The Australian
    "TheAustralian_Analysis":        {"publication": "The Australian", "content_type": "Analysis"},
    "TheAustralian_Inquirer":        {"publication": "The Australian", "content_type": None},  # auto-classified
    "Australian_Inquirer_2122":      {"publication": "The Australian", "content_type": None},  # auto-classified
    "TheAustralian_SpecificEditors": {"publication": "The Australian", "content_type": "Columnist"},
    "TheAustralian_Letters":         {"publication": "The Australian", "content_type": "Letters"},
    "Australian_Editor": {"publication": "The Australian", "content_type": "Editorial"},
}

# ── Input: Guardian API ────────────────────────────────────────────────────────
GUARDIAN_API_KEY  = "key-from-api"   # https://open-platform.theguardian.com/
GUARDIAN_QUERIES  = ["climate change", "global warming", "climate emergency"]
GUARDIAN_FROM     = "1999-01-01"
GUARDIAN_TO       = "2026-04-30"

# Sections and the edition label assigned to each.
# commentisfree and environment are produced by the UK newsroom.
# australia-news is produced by Guardian Australia.
GUARDIAN_SECTION_EDITIONS = {
    "commentisfree": "UK",
    "environment":   "UK",
    "australia-news": "AU",
}
GUARDIAN_SECTIONS = list(GUARDIAN_SECTION_EDITIONS.keys())  # kept for back-compat

# Tone tags (Guardian taxonomy) — used to exclude straight news reporting.
# commentisfree articles carry tone/comment by default; environment and
# australia-news carry a mix of tone/news, tone/analysis, tone/features.
GUARDIAN_EXCLUDE_TONES = {"tone/news"}
GUARDIAN_INCLUDE_TONES = {
    "tone/comment",
    "tone/analysis",
    "tone/features",
    "tone/editorials",
    "tone/letters",       # exclude if you don't want letters — remove this line
}

# ── Input files (live in parent folder, alongside the PDF folders) ─────────────
# guardian_articles.csv and article_catalogue.csv are kept one level above the
# repo because they are too large / licensing-sensitive to commit to git.
PARENT_DIR      = ROOT.parent
GUARDIAN_CSV    = PARENT_DIR / "guardian_articles.csv"       # original pull (back-compat)
GUARDIAN_V2_CSV = PARENT_DIR / "guardian_articles_v2.csv"   # edition-tagged, tone-filtered
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
CLIMATE_MENTIONS_THRESHOLD = 3   # lowered from 4

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
