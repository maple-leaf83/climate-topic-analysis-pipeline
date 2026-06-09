"""
config.py — Central configuration for the climate opinion corpus pipeline.
Edit paths here; all other scripts import from this module.
"""

from pathlib import Path

# ── Root paths ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent          # repo root
DATA_DIR    = ROOT / "data"                  # output data directory
FIGURES_DIR = ROOT / "figures"               # output figures directory
MODELS_DIR  = ROOT / "models"                # local model cache (avoids ~/.cache)
ST_MODEL_DIR = MODELS_DIR / "all-MiniLM-L6-v2"  # sentence transformer cache

# ── Input: NewsBank PDF folders ────────────────────────────────────────────────
# Set NEWSBANK_ROOT to the directory containing your downloaded PDF folders.
NEWSBANK_ROOT = Path("../")   # adjust to where your folders live

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
    # The Age
    "TheAge_PolEditor":              {"publication": "The Age", "content_type": "Columnist"},
    "TheAge_Davdison_Grattan_Ross":  {"publication": "The Age", "content_type": "Columnist"},
    "TheAge_Analysis":               {"publication": "The Age", "content_type": "Analysis"},
    "TheAge_Letter_Insight":         {"publication": "The Age", "content_type": "Analysis"},
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
}

# ── Input: Guardian API ────────────────────────────────────────────────────────
GUARDIAN_API_KEY  = "fef3eb2a-abbd-4b61-b58d-c09263627466"   # https://open-platform.theguardian.com/
GUARDIAN_SECTIONS = ["commentisfree", "environment", "australia-news"]
GUARDIAN_QUERIES  = ["climate change", "global warming", "climate emergency"]
GUARDIAN_FROM     = "1999-01-01"
GUARDIAN_TO       = "2026-04-30"

# ── Input files (live in parent folder, alongside the PDF folders) ─────────────
# guardian_articles.csv and article_catalogue.csv are kept one level above the
# repo because they are too large / licensing-sensitive to commit to git.
PARENT_DIR      = ROOT.parent
GUARDIAN_CSV    = PARENT_DIR / "guardian_articles.csv"
CATALOGUE_CSV   = DATA_DIR / "articles_scored.csv"

# ── Output files (written into repo/data/ by the pipeline) ────────────────────
EXCEL_OUT       = DATA_DIR / "article_catalogue_review.xlsx"

# ── Relevance scoring vocabulary ───────────────────────────────────────────────
CLIMATE_TERMS = [
    "climate change", "global warming", "climate emergency", "greenhouse",
    "carbon emission", "carbon dioxide", "co2", "net zero", "net-zero",
    "carbon tax", "carbon price", "carbon trading", "emissions trading",
    "renewable energy", "fossil fuel", "coal", "natural gas", "sea level",
    "arctic", "antarctic", "glacier", "drought", "bushfire", "wildfire",
    "flood", "extreme weather", "ipcc", "paris agreement", "kyoto",
    "decarboni", "clean energy", "solar", "wind energy", "climate action",
    "climate policy", "climate science", "climate denial", "climate sceptic",
    "climate skeptic", "adapt", "mitigation", "cop26", "cop27", "cop28",
    "climate crisis", "carbon neutral", "zero emission",
]

# ── Inclusion criterion ────────────────────────────────────────────────────────
# An article is included if ANY of the following hold:
#   (a) "climate change" + "global warming" combined count >= CC_GW_THRESHOLD
#   (b) Either phrase appears in the title
#   (c) Either phrase appears >= 1 time AND climate_mentions >= CLIMATE_MENTIONS_THRESHOLD
CC_GW_THRESHOLD         = 3
CLIMATE_MENTIONS_THRESHOLD = 4

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
