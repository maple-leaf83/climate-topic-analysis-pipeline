# A Climate of Opinion: Computational Analysis of Australian Climate Opinion Journalism, 1987–2026

**Bhavna J. Antony, Cameron Foale, Savin Chand**

*Institute of Innovation, Science and Sustainability, Federation University Australia*

### Abstract

Despite an extensive literature on climate change news coverage, long-form opinion and editorial journalism has received comparatively little computational attention, particularly in the Australian context. This paper presents a longitudinal computational analysis of climate opinion discourse across four major Australian newspapers — *The Australian*, *The Age*, the *Sydney Morning Herald*, and *The Canberra Times* — comprising 9,863 articles spanning 1987 to 2026. A central methodological contribution is the application of `nomic-embed-text-v1` — a sentence embedding model with an 8,192-token context window — paired with BERTopic for topic discovery. Standard transformer encoders truncate documents at 512 tokens (approximately 375 words), discarding the argumentative body of most editorial texts; the long-context embedder encodes each article in full, preserving the complete rhetorical arc in a single dense representation. The model recovered 84 topics, which were manually grouped into 7 thematic categories. Differential outlet attention was assessed using binomial representation ratios and z-scores against each publication's corpus-level baseline.

The analysis shows that climate change in Australian opinion journalism is overwhelmingly discussed through a political lens: the *Political Leadership & Party Dynamics* group accounts for 35.6% of all articles, suggesting the discourse is structured primarily around political conflict rather than scientific evidence or ecological consequence. Outlet-level differences in thematic emphasis are structural and stable across government eras: *The Australian* is systematically under-represented in climate science (r = 0.74) and carbon pricing (r = 0.78) relative to its corpus share, while *The Canberra Times* shows persistent concentration in environment and biodiversity coverage (r = 2.04). Temporal co-occurrence analysis reveals that climate science language has become increasingly embedded within political leadership discourse since the Turnbull–Morrison era, suggesting that attribution science is beginning to reshape the register of Australian climate commentary without displacing the dominant political frame. These findings provide the first computational characterisation of Australian climate opinion journalism at scale and establish a principled thematic scaffold for subsequent framing and stance analyses.

---

### Key Results

![Outlet attention by topic group](outlet_topic_attention_dotplot-1.png)

*Binomial effect sizes (z-score, square-root transformed axis) for each outlet–topic-group combination. Filled triangles indicate over- (▶, r > 1.25) or under-representation (◀, r < 0.75) relative to the outlet's corpus share; open squares indicate the outlet is within the expected range.*

The outlet–topic structure reveals systematic editorial divergence consistent with political parallelism. *The Australian* is over-represented in *Political Leadership & Party Dynamics* (r = 1.26) and under-represented in *Climate Science & Physical Impacts* (r = 0.74) and *Environment & Biodiversity* (r = 0.50) across every government era. *The Canberra Times* shows the most pronounced specialisation, with the largest representation ratio in the corpus for *Environment & Biodiversity* (r = 2.04), consistent with its proximity to Commonwealth environmental governance. *International Climate Diplomacy* is the only group for which no outlet crosses either threshold.

---

Reproducible code for the corpus construction and topic modelling pipeline used in:

> Antony, B., Foale, C. & Chand, S. (in prep). *A Climate of Opinion: Computational Analysis of Australian Climate Opinion Journalism, 1987–2026.*

Code repository: https://github.com/maple-leaf83/climate-topic-analysis-pipeline

---

## Overview

This pipeline parses, scores, and topic-models climate-related editorial and opinion articles from four Australian broadsheet publications spanning 1987–2026:

| Publication | Access method |
|---|---|
| The Australian | NewsBank Australia (PDF export) |
| The Age | NewsBank Australia (PDF export) |
| Sydney Morning Herald | NewsBank Australia (PDF export) |
| The Canberra Times | NewsBank Australia (PDF export) |

The final corpus contains **9,863** editorial and opinion articles after relevance screening.

> **Note:** Article body text is not included in this repository due to NewsBank licensing restrictions. The pipeline scripts are provided for transparency and reproducibility; to run them you will need your own NewsBank institutional access.

---

## Pipeline

The pipeline runs in four steps:

```
1. [manual] download NewsBank PDFs
2. cache_bodies.py            →  newsbank_bodies.parquet
3. build_articles_scored.py   →  data/articles_scored.csv
4. run_bertopic.py            →  data/australian-no-letters/topic_assignments.csv
```

---

## Setup

### 1. Install system dependency

`parse_newsbank.py` calls `pdftotext` (part of `poppler-utils`) to extract text from NewsBank PDF exports. Install it before running the pipeline:

```bash
# Ubuntu/Debian
sudo apt install poppler-utils

# macOS
brew install poppler
```

### 2. Install Python packages

```bash
pip install -r requirements.txt
```

### 3. Configure paths

Edit `config.py` and set `NEWSBANK_ROOT` to the directory containing your downloaded NewsBank PDF folders.

---

## Running the pipeline

### Step 1 — Download NewsBank PDFs (manual)

Log in to [NewsBank Australia](https://infoweb.newsbank.com) and export PDF bundles for each publication and content category as configured in `NEWSBANK_FOLDERS` in `config.py`. Place the folders under `NEWSBANK_ROOT`.

### Step 2 — Cache NewsBank body text

```bash
python cache_bodies.py
```

Parses all NewsBank PDFs using `pdftotext` and writes body text to `data/newsbank_bodies.parquet`. Run with `--force` to rebuild from scratch.

### Step 3 — Build scored article catalogue

```bash
python build_articles_scored.py
```

Applies the relevance criterion, assigns content-type classifications, and writes `data/articles_scored.csv`.

### Step 4 — Run BERTopic topic modelling

```bash
python run_bertopic.py --corpus australian --exclude-letters
```

Outputs `data/australian-no-letters/topic_assignments.csv`.

Key options:

| Flag | Default | Description |
|---|---|---|
| `--embedding-model` | `nomic-ai/nomic-embed-text-v1` | Sentence embedding model (8,192-token context) |
| `--min-topic-size` | `50` | Minimum cluster size |
| `--outlier-strategy` | `embeddings` | Outlier reassignment method |
| `--outlier-threshold` | `0.5` | Cosine similarity threshold for reassignment |
| `--device` | `auto` | `cpu` or `cuda` |

---

## Relevance criterion

An article is **included** if **any** of the following hold:

| Condition | Rule |
|---|---|
| (a) High direct frequency | `cc_count + gw_count ≥ 3` |
| (b) Title hit | Title contains "climate change" or "global warming" |
| (c) Broad climate vocabulary | `cc_count + gw_count ≥ 1` AND `climate_mentions ≥ 4` |

`climate_mentions` counts occurrences of any term in the 47-term `CLIMATE_TERMS` vocabulary defined in `config.py`.

---

## Repository structure

```
repo/
├── config.py                  # Paths, thresholds, folder definitions
├── parse_newsbank.py          # NewsBank PDF parser (called by cache_bodies)
├── cache_bodies.py            # Step 2: NewsBank body text cache builder
├── score_and_classify.py      # Relevance scoring logic (called by build_articles_scored)
├── build_articles_scored.py   # Step 3: scored article catalogue builder
├── run_bertopic.py            # Step 4: BERTopic topic modelling pipeline
├── analyse_clusters.py        # Temporal and outlet visualisations per topic group
├── analyse_cohesion.py        # Cosine cohesion boxplot (Figure 4 in paper)
├── outlet_topic_attention.py  # Binomial representation analysis across outlets
├── make_figures.py            # Corpus overview figures
├── make_prisma.py             # PRISMA flow diagram
├── requirements.txt
└── README.md
```

---

## Citation

If you use this pipeline, please cite:

> Antony, B., Foale, C. & Chand, S. (in prep). *A Climate of Opinion: Computational Analysis of Australian Climate Opinion Journalism, 1987–2026.*

---

## Licence

Code: MIT  
Article content: not included (subject to NewsBank licensing terms)
