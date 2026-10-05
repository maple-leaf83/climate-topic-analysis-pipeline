# A Climate of Opinion: Computational Analysis of Australian Climate Opinion Journalism

**Bhavna J. Antony, Cameron Foale, Savin Chand**

*Institute of Innovation, Science and Sustainability, Federation University Australia*

Reproducible code for the topic modelling and analysis pipeline used in:

> Antony, B., Foale, C. & Chand, S. (in prep). *A Climate of Opinion: Computational Analysis of Australian Climate Opinion Journalism, 2001–2025.*

Code repository: https://github.com/maple-leaf83/climate-topic-analysis-pipeline

---

## Overview

This pipeline takes scored CSV corpora of climate opinion articles and runs BERTopic topic modelling, taxonomy harmonisation, representation analysis, co-occurrence analysis, temporal analysis, and figure generation.

**The pipeline assumes you already have two scored article corpora as CSV files:**

| File | Contents |
|---|---|
| `data/articles_scored_australian.csv` | AU broadsheet opinion articles after relevance screening |
| `data/guardian/guardian_articles_scored.csv` | *Guardian Australia* opinion articles after relevance screening |

Relevance scoring scripts are included for transparency (see [Scoring](#1-relevance-scoring) below), but article body text is not included in this repository due to NewsBank and Guardian licensing restrictions.

---

## Pipeline

```
1. score_and_classify.py        →  data/articles_scored_australian.csv
   guardian_build_scored.py     →  data/guardian/guardian_articles_scored.csv

2. run_bertopic.py              →  data/australian-no-letters/topic_assignments.csv
                                   data/guardian/topic_assignments_guardian.csv

3. build_topic_keywords.py      →  topic_keywords.xlsx

4. report_topics.py             →  data/topic_combined.csv

5. outlet_topic_attention.py    →  data/australian-no-letters/outlet_binomial_zscores.csv
   outlet_attention_comparison.py
   era_stratified_representation.py

6. compare_corpora.py           →  figures/comparison/

7. temporal_comparison.py       →  figures/comparison/temporal_*.pdf

8. cohesion_analysis.py         →  data/*/cohesion_scores_*.csv
   analyse_cohesion.py          →  figures/cohesion/

9. make_prisma.py               →  figures/fig1_prisma.pdf
   make_prisma_guardian.py      →  figures/fig_prisma_guardian.pdf
```

---

## Setup

### Install Python packages

```bash
pip install -r requirements.txt
```

### Configure paths

Edit `config.py` to set corpus paths and any local embedding model paths if running offline.

---

## Running the pipeline

### 1. Relevance scoring

**Australian corpus**

```bash
python score_and_classify.py
```

Applies the hybrid relevance criterion to raw article CSVs and writes `data/articles_scored_australian.csv`. An article is included if any of the following hold:

| Condition | Rule |
|---|---|
| (a) High direct frequency | `cc_count + gw_count ≥ 3` |
| (b) Title hit | Title contains "climate change" or "global warming" |
| (c) Broad climate vocabulary | `cc_count + gw_count ≥ 1` AND `climate_mentions ≥ 4` |

`climate_mentions` counts occurrences of any term in the 47-term `CLIMATE_TERMS` vocabulary defined in `config.py`.

**Guardian corpus**

```bash
python guardian_build_scored.py
```

Applies equivalent relevance screening to the Guardian article CSV and writes `data/guardian/guardian_articles_scored.csv`.

---

### 2. BERTopic topic modelling

```bash
python run_bertopic.py --corpus australian --exclude-letters
python run_bertopic.py --corpus guardian
```

Fits independent BERTopic models for each corpus using `nomic-embed-text-v1` (8,192-token context window), which encodes each article in full without truncation. Outlier articles are reassigned by cosine similarity to the nearest topic embedding.

Outputs:
- `data/australian-no-letters/topic_assignments.csv`
- `data/australian-no-letters/topic_summary.csv`
- `data/guardian/topic_assignments_guardian.csv`
- `data/guardian/topic_summary_guardian.csv`
- Saved models under `models/`

Key options:

| Flag | Default | Description |
|---|---|---|
| `--corpus` | `both` | `australian`, `guardian`, `both`, or `combined` |
| `--embedding-model` | `nomic-ai/nomic-embed-text-v1` | Sentence embedding model |
| `--au-min-topic-size` | `20` | Minimum cluster size for AU corpus |
| `--min-topic-size` | `50` | Minimum cluster size for Guardian corpus |
| `--outlier-strategy` | `embeddings` | Outlier reassignment method (`embeddings` or `c-tf-idf`) |
| `--outlier-threshold` | `0.5` | Cosine similarity threshold for reassignment |
| `--device` | `auto` | `cpu` or `cuda` |
| `--exclude-letters` | — | Exclude letters-to-the-editor from topic modelling |

> **Note:** The models were fitted on a GPU cluster. Embedding 9,863 AU articles + 5,171 Guardian articles with `nomic-embed-text-v1` at full context requires substantial RAM and benefits from a CUDA-capable GPU.

---

### 3. Topic keywords

```bash
python build_topic_keywords.py
```

Extracts the top-N c-TF-IDF keywords for every BERTopic topic in both corpora and writes them to `topic_keywords.xlsx`. Used during manual taxonomy harmonisation.

---

### 4. Topic reporting and taxonomy harmonisation

```bash
python report_topics.py
```

Loads both sets of topic assignments, applies the manually defined 7-category harmonised taxonomy (configured in `config.py` via `HARMONIZED_COLORS` and the group-name mappings), and writes:
- `data/topic_combined.csv` — all articles with harmonised group labels
- `data/topic_alignment.csv` — cross-corpus Jaccard alignment table
- `figures/fig4_topic_table.pdf` — topic summary table

> **Manual step:** After running `build_topic_keywords.py`, inspect the keyword output and assign each BERTopic topic to one of the 7 harmonised categories. Record the mappings in the `AU_TO_HARMONIZED` and `G_TO_HARMONIZED` dicts in `config.py` before proceeding.

---

### 5. Representation analysis

**Outlet × topic-group binomial analysis**

```bash
python outlet_topic_attention.py
```

Computes whether each outlet devotes significantly more or less attention to each harmonised topic group than its corpus-level share would predict. Uses binomial z-scores and representation ratios *r*. Outputs to `data/australian-no-letters/` and `figures/`.

**Cross-corpus outlet comparison**

```bash
python outlet_attention_comparison.py
```

Produces comparison figures of outlet-level attention between the AU and Guardian corpora.

**Era-stratified representation**

```bash
python era_stratified_representation.py
```

Runs the outlet × topic-group binomial analysis separately for each government era (Howard, Rudd/Gillard, Abbott/Turnbull, Morrison, Albanese) to test whether structural differences persist across political periods.

---

### 6. Cross-corpus comparison and co-occurrence

```bash
python compare_corpora.py
```

Two main analyses:

1. **Cross-corpus co-occurrence** — for each harmonised topic group, identifies the top 10 c-TF-IDF keywords and computes the proportion of articles per era in which each keyword co-occurs with the group's primary signal. Outputs `figures/comparison/cooccurrence_heatmap.pdf`.

2. **Temporal lines by outlet** — plots yearly topic-group share for each AU outlet individually alongside the *Guardian* mean, with a dotted AU mean line. Outputs `figures/comparison/temporal_lines_by_outlet.pdf`.

---

### 7. Temporal analysis

```bash
python temporal_comparison.py
```

Produces era-level stacked bar charts and year-by-year topic share lines comparing the AU and Guardian corpora. Outputs to `figures/comparison/`.

---

### 8. Semantic cohesion analysis

```bash
python cohesion_analysis.py
python analyse_cohesion.py
```

`cohesion_analysis.py` computes the cosine similarity between each article's embedding and its assigned harmonised group centroid for both corpora, writing `data/australian-no-letters/cohesion_scores_aus.csv` and `data/guardian/cohesion_scores_guardian.csv`.

`analyse_cohesion.py` generates the cohesion boxplot figure (`figures/cohesion/cohesion_clusters.pdf`) comparing within-group embedding tightness across corpora.

---

### 9. PRISMA diagrams

```bash
python make_prisma.py
python make_prisma_guardian.py
```

Generates PRISMA-style flow diagrams for each corpus's inclusion/exclusion screening.

---

## Repository structure

```
repo/
├── config.py                        # Paths, colour palettes, taxonomy mappings, thresholds
│
├── score_and_classify.py            # Step 1a: AU relevance scoring
├── guardian_build_scored.py         # Step 1b: Guardian relevance scoring
│
├── run_bertopic.py                  # Step 2: BERTopic modelling (AU and Guardian)
├── build_topic_keywords.py          # Step 3: c-TF-IDF keyword extraction
│
├── report_topics.py                 # Step 4: topic report and taxonomy harmonisation
├── analyse_clusters.py              # Per-group publication/era breakdowns
│
├── outlet_topic_attention.py        # Step 5a: outlet binomial representation analysis
├── outlet_attention_comparison.py   # Step 5b: cross-corpus outlet comparison
├── era_stratified_representation.py # Step 5c: era-stratified representation
│
├── compare_corpora.py               # Step 6: co-occurrence and outlet temporal lines
├── temporal_comparison.py           # Step 7: era-level and yearly temporal figures
│
├── cohesion_analysis.py             # Step 8a: compute cosine cohesion scores
├── analyse_cohesion.py              # Step 8b: cohesion boxplot figure
│
├── make_prisma.py                   # Step 9a: PRISMA diagram (AU corpus)
├── make_prisma_guardian.py          # Step 9b: PRISMA diagram (Guardian corpus)
│
├── requirements.txt
└── README.md
```

---

## Data outputs

```
data/
├── articles_scored_australian.csv           # AU corpus after relevance screening
├── australian-no-letters/
│   ├── topic_assignments.csv                # BERTopic assignments (AU)
│   ├── topic_summary.csv                    # Topic-level c-TF-IDF summary (AU)
│   ├── australian_topic_groups.csv          # Harmonised group labels (AU)
│   ├── outlet_binomial_zscores.csv          # Representation z-scores
│   ├── outlet_representation_ratios.csv     # Representation ratios r
│   └── cohesion_scores_aus.csv              # Cosine cohesion (AU)
└── guardian/
    ├── guardian_articles_scored.csv         # Guardian corpus after relevance screening
    ├── topic_assignments_guardian.csv       # BERTopic assignments (Guardian)
    ├── topic_summary_guardian.csv           # Topic-level c-TF-IDF summary (Guardian)
    ├── guardian_topic_groups.csv            # Harmonised group labels (Guardian)
    └── cohesion_scores_guardian.csv         # Cosine cohesion (Guardian)
```

---

## Citation

If you use this pipeline, please cite:

> Antony, B., Foale, C. & Chand, S. (in prep). *A Climate of Opinion: Computational Analysis of Australian Climate Opinion Journalism, 2001–2025.*

---

## Licence

Code: MIT  
Article content: not included (subject to NewsBank and Guardian licensing terms)
