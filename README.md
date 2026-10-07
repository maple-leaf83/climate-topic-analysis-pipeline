# A Climate of Opinion: Computational Analysis of Australian Climate Opinion Journalism

**Bhavna J. Antony, Cameron Foale, Savin Chand**

*Institute of Innovation, Science and Sustainability, Federation University Australia*

Reproducible code for the topic modelling and analysis pipeline used in:

> Antony, B., Foale, C. & Chand, S. (in prep). *A Climate of Opinion: Computational Analysis of Australian Climate Opinion Journalism, 2001–2025.*

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
                                   data/guardian/guardian_articles_scored.csv

2. run_bertopic.py              →  data/australian/topic_assignments.csv
                                   data/guardian/topic_assignments_guardian.csv

3. build_topic_keywords.py      →  topic_keywords.xlsx

4. outlet_topic_attention.py    →  data/australian/outlet_binomial_zscores.csv
   outlet_attention_comparison.py
   

6. compare_corpora.py           →  figures/comparison/

7. temporal_comparison.py       →  figures/comparison/temporal_*.pdf

8. cohesion_analysis.py         →  data/*/cohesion_scores_*.csv
                                →  figures/cohesion/

```

---

## Setup

### Install Python packages

```bash
pip install -r requirements.txt
```

### Configure paths

Edit `scripts/config.py` to set any local embedding model paths if running offline.

---

## Running the pipeline

### 1. Relevance scoring

`score_and_classify.py` is a general-purpose CLI that works on any article CSV. Run it separately for each corpus:

```bash
python scripts/score_and_classify.py input.csv output.csv
python scripts/score_and_classify.py input.csv output.csv --include-only
```

Use `--title-col` and `--body-col` if your CSV uses different column names (defaults: `title`, `body`). The `--include-only` flag writes only articles that pass the criterion.

An article is included if any of the following hold:

| Condition | Rule |
|---|---|
| (a) High direct frequency | Any `CORE_CLIMATE_PHRASE` appears ≥ 3 times in body |
| (b) Title hit | Any `CORE_CLIMATE_PHRASE` appears in the title |
| (c) Broad climate vocabulary | Any `CORE_CLIMATE_PHRASE` appears ≥ 1 time AND `climate_mentions ≥ 3` |

`climate_mentions` counts occurrences of any term in the `CLIMATE_TERMS` vocabulary defined in `config.py`. `CORE_CLIMATE_PHRASES` covers canonical identifiers, contemporary equivalents, scientific mechanisms, and policy terms (also in `config.py`).

---

### 2. BERTopic topic modelling

```bash
python scripts/run_bertopic.py --corpus australian 
python scripts/run_bertopic.py --corpus guardian
```

Fits independent BERTopic models for each corpus using `nomic-embed-text-v1` (8,192-token context window), which encodes each article in full without truncation. Outlier articles are reassigned by cosine similarity to the nearest topic embedding.


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

> **Note:** The models were fitted on a GPU cluster. Embedding 9,863 AU articles + 9,976 Guardian articles with `nomic-embed-text-v1` at full context requires substantial RAM and benefits from a CUDA-capable GPU.

---

### 3. Topic keywords

```bash
python scripts/build_topic_keywords.py
```

Extracts the top-N c-TF-IDF keywords for every BERTopic topic in both corpora and writes them to `topic_keywords.xlsx`. Used during manual taxonomy harmonisation.

---

### 4. Taxonomy harmonisation (manual step)

After running `build_topic_keywords.py`, inspect the keyword output and assign each BERTopic topic to one of the 7 harmonised categories. Record the mappings in the `AU_TO_HARMONIZED` and `G_TO_HARMONIZED` dicts in `config.py` before proceeding to the analysis steps below.

---

### 5. Representation analysis

**Outlet × topic-group binomial analysis**

```bash
python scripts/outlet_topic_attention.py
```

Computes whether each outlet devotes significantly more or less attention to each harmonised topic group than its corpus-level share would predict. Uses binomial z-scores and representation ratios *r*. 

**Cross-corpus outlet comparison**

```bash
python scripts/outlet_attention_comparison.py
```

Produces comparison figures of outlet-level attention between the AU and Guardian corpora.

---

### 6. Cross-corpus comparison and co-occurrence

```bash
python scripts/compare_corpora.py
```

Two main analyses:

1. **Cross-corpus co-occurrence** — for each harmonised topic group, identifies the top 10 c-TF-IDF keywords and computes the proportion of articles per era in which each keyword co-occurs with the group's primary signal. 

2. **Temporal lines by outlet** — plots yearly topic-group share for each AU outlet individually alongside the *Guardian* mean.

---

### 7. Temporal analysis

```bash
python scripts/temporal_comparison.py
```

Produces era-level stacked bar charts and year-by-year topic share lines comparing the AU and Guardian corpora. 

---

### 8. Semantic cohesion analysis

```bash
python scripts/cohesion_analysis.py
```

`cohesion_analysis.py` computes the cosine similarity between each article's embedding and its assigned harmonised group centroid for both corpora.

---

## Citation

If you use this pipeline, please cite:

> Antony, B., Foale, C. & Chand, S. (in prep). *A Climate of Opinion: Computational Analysis of Australian Climate Opinion Journalism, 2001–2025.*

---

## Licence

Code: MIT  
Article content: not included (subject to NewsBank and Guardian licensing terms)
