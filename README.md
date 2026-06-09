# A Climate of Opinion — Corpus Construction Pipeline

Reproducible code for building the climate opinion corpus used in:

> Santony, B. (in prep). *A Climate of Opinion: Framing and sentiment in Australian newspaper coverage of climate change, 1990–2025.* Environmental Communication.

---

## Overview

This pipeline collects, parses, scores, and catalogues ~45,000 climate-related editorial and opinion articles from four Australian/Australian-edition newspapers (1990–2025). The final corpus contains **23,520 included articles** after applying the hybrid relevance criterion described below.

**Sources:**
| Publication | Access method |
|---|---|
| Sydney Morning Herald | NewsBank (PDF export) |
| The Age | NewsBank (PDF export) |
| Canberra Times | NewsBank (PDF export) |
| The Guardian (Australia) | Guardian Open Platform API |

---

## Repository structure

```
repo/
├── config.py               # All paths, API keys, thresholds, folder definitions
├── fetch_guardian.py       # Guardian API fetcher
├── parse_newsbank.py       # NewsBank multi-article PDF parser
├── score_and_classify.py   # Relevance scoring + hybrid criterion
├── build_catalogue.py      # Master pipeline orchestrator
├── build_excel.py          # Review workbook builder
├── make_figures.py         # Figures 2a/b/c
├── requirements.txt
└── README.md
```

Output files are written to `data/` and `figures/` (created automatically).

> **Note:** Article body text is not included in this repository due to copyright restrictions. The catalogue CSV contains metadata only (title, author, date, publication, word count, relevance scores).

---

## Setup

### 1. Install system dependency

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

### 3. Configure paths and API key

Edit `config.py`:

- Set `NEWSBANK_ROOT` to the directory containing your downloaded PDF folders (e.g. `../PDFs/`).
- Set `GUARDIAN_API_KEY` to your Guardian Open Platform API key (see below).

#### Obtaining a Guardian API key

The Guardian Open Platform provides free API access for non-commercial and research use.

1. Register at https://open-platform.theguardian.com/access/
2. Select **Developer** tier (free, up to 500 calls/day — sufficient for this pipeline).
3. You will receive a key by email within a few minutes.
4. Paste the key into `config.py`:

```python
GUARDIAN_API_KEY = "your-key-here"
```

> **Important:** never commit your API key to version control. If you fork this repository, add `config.py` to your local `.gitignore` or store the key in an environment variable and read it with `os.environ.get("GUARDIAN_API_KEY")`.

---

## Running the pipeline

### Full pipeline (fresh)

```bash
python build_catalogue.py
```

This will:
1. Parse all configured NewsBank PDF folders
2. Fetch Guardian articles from the API
3. Merge, deduplicate, and score all articles
4. Write `data/article_catalogue.csv`

### Skip PDF parsing (use cached CSV)

```bash
python build_catalogue.py --skip-newsbank
```

### Skip Guardian API (use cached CSV)

```bash
python build_catalogue.py --skip-guardian
```

### Build Excel review workbook

```bash
python build_excel.py
```

Writes a three-sheet workbook to `data/article_catalogue_review.xlsx`.

### Generate figures

```bash
python make_figures.py
```

Writes `figures/fig2a_articles_by_source.pdf` etc.

---

## Relevance criterion

An article is **included** if **any** of the following hold:

| Condition | Rule |
|---|---|
| (a) High direct frequency | `cc_count + gw_count ≥ 3` |
| (b) Title hit | Title contains "climate change" or "global warming" |
| (c) Broad climate coverage | `cc_count + gw_count ≥ 1` AND `climate_mentions ≥ 4` |

Where `cc_count` = occurrences of "climate change", `gw_count` = "global warming", and `climate_mentions` = total occurrences of any term in the 45-term `CLIMATE_TERMS` vocabulary (configured in `config.py`).

---

## NewsBank folder structure

NewsBank PDFs should be placed in subfolders of `NEWSBANK_ROOT`, matching the keys in `NEWSBANK_FOLDERS` in `config.py`. Each folder maps to a publication and content type. Subfolders one level deep are traversed automatically (e.g. `CT_opinion/pre2012`).

Two header formats are handled automatically:
- **Standard** (post ~2010): includes an `Author:` field
- **Legacy CT** (pre ~2012): `Section:` field only, no `Author:` line

---

## Citation

If you use this pipeline, please cite the paper above and link to this repository.

---

## Licence

Code: MIT  
Article content: not included (subject to NewsBank and Guardian licensing terms)
