"""
run_bertopic.py
BERTopic topic modelling pipeline for "A Climate of Opinion".

Loads article body text from Guardian CSV and (optionally) NewsBank cache,
fits BERTopic models (Guardian and/or Australian papers separately), and saves:
  - data/{corpus}/topic_assignments.csv   : per-article topic labels
  - data/{corpus}/topic_summary.csv       : topic label, size, keywords
  - figures/{corpus}/fig3a_topics_over_time.pdf/png
  - figures/{corpus}/fig3b_topics_by_pub.pdf/png
  - figures/{corpus}/fig3c_topics_by_era.pdf/png
  - models/{corpus}/bertopic_model/       : serialised model
  - data/topic_alignment.csv              : cross-corpus topic comparison (--corpus both)

Usage:
    python run_bertopic.py [--corpus guardian|australian|both|combined]
                          [--min-topic-size N]
                          [--au-min-topic-size N]
                          [--nr-topics N]
                          [--device cpu|cuda]
                          [--transform-only [--source-model LABEL]]

Options:
    --corpus            Which corpus to run: guardian, australian, both, or combined (default: both)
    --min-topic-size N  Min articles per topic for Guardian corpus  (default: 50)
    --au-min-topic-size N  Min articles per topic for Australian corpus (default: 20)
    --nr-topics N       Reduce to N topics after fit (default: auto)
    --device cpu|cuda   Embedding device (default: auto-detect)
    --transform-only    Project letters into an existing model using BERTopic.transform()
                        rather than fitting a new model. Saves to data/letters/.
    --source-model      Label of the saved model directory to load for --transform-only
                        (default: combined-no-letters-no-aunews)

Requirements:
    pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn
    pip install pandas matplotlib numpy tqdm
    Run cache_bodies.py once before this script to build the NewsBank body cache.
"""

import os
import argparse
import warnings
from pathlib import Path

# Redirect ALL HuggingFace/transformers cache away from ~/.cache
# Must happen before any sentence_transformers / transformers imports.
# Force-set rather than setdefault — cluster environments often pre-set HF_HOME
# to ~/.cache/huggingface, which would point the modules cache to the wrong place.
_REPO_ROOT = Path(__file__).resolve().parent
_HF_CACHE  = str(_REPO_ROOT / "models" / "hf_cache")
os.environ["HF_HUB_CACHE"]              = _HF_CACHE   # highest-priority cache var
os.environ["HF_HOME"]                   = _HF_CACHE
os.environ["TRANSFORMERS_CACHE"]        = _HF_CACHE
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(_REPO_ROOT / "models")
# Always run in offline mode — models must be pre-downloaded to HF_HUB_CACHE
# (GPU compute nodes have no internet access; login nodes are used for downloads)
os.environ["HF_HUB_OFFLINE"]      = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (
    DATA_DIR, FIGURES_DIR, CATALOGUE_CSV, GUARDIAN_CSV,
    MODELS_DIR, ST_MODEL_DIR,
)

# Prefer the scored catalogue (output of build_articles_scored.py) if it exists;
# fall back to the legacy article_catalogue.csv for backward compatibility.
SCORED_CSV = DATA_DIR / "articles_scored.csv"
from cache_bodies import load_cache as _load_nb_cache

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Government eras ───────────────────────────────────────────────────────────
ERAS = [
    (1987, 1996, "Pre-Howard"),
    (1996, 2007, "Howard"),
    (2007, 2013, "Rudd/Gillard"),
    (2013, 2015, "Abbott"),
    (2015, 2022, "Turnbull/Morrison"),
    (2022, 2026, "Albanese"),
]


def year_to_era(year):
    if pd.isna(year):
        return "Unknown"
    for start, end, label in ERAS:
        if start <= int(year) < end:
            return label
    return "Unknown"


def _corpus_dirs(corpus_name: str):
    """Return (data_dir, figures_dir, models_dir) for a named corpus."""
    d = DATA_DIR / corpus_name
    f = FIGURES_DIR / corpus_name
    m = MODELS_DIR / corpus_name
    for p in (d, f, m):
        p.mkdir(parents=True, exist_ok=True)
    return d, f, m


# ── Step 1: Load body text ────────────────────────────────────────────────────

def load_guardian_bodies(catalogue: pd.DataFrame,
                         exclude_au_news: bool = False) -> pd.DataFrame:
    print("[load] Guardian body text …")
    g = pd.read_csv(GUARDIAN_CSV, low_memory=False)
    # Keep section for filtering; rename body
    g = g[["title", "date", "section", "body"]].rename(columns={"body": "body_text"})
    if exclude_au_news:
        before = len(g)
        g = g[g["section"] != "Australia news"]
        print(f"  Guardian: excluded {before - len(g):,} Australia news articles")
    guardian_cat = catalogue[catalogue["publication"].str.contains("Guardian", na=False)].copy()
    # Drop pre-existing 'section' from catalogue (if present) to avoid pandas
    # producing section_x / section_y after the merge below.
    if "section" in guardian_cat.columns:
        guardian_cat = guardian_cat.drop(columns=["section"])
    merged = guardian_cat.merge(g, on=["title", "date"], how="left")
    # Guardian body CSV has one row per article, so title+date is unique there;
    # but guard against any accidental fan-out just in case
    merged = merged.drop_duplicates(subset=["title", "date", "publication"])
    # Retain section as content_type for Guardian
    merged["content_type"] = merged["section"].fillna(merged["content_type"])
    n_matched = merged['body_text'].notna().sum()
    n_total   = len(merged)
    print(f"  Guardian: {n_matched:,} / {n_total:,} matched")
    if n_matched < n_total:
        missing = merged[merged['body_text'].isna()][["title", "date"]].head(20)
        print(f"  [warn] {n_total - n_matched} Guardian articles have no body text in CSV:")
        print(missing.to_string(index=False))
    return merged


def load_newsbank_bodies(catalogue: pd.DataFrame) -> pd.DataFrame:
    nb_cat = catalogue[~catalogue["publication"].str.contains("Guardian", na=False)].copy()
    try:
        print("[load] NewsBank body text from cache …")
        cache = _load_nb_cache().rename(columns={"body": "body_text"})
        # Deduplicate cache before merging to prevent fan-out on repeated entries.
        # Older cache files may not have a 'publication' column — fall back gracefully.
        if "publication" in cache.columns:
            merge_keys = ["title", "date", "publication"]
        else:
            print("  [warn] Cache has no 'publication' column — merging on title+date only")
            merge_keys = ["title", "date"]
        cache = cache.drop_duplicates(subset=merge_keys)
        merged = nb_cat.merge(cache, on=merge_keys, how="left")
    except FileNotFoundError:
        raise FileNotFoundError(
            "No NewsBank body cache found. Run:  python cache_bodies.py\n"
            "Then re-run this script."
        )
    print(f"  NewsBank: {merged['body_text'].notna().sum():,} / {len(merged):,} matched")
    return merged


def build_text_corpus(catalogue: pd.DataFrame, corpus: str,
                      exclude_letters: bool = False,
                      exclude_au_news: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame of included articles with 'body_text'.
    corpus: 'guardian' | 'australian' | 'combined'
    exclude_letters: drop Letters content_type (Australian papers)
    exclude_au_news: drop Guardian Australia news section (keeps Opinion + Environment only)

    Body text is read directly from the catalogue's 'body' column when present
    (i.e. when articles_scored.csv was built by the current build_articles_scored.py).
    This avoids re-merging against source files and eliminates title/date mismatch
    errors between the catalogue and the raw CSVs.  Legacy catalogues without a
    'body' column fall back to the original source-file merge.
    """
    # ── Select included editorials ────────────────────────────────────────────
    if "final_status" in catalogue.columns:
        included = catalogue[catalogue["final_status"] == "Included-Editorial"].copy()
        print(f"[corpus] {len(included):,} editorial articles (from final_status column)")
    else:
        included = catalogue[catalogue["relevance"] == "Include"].copy()
        if exclude_letters:
            before = len(included)
            included = included[included["content_type"] != "Letters"]
            print(f"[corpus] Excluded {before - len(included):,} letters "
                  f"({before - len(included)}/{before} = "
                  f"{(before-len(included))/before*100:.1f}%)")

    # ── Filter by corpus scope ────────────────────────────────────────────────
    if corpus == "guardian":
        included = included[included["publication"].str.contains("Guardian", na=False)].copy()
    elif corpus == "australian":
        included = included[~included["publication"].str.contains("Guardian", na=False)].copy()
    # combined: use all of included as-is

    if exclude_au_news and corpus in ("guardian", "combined"):
        before = len(included)
        included = included[
            ~(included["publication"].str.contains("Guardian", na=False) &
              included["section"].str.lower().isin(["australia news", "australia-news"]))
        ].copy()
        if len(included) < before:
            print(f"  [scope] Excluded {before - len(included):,} Guardian Australia news articles")

    # ── Body text: use catalogue column if available (fast path) ─────────────
    if "body" in included.columns:
        df = included.copy()
        df = df.rename(columns={"body": "body_text"})
        print(f"[corpus] Body text read directly from catalogue "
              f"({df['body_text'].notna().sum():,} / {len(df):,} present)")
    else:
        # Legacy fallback: re-merge against source files
        print("[corpus] No 'body' column in catalogue — falling back to source-file merge")
        if corpus == "guardian":
            df = load_guardian_bodies(included, exclude_au_news=False)  # already filtered above
        elif corpus == "australian":
            df = load_newsbank_bodies(included)
        else:
            g_df  = load_guardian_bodies(included, exclude_au_news=False)
            nb_df = load_newsbank_bodies(included)
            df = pd.concat([g_df, nb_df], ignore_index=True)
            before = len(df)
            df = df.drop_duplicates(subset=["title", "date", "publication"])
            if before > len(df):
                print(f"  [dedup] Removed {before - len(df):,} residual duplicates after concat")

    df = df[df["body_text"].notna() & (df["body_text"].str.len() > 50)]
    df["era"] = df["year"].apply(year_to_era)
    print(f"[corpus:{corpus}] {len(df):,} articles with usable body text")
    return df


# ── Step 2: Fit BERTopic ──────────────────────────────────────────────────────

def fit_bertopic(docs: list, min_topic_size: int = 50, nr_topics=None,
                 device: str = "auto",
                 embedding_model_path: str = "nomic-ai/nomic-embed-text-v1",
                 batch_size: int = 256) -> tuple:
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bertopic] Device: {device}  |  min_topic_size: {min_topic_size}")

    # If a local snapshot path is provided, disable all HF network lookups so
    # the model loads purely from disk (required on HPC compute nodes with no
    # internet access).  The env-var is set here rather than globally so that
    # it only affects this process and only when needed.
    print(f"[bertopic] Loading embedding model: {embedding_model_path}  (cache: {_HF_CACHE})")
    embedding_model = SentenceTransformer(
        embedding_model_path, device=device,
        local_files_only=True,
        model_kwargs={"torch_dtype": "float16"},   # fp16 halves activation memory
    )

    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0,
        metric="cosine", random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_topic_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    EXTRA_STOPWORDS = [
        # Generic corpus noise
        "climate", "change", "global", "warming", "australia", "australian",
        "said", "says", "year", "years", "time", "new", "also", "like",
        "just", "one", "two", "three", "would", "could", "people",
        "government", "minister", "mr", "ms", "dr", "per", "cent",
        # Publication name fragments (URL artifacts and masthead references)
        "canberratimes", "canberratimes com", "com au", "canberra times",
        "sydney morning herald", "smh com", "theage com", "age com",
        "guardian com", "theguardian",
        # Pandemic terms (bleed into climate articles by date proximity)
        "covid", "covid 19", "pandemic", "coronavirus",
        # Letter/opinion form words that cluster on register not theme
        "letter", "letters", "editor", "dear", "sincerely", "regards",
        "write", "writer", "columnist", "opinion", "sir", "madam",
    ]
    vectorizer = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS) + EXTRA_STOPWORDS,
        ngram_range=(1, 2), min_df=5, max_df=0.85,
    )

    # Two-stage representation pipeline:
    #   1. KeyBERTInspired selects candidate keywords by cosine similarity to
    #      the mean topic embedding (semantically representative over merely frequent).
    #   2. MaximalMarginalRelevance diversifies the final keyword set by penalising
    #      redundancy — preventing near-synonyms (e.g. "carbon price", "carbon tax",
    #      "carbon pricing") from dominating the topic label.
    representation_model = [
        KeyBERTInspired(),
        MaximalMarginalRelevance(diversity=0.3),
    ]

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=representation_model,
        nr_topics=nr_topics,
        top_n_words=10,
        verbose=True,
        calculate_probabilities=False,
    )

    # Pre-compute embeddings with a large batch size to saturate the GPU.
    # Passing pre-computed embeddings to fit_transform also means they can be
    # reused if BERTopic needs to be rerun with different parameters.
    # batch_size=512 is safe for an 80GB A100 with nomic-embed-text-v1;
    # raise to 1024 if GPU memory headroom is still available.
    embed_cache = Path(embedding_model_path).parent.parent.parent / "embeddings_cache.npy"
    if embed_cache.exists():
        embeddings = np.load(str(embed_cache))
        if embeddings.shape[0] != len(docs):
            print(f"[bertopic] Cached embeddings shape {embeddings.shape} does not match "
                  f"corpus size {len(docs):,} — discarding cache and recomputing.")
            embed_cache.unlink()
            embeddings = None
        else:
            print(f"[bertopic] Loading cached embeddings from {embed_cache} "
                  f"(shape: {embeddings.shape})")
    else:
        embeddings = None

    if embeddings is None:
        print(f"[bertopic] Embedding {len(docs):,} documents "
              f"(batch_size={batch_size}, fp16) …")
        embeddings = embedding_model.encode(
            docs, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True,
        )
        np.save(str(embed_cache), embeddings)
        print(f"[bertopic] Embeddings saved to {embed_cache}")

    print(f"[bertopic] Fitting on {len(docs):,} documents …")
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)
    n_topics = len(set(topics)) - 1
    n_outliers = sum(1 for t in topics if t == -1)
    print(f"[bertopic] {n_topics} topics | {n_outliers:,} outliers "
          f"({n_outliers/len(topics)*100:.1f}%) — outlier reduction not yet applied")
    return topic_model, topics, probs, embeddings


# ── Step 2b: Outlier reduction (separate from fit so it can be rerun) ─────────

def reduce_topic_outliers(topic_model, docs: list, topics: list,
                          strategy: str = "c-tf-idf",
                          threshold: float = 0.5,
                          embeddings: np.ndarray = None) -> list:
    """
    Reassign outlier documents (topic -1) to the nearest topic.

    Parameters
    ----------
    strategy   : "c-tf-idf" (fast, vocabulary-based) or "embeddings" (uses
                 cosine similarity in the full embedding space — more effective
                 but requires the embeddings array to be passed in).
    threshold  : Minimum similarity score to accept a reassignment.
                 For c-tf-idf scores are sparse so 0.5 rarely fires; prefer
                 0.0–0.1 with c-tf-idf.
                 For embeddings, cosine similarity is denser; 0.3–0.5 works well.
    embeddings : Pre-computed document embeddings (n_docs × dim numpy array).
                 Required when strategy="embeddings"; ignored otherwise.

    Returns the updated topics list (plain Python ints).
    """
    n_before = sum(1 for t in topics if t == -1)
    if n_before == 0:
        print("[outliers] No outliers to reduce.")
        return topics

    if strategy == "embeddings" and embeddings is None:
        print("[outliers] WARNING: strategy='embeddings' requested but no embeddings "
              "supplied — falling back to c-tf-idf.")
        strategy = "c-tf-idf"

    print(f"[outliers] Reducing {n_before:,} outliers "
          f"(strategy={strategy}, threshold={threshold}) …")

    if strategy == "embeddings" and embeddings is not None:
        # Manual centroid-based reassignment — works for both fitted docs and
        # transformed docs (where topic_model.reduce_outliers() raises
        # "No outliers to reduce" because the docs aren't stored in the model).
        topic_centroids = topic_model.topic_embeddings_   # shape (n_topics+1, dim); row 0 = outlier centroid
        # Build an ordered list of (topic_id, centroid) skipping the outlier centroid (topic -1)
        topic_ids   = sorted([t for t in topic_model.get_topics().keys() if t != -1])
        # topic_embeddings_ rows are ordered: outlier (-1) first, then topics 0, 1, 2 …
        # The row index for topic t is t + 1 (outlier at index 0)
        centroids   = np.array([topic_centroids[t + 1] for t in topic_ids])  # (n_topics, dim)
        # Normalise for cosine similarity
        emb_norm    = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        cent_norm   = centroids  / (np.linalg.norm(centroids,  axis=1, keepdims=True) + 1e-10)
        sims        = emb_norm @ cent_norm.T   # (n_docs, n_topics)
        topics = list(topics)
        for i, t in enumerate(topics):
            if t == -1:
                best_idx  = int(np.argmax(sims[i]))
                best_sim  = float(sims[i, best_idx])
                if best_sim >= threshold:
                    topics[i] = topic_ids[best_idx]
    else:
        # Fitted-model path: use BERTopic's built-in method (only valid for
        # documents that were part of the original fit).
        kwargs = dict(strategy=strategy, threshold=threshold)
        topics = topic_model.reduce_outliers(docs, topics, **kwargs)
        topics = [int(t) for t in topics]
        topic_model.update_topics(docs, topics=topics,
                                  vectorizer_model=topic_model.vectorizer_model,
                                  representation_model=topic_model.representation_model)

    topics = [int(t) for t in topics]
    n_after = sum(1 for t in topics if t == -1)
    print(f"[outliers] {len(set(topics)) - 1} topics | "
          f"{n_after:,} still unassigned after reduction "
          f"({n_after/len(topics)*100:.1f}%)")
    return topics


def sample_noise_cluster(corpus_df: pd.DataFrame, topics: list,
                         n_sample: int = 100,
                         data_dir: Path = None,
                         random_state: int = 42) -> pd.DataFrame:
    """
    Draw a random sample from the remaining noise cluster (topic == -1)
    and save to <data_dir>/noise_sample.csv for manual inspection.

    Returns the sample DataFrame (empty if no noise remains).
    """
    df = corpus_df.copy()
    df["topic_id"] = topics
    noise = df[df["topic_id"] == -1]
    n_noise = len(noise)

    if n_noise == 0:
        print("[noise] No noise-cluster articles remain.")
        return pd.DataFrame()

    n_draw = min(n_sample, n_noise)
    sample = noise.sample(n=n_draw, random_state=random_state)

    print(f"[noise] {n_noise:,} articles in noise cluster — "
          f"saving random sample of {n_draw} for inspection")

    out_cols = ["title", "author", "publication", "content_type",
                "date", "year", "body_text"]
    out = sample[[c for c in out_cols if c in sample.columns]]

    if data_dir is not None:
        path = data_dir / "noise_sample.csv"
        out.to_csv(path, index=False)
        print(f"[noise] Sample → {path}")

    return out


# ── Step 3: Save outputs ──────────────────────────────────────────────────────

def save_topic_assignments(corpus_df: pd.DataFrame, topics: list,
                           probs, data_dir: Path) -> pd.DataFrame:
    df = corpus_df.copy()
    df["topic_id"] = topics
    df["topic_prob"] = (
        [max(p) for p in probs] if probs is not None and hasattr(probs[0], "__iter__")
        else probs if probs is not None
        else None
    )
    out_cols = ["title", "author", "publication", "content_type",
                "date", "year", "era", "folder", "filename",
                "topic_id", "topic_prob"]
    out = df[[c for c in out_cols if c in df.columns]]
    path = data_dir / "topic_assignments.csv"
    out.to_csv(path, index=False)
    print(f"[save] Assignments → {path}")
    return df


def save_topic_summary(topic_model, corpus_df: pd.DataFrame,
                       data_dir: Path) -> pd.DataFrame:
    info = topic_model.get_topic_info()
    rows = []
    for _, row in info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            rep = "[Outlier cluster]"
        else:
            titles = corpus_df[corpus_df["topic_id"] == tid]["title"].dropna()
            rep = " | ".join(titles.head(3).tolist())
        rows.append({
            "topic_id": tid,
            "label":    row.get("Name", ""),
            "count":    row["Count"],
            "keywords": row.get("Representation", ""),
            "rep_titles": rep,
        })
    df_sum = pd.DataFrame(rows)
    path = data_dir / "topic_summary.csv"
    df_sum.to_csv(path, index=False)
    print(f"[save] Summary     → {path}")
    return df_sum


def save_model(topic_model, models_dir: Path):
    path = str(models_dir / "bertopic_model")
    topic_model.save(path, serialization="safetensors",
                     save_ctfidf=True, save_embedding_model=False)
    print(f"[save] Model       → {path}/")


# ── Step 4: Figures ───────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

ELECTIONS = [1996, 1998, 2001, 2004, 2007, 2010, 2013, 2016, 2019, 2022, 2025]


def fig_topics_over_time(topic_model, corpus_df: pd.DataFrame,
                         figures_dir: Path, corpus_name: str, top_n: int = 8):
    print("[fig] Topics over time …")
    corp = corpus_df[corpus_df["topic_id"] != -1].copy()
    corp["year"] = pd.to_numeric(corp["year"], errors="coerce")
    corp = corp[corp["year"].between(1987, 2026)]

    top_topics = corp["topic_id"].value_counts().head(top_n).index.tolist()
    info = topic_model.get_topic_info().set_index("Topic")
    labels = {t: info.loc[t, "Name"] if t in info.index else f"Topic {t}"
              for t in top_topics}

    years = sorted(corp["year"].unique().astype(int))
    totals = corp.groupby("year").size()

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.get_cmap("tab10", top_n)
    for i, tid in enumerate(top_topics):
        sub = corp[corp["topic_id"] == tid].groupby("year").size()
        pct = (sub / totals * 100).reindex(years, fill_value=0)
        ax.plot(years, pct, label=labels[tid], color=cmap(i), linewidth=1.6)

    for yr in ELECTIONS:
        if min(years) <= yr <= max(years):
            ax.axvline(yr, color="grey", linewidth=0.5, linestyle=":", alpha=0.6)

    ax.set_xlabel("Year")
    ax.set_ylabel("% of annual articles")
    ax.set_title(f"Top {top_n} topics over time — {corpus_name}")
    ax.legend(frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7.5)
    ax.set_xlim(min(years), max(years))
    fig.tight_layout()
    stem = figures_dir / "fig3a_topics_over_time"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  → {stem}.pdf")


def fig_topics_by_publication(topic_model, corpus_df: pd.DataFrame,
                               figures_dir: Path, corpus_name: str, top_n: int = 10):
    print("[fig] Topics by publication …")
    corp = corpus_df[corpus_df["topic_id"] != -1].copy()
    info = topic_model.get_topic_info().set_index("Topic")
    top_topics = corp["topic_id"].value_counts().head(top_n).index.tolist()
    labels = [info.loc[t, "Name"] if t in info.index else f"Topic {t}"
              for t in top_topics]

    PUB_MAP = {
        "Sydney Morning Herald": "SMH",
        "Age, The": "The Age", "The Age": "The Age",
        "Canberra Times": "Canberra Times",
        "Guardian": "Guardian",
    }

    def norm_pub(p):
        p = str(p)
        for k, v in PUB_MAP.items():
            if k in p:
                return v
        if "Guardian" in p:
            return "Guardian"
        return None

    corp["pub"] = corp["publication"].apply(norm_pub)
    corp = corp[corp["pub"].notna()]
    pubs_present = [p for p in ["Guardian", "The Age", "SMH", "Canberra Times"]
                    if p in corp["pub"].unique()]

    mat = pd.crosstab(corp["pub"], corp["topic_id"])
    mat = mat[[t for t in top_topics if t in mat.columns]]
    mat = mat.div(mat.sum(axis=1), axis=0) * 100
    mat = mat.reindex(pubs_present)

    fig, ax = plt.subplots(figsize=(10, max(2.5, len(pubs_present) * 0.8)))
    im = ax.imshow(mat.values, aspect="auto", cmap="Blues",
                   vmin=0, vmax=mat.values[~np.isnan(mat.values)].max())
    ax.set_xticks(range(len(top_topics)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(pubs_present)))
    ax.set_yticklabels(pubs_present, fontsize=9)
    plt.colorbar(im, ax=ax, label="% of publication's articles")
    ax.set_title(f"Topics by publication — {corpus_name}")
    fig.tight_layout()
    stem = figures_dir / "fig3b_topics_by_publication"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  → {stem}.pdf")


def fig_topics_by_era(topic_model, corpus_df: pd.DataFrame,
                      figures_dir: Path, corpus_name: str, top_n: int = 10):
    print("[fig] Topics by era …")
    corp = corpus_df[corpus_df["topic_id"] != -1].copy()
    info = topic_model.get_topic_info().set_index("Topic")
    top_topics = corp["topic_id"].value_counts().head(top_n).index.tolist()
    labels = [info.loc[t, "Name"] if t in info.index else f"Topic {t}"
              for t in top_topics]

    era_order = [e[2] for e in ERAS]
    mat = pd.crosstab(corp["era"], corp["topic_id"])
    mat = mat[[t for t in top_topics if t in mat.columns]]
    mat = mat.div(mat.sum(axis=1), axis=0) * 100
    mat = mat.reindex([e for e in era_order if e in mat.index])

    fig, ax = plt.subplots(figsize=(10, max(2.5, len(mat) * 0.7)))
    im = ax.imshow(mat.values, aspect="auto", cmap="Purples",
                   vmin=0, vmax=mat.values[~np.isnan(mat.values)].max())
    ax.set_xticks(range(len(top_topics)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index.tolist(), fontsize=9)
    plt.colorbar(im, ax=ax, label="% of era's articles")
    ax.set_title(f"Topics by government era — {corpus_name}")
    fig.tight_layout()
    stem = figures_dir / "fig3c_topics_by_era"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  → {stem}.pdf")


# ── Step 5: Transform-only — project held-out docs into an existing model ──────

def transform_letters(catalogue: pd.DataFrame,
                      source_label: str = "combined-no-letters-no-aunews",
                      device: str = "auto",
                      embedding_model_path: str = "nomic-ai/nomic-embed-text-v1",
                      batch_size: int = 256,
                      topic_model=None,
                      outlier_strategy: str = "embeddings",
                      outlier_threshold: float = 0.5,
                      noise_sample_size: int = 100):
    """
    Project letters into the topic space of a previously fitted BERTopic model.

    Rather than fitting a new model on the 317 letters, this uses
    BERTopic.transform() so that letters are assigned to the same 51-topic space
    as the main corpus.  This is the correct approach for the letters-vs-editorials
    comparison: both document sets share one topic vocabulary.

    Steps
    -----
    1. Load the saved model from models/{source_label}/bertopic_model/
    2. Load Letters articles from the catalogue and NewsBank cache
    3. Embed them with the same embedding model (fp16)
    4. Call topic_model.transform(docs, embeddings=embeddings)
    5. Reassign any outliers via c-TF-IDF similarity (threshold=0.0)
    6. Save to data/letters/topic_assignments.csv

    Usage
    -----
        python run_bertopic.py --transform-only \\
            --source-model combined-no-letters-no-aunews \\
            --embedding-model /path/to/nomic-embed-text-v1/snapshot
    """
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    import torch

    # ── Load the saved model (skip if already provided) ──────────────────────
    if topic_model is None:
        model_path = str(MODELS_DIR / source_label / "bertopic_model")
        print(f"[transform] Loading model from {model_path} …")
        topic_model = BERTopic.load(model_path)
    print(f"[transform] Model ready — {len(topic_model.get_topic_info()) - 1} topics")

    # ── Load letters ──────────────────────────────────────────────────────────
    print("[transform] Loading letters corpus …")
    if "final_status" in catalogue.columns:
        letters = catalogue[catalogue["final_status"] == "Included-Letter"].copy()
    else:
        included = catalogue[catalogue["relevance"] == "Include"].copy()
        letters  = included[included["content_type"] == "Letters"].copy()
    print(f"[transform] {len(letters):,} letter articles in catalogue")

    # Letters are from Australian papers only (NewsBank)
    letters_nb = letters[~letters["publication"].str.contains("Guardian", na=False)].copy()
    if "body" in letters_nb.columns:
        nb_df = letters_nb.copy().rename(columns={"body": "body_text"})
    else:
        nb_df = load_newsbank_bodies(letters_nb)
    nb_df = nb_df[nb_df["body_text"].notna() & (nb_df["body_text"].str.len() > 50)]
    nb_df["era"] = nb_df["year"].apply(year_to_era)
    print(f"[transform] {len(nb_df):,} letters with usable body text")

    if nb_df.empty:
        print("[transform] No letters found — check catalogue and body cache.")
        return

    docs = nb_df["body_text"].tolist()

    # ── Embed letters ─────────────────────────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[transform] Loading embedding model: {embedding_model_path}  (cache: {_HF_CACHE})")
    embedding_model = SentenceTransformer(
        embedding_model_path, device=device,
        local_files_only=True,
        model_kwargs={"torch_dtype": "float16"},
    )

    # Check for a cached letters embedding (convenient when rerunning)
    letters_embed_cache = MODELS_DIR / source_label / "letters_embeddings_cache.npy"
    embeddings = None
    if letters_embed_cache.exists():
        cached = np.load(str(letters_embed_cache))
        if cached.shape[0] == len(docs):
            print(f"[transform] Loading cached letter embeddings from {letters_embed_cache}")
            embeddings = cached
        else:
            print(f"[transform] Cache shape {cached.shape} doesn't match "
                  f"{len(docs):,} docs — re-embedding")
    if embeddings is None:
        print(f"[transform] Embedding {len(docs):,} letters "
              f"(batch_size={batch_size}, fp16) …")
        embeddings = embedding_model.encode(
            docs, batch_size=batch_size, show_progress_bar=True,
            convert_to_numpy=True,
        )
        np.save(str(letters_embed_cache), embeddings)
        print(f"[transform] Letter embeddings saved to {letters_embed_cache}")

    # ── Project into existing topic space ─────────────────────────────────────
    print("[transform] Projecting letters into topic space …")
    topics, probs = topic_model.transform(docs, embeddings=embeddings)
    # Convert to a plain Python list of ints — some BERTopic versions fail in
    # reduce_outliers when topics is a numpy array of numpy int64 values.
    topics = [int(t) for t in topics]
    n_outliers = sum(1 for t in topics if t == -1)
    print(f"[transform] {n_outliers:,} outliers before reassignment "
          f"({n_outliers/len(topics)*100:.1f}%)")

    if n_outliers > 0:
        topics = reduce_topic_outliers(topic_model, docs, topics,
                                       strategy=outlier_strategy,
                                       threshold=outlier_threshold,
                                       embeddings=embeddings)

    if noise_sample_size > 0:
        data_dir, _, _ = _corpus_dirs("letters")
        sample_noise_cluster(nb_df, topics,
                             n_sample=noise_sample_size,
                             data_dir=data_dir)

    # ── Save assignments ──────────────────────────────────────────────────────
    data_dir, _, _ = _corpus_dirs("letters")
    save_topic_assignments(nb_df, topics, probs, data_dir)
    print(f"[transform] Done. Letter topic assignments → data/letters/topic_assignments.csv")


# ── Step 6: Cross-corpus topic alignment (validation) ─────────────────────────

def align_topics(summary_g: pd.DataFrame, summary_au: pd.DataFrame) -> pd.DataFrame:
    """
    For each Guardian topic, find the best-matching Australian topic by
    Jaccard similarity on top-10 keyword sets. Saves data/topic_alignment.csv.

    A Jaccard score > 0.15 suggests meaningful thematic overlap; > 0.30 is strong.
    """
    print("[align] Computing cross-corpus topic alignment …")

    def kw_set(rep_str):
        """Parse keyword list from BERTopic representation string."""
        if not isinstance(rep_str, str):
            return set()
        # Representation column is usually a list-like string: "['word1', 'word2', ...]"
        import ast
        try:
            items = ast.literal_eval(rep_str)
            return {str(w).lower().strip() for w in items}
        except Exception:
            return {w.strip().strip("'\"[]").lower()
                    for w in rep_str.split(",")}

    rows = []
    g_topics = summary_g[summary_g["topic_id"] != -1].copy()
    au_topics = summary_au[summary_au["topic_id"] != -1].copy()

    for _, g_row in g_topics.iterrows():
        g_kw = kw_set(g_row["keywords"])
        best_j, best_id, best_label, best_au_kw = 0.0, None, "", ""
        for _, au_row in au_topics.iterrows():
            au_kw = kw_set(au_row["keywords"])
            if not g_kw or not au_kw:
                continue
            j = len(g_kw & au_kw) / len(g_kw | au_kw)
            if j > best_j:
                best_j, best_id = j, au_row["topic_id"]
                best_label = au_row["label"]
                best_au_kw = au_row["keywords"]
        rows.append({
            "guardian_topic_id":    g_row["topic_id"],
            "guardian_label":       g_row["label"],
            "guardian_count":       g_row["count"],
            "guardian_keywords":    g_row["keywords"],
            "au_topic_id":          best_id,
            "au_label":             best_label,
            "au_keywords":          best_au_kw,
            "jaccard_similarity":   round(best_j, 4),
            "match_strength":       ("strong" if best_j >= 0.30
                                     else "moderate" if best_j >= 0.15
                                     else "weak"),
        })

    df_align = pd.DataFrame(rows).sort_values("jaccard_similarity", ascending=False)
    path = DATA_DIR / "topic_alignment.csv"
    df_align.to_csv(path, index=False)
    print(f"[align] Alignment table → {path}")
    strong = (df_align["match_strength"] == "strong").sum()
    moderate = (df_align["match_strength"] == "moderate").sum()
    print(f"  Strong matches (J≥0.30): {strong}  |  Moderate (J≥0.15): {moderate}")
    return df_align


# ── Run one corpus ────────────────────────────────────────────────────────────

def run_corpus(catalogue: pd.DataFrame, corpus_name: str,
               min_topic_size: int, nr_topics, device: str,
               exclude_letters: bool = False,
               exclude_au_news: bool = False,
               embedding_model_path: str = "nomic-ai/nomic-embed-text-v1",
               batch_size: int = 256,
               outlier_strategy: str = "embeddings",
               outlier_threshold: float = 0.5,
               noise_sample_size: int = 100) -> pd.DataFrame:
    """Fit BERTopic on one corpus partition and save all outputs."""
    label = corpus_name
    if exclude_letters: label += "-no-letters"
    if exclude_au_news: label += "-no-aunews"
    print(f"\n{'='*60}")
    print(f"  Running BERTopic: {corpus_name.upper()} corpus"
          f"{' (letters excluded)' if exclude_letters else ''}"
          f"{' (AU news excluded)' if exclude_au_news else ''}")
    print(f"{'='*60}")

    data_dir, figures_dir, models_dir = _corpus_dirs(label)

    # ── Stage 1: load text ────────────────────────────────────────────────────
    corpus_df = build_text_corpus(catalogue, corpus=corpus_name,
                                  exclude_letters=exclude_letters,
                                  exclude_au_news=exclude_au_news)
    docs = corpus_df["body_text"].tolist()
    if not docs:
        raise ValueError(f"No documents loaded for corpus '{corpus_name}'.")

    # ── Stage 2: fit BERTopic (returns raw topics with -1 noise intact) ───────
    topic_model, topics, probs, embeddings = fit_bertopic(
        docs, min_topic_size=min_topic_size, nr_topics=nr_topics, device=device,
        embedding_model_path=embedding_model_path, batch_size=batch_size,
    )

    # ── Stage 2b: reduce outliers with configurable threshold ─────────────────
    topics = reduce_topic_outliers(topic_model, docs, topics,
                                   strategy=outlier_strategy,
                                   threshold=outlier_threshold,
                                   embeddings=embeddings)

    # ── Stage 2c: inspect remaining noise cluster ─────────────────────────────
    if noise_sample_size > 0:
        sample_noise_cluster(corpus_df, topics,
                             n_sample=noise_sample_size,
                             data_dir=data_dir)

    # ── Stage 3: save ─────────────────────────────────────────────────────────
    corpus_df = save_topic_assignments(corpus_df, topics, probs, data_dir)
    summary = save_topic_summary(topic_model, corpus_df, data_dir)
    save_model(topic_model, models_dir)

    # ── Stage 4: figures ──────────────────────────────────────────────────────
    fig_topics_over_time(topic_model, corpus_df, figures_dir, corpus_name)
    fig_topics_by_publication(topic_model, corpus_df, figures_dir, corpus_name)
    fig_topics_by_era(topic_model, corpus_df, figures_dir, corpus_name)

    return summary, topic_model


# ── Main ──────────────────────────────────────────────────────────────────────

def main(corpus: str = "both", min_topic_size: int = 50,
         au_min_topic_size: int = 20, nr_topics=None, device: str = "auto",
         exclude_letters: bool = False, exclude_au_news: bool = False,
         embedding_model_path: str = "nomic-ai/nomic-embed-text-v1",
         batch_size: int = 256,
         transform_only: bool = False,
         source_model: str = "combined-no-letters-no-aunews",
         outlier_strategy: str = "embeddings",
         outlier_threshold: float = 0.5,
         noise_sample_size: int = 100):

    cat_path = SCORED_CSV if SCORED_CSV.exists() else CATALOGUE_CSV
    print(f"[load] Catalogue: {cat_path}")
    catalogue = pd.read_csv(cat_path, low_memory=False)
    DATA_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # Transform-only mode: project letters into an existing model's topic space
    if transform_only:
        transform_letters(catalogue,
                          source_label=source_model,
                          device=device,
                          embedding_model_path=embedding_model_path,
                          batch_size=batch_size,
                          outlier_strategy=outlier_strategy,
                          outlier_threshold=outlier_threshold,
                          noise_sample_size=noise_sample_size)
        return

    if corpus == "combined":
        # Single model: Guardian opinion+analysis + all Australian opinion/analysis
        summary, topic_model = run_corpus(
            catalogue, "combined",
            min_topic_size, nr_topics, device,
            exclude_letters=exclude_letters,
            exclude_au_news=exclude_au_news,
            embedding_model_path=embedding_model_path,
            batch_size=batch_size,
            outlier_strategy=outlier_strategy,
            outlier_threshold=outlier_threshold,
            noise_sample_size=noise_sample_size,
        )
        _label = "combined"
        if exclude_letters: _label += "-no-letters"
        if exclude_au_news: _label += "-no-aunews"

        # Free the embedding model from GPU before loading a second instance
        # for letters encoding — both would otherwise sit on GPU simultaneously.
        # topic_model.embedding_model is a BERTopic SentenceTransformerBackend wrapper;
        # the actual torch model lives one level deeper at .embedding_model.
        import torch
        if hasattr(topic_model, "embedding_model") and topic_model.embedding_model is not None:
            inner = getattr(topic_model.embedding_model, "embedding_model", None)
            if inner is not None and hasattr(inner, "to"):
                inner.to("cpu")
            topic_model.embedding_model = None
        torch.cuda.empty_cache()
        print("[memory] Embedding model offloaded from GPU before letters transform.")

        transform_letters(catalogue,
                          source_label=_label,
                          device=device,
                          embedding_model_path=embedding_model_path,
                          batch_size=batch_size,
                          topic_model=topic_model,
                          outlier_strategy=outlier_strategy,
                          outlier_threshold=outlier_threshold,
                          noise_sample_size=noise_sample_size)
        print("\n[done] Combined pipeline complete.")
        print(f"  Editorials → data/combined/   |   Letters → data/letters/")
        return

    if corpus in ("guardian", "both"):
        summary_g, _ = run_corpus(catalogue, "guardian",
                                  min_topic_size, nr_topics, device,
                                  exclude_au_news=exclude_au_news,
                                  embedding_model_path=embedding_model_path,
                                  batch_size=batch_size,
                                  outlier_strategy=outlier_strategy,
                                  outlier_threshold=outlier_threshold,
                                  noise_sample_size=noise_sample_size)

    if corpus in ("australian", "both"):
        summary_au, _ = run_corpus(catalogue, "australian",
                                   au_min_topic_size, nr_topics, device,
                                   exclude_letters=exclude_letters,
                                   embedding_model_path=embedding_model_path,
                                   batch_size=batch_size,
                                   outlier_strategy=outlier_strategy,
                                   outlier_threshold=outlier_threshold,
                                   noise_sample_size=noise_sample_size)

    if corpus == "both":
        align_topics(summary_g, summary_au)

    print("\n[done] Pipeline complete.")
    print(f"  Outputs in data/guardian/ and data/australian/")
    print(f"  Alignment → data/topic_alignment.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BERTopic pipeline — Guardian vs Australian papers"
    )
    parser.add_argument(
        "--corpus", choices=["guardian", "australian", "both", "combined"],
        default="both",
        help="Which corpus to run. 'combined' fits one model on all opinion/analysis.",
    )
    parser.add_argument(
        "--exclude-au-news", action="store_true",
        help="Exclude Guardian Australia news section (keep Opinion + Environment only)",
    )
    parser.add_argument(
        "--min-topic-size", type=int, default=50, metavar="N",
        help="Min cluster size for Guardian corpus (default: 50)",
    )
    parser.add_argument(
        "--au-min-topic-size", type=int, default=20, metavar="N",
        help="Min cluster size for Australian corpus (default: 20)",
    )
    parser.add_argument(
        "--nr-topics", type=int, default=None, metavar="N",
        help="Reduce to N topics after fitting (default: auto)",
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Embedding device (default: auto-detect)",
    )
    parser.add_argument(
        "--exclude-letters", action="store_true",
        help="Exclude Letters content_type from topic modelling (analyse separately)",
    )
    parser.add_argument(
        "--embedding-model",
        default="nomic-ai/nomic-embed-text-v1",
        metavar="PATH_OR_NAME",
        help=(
            "Sentence-Transformers model for embeddings.  "
            "Default: nomic-ai/nomic-embed-text-v1 (8192-token context, trust_remote_code=True).  "
            "Pass a HuggingFace model ID OR the absolute path to a local snapshot directory.  "
            "When a local path is given the script sets HF_HUB_OFFLINE=1 "
            "automatically so no network access is attempted."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, metavar="N",
        help="Embedding batch size (default: 256). Reduce if OOM; raise if GPU has headroom.",
    )
    parser.add_argument(
        "--transform-only", action="store_true",
        help=(
            "Project letters into an existing fitted model's topic space using "
            "BERTopic.transform() rather than fitting a new model. "
            "Outputs to data/letters/topic_assignments.csv. "
            "Use --source-model to specify which saved model to load."
        ),
    )
    parser.add_argument(
        "--source-model", default="combined-no-letters-no-aunews", metavar="LABEL",
        help=(
            "Label of the saved model to load for --transform-only mode. "
            "Must match the folder name under models/. "
            "Default: combined-no-letters-no-aunews"
        ),
    )
    parser.add_argument(
        "--outlier-strategy", default="embeddings",
        choices=["embeddings", "c-tf-idf"],
        help=(
            "Strategy for reduce_outliers. 'embeddings' uses cosine similarity "
            "in the full embedding space (recommended); 'c-tf-idf' uses vocabulary "
            "overlap (fast but scores rarely exceed 0.1). Default: embeddings"
        ),
    )
    parser.add_argument(
        "--outlier-threshold", type=float, default=0.5, metavar="T",
        help=(
            "Minimum c-TF-IDF similarity required to reassign a noise-cluster "
            "document to a topic.  0.0 reassigns everything; 1.0 reassigns nothing. "
            "Default: 0.5"
        ),
    )
    parser.add_argument(
        "--noise-sample-size", type=int, default=100, metavar="N",
        help=(
            "Number of remaining noise-cluster articles to sample and save to "
            "noise_sample.csv for manual inspection. 0 disables this step. "
            "Default: 100"
        ),
    )
    args = parser.parse_args()

    main(
        corpus=args.corpus,
        min_topic_size=args.min_topic_size,
        au_min_topic_size=args.au_min_topic_size,
        nr_topics=args.nr_topics,
        device=args.device,
        exclude_letters=args.exclude_letters,
        exclude_au_news=args.exclude_au_news,
        embedding_model_path=args.embedding_model,
        batch_size=args.batch_size,
        transform_only=args.transform_only,
        source_model=args.source_model,
        outlier_strategy=args.outlier_strategy,
        outlier_threshold=args.outlier_threshold,
        noise_sample_size=args.noise_sample_size,
    )
