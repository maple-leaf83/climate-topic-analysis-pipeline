"""
fetch_guardian.py
Retrieves climate-related articles from the Guardian Open Platform API and
saves them to data/guardian_articles.csv.

Usage:
    python fetch_guardian.py

Requirements:
    pip install requests pandas tqdm
    Set GUARDIAN_API_KEY in config.py before running.
"""

import time
import requests
import pandas as pd
from tqdm import tqdm
from config import (
    GUARDIAN_API_KEY, GUARDIAN_SECTIONS, GUARDIAN_QUERIES,
    GUARDIAN_FROM, GUARDIAN_TO, GUARDIAN_CSV, DATA_DIR,
)

DATA_DIR.mkdir(exist_ok=True)

BASE_URL   = "https://content.guardianapis.com/search"
PAGE_SIZE  = 200
RATE_LIMIT = 0.2   # seconds between requests (API allows 12/s for free tier)


def fetch_section(query: str, section: str) -> list[dict]:
    """Fetch all pages for a single query × section combination."""
    records, page, total_pages = [], 1, 1
    params = {
        "api-key":      GUARDIAN_API_KEY,
        "q":            query,
        "section":      section,
        "from-date":    GUARDIAN_FROM,
        "to-date":      GUARDIAN_TO,
        "page-size":    PAGE_SIZE,
        "show-fields":  "bodyText,wordcount,byline",
        "order-by":     "oldest",
    }

    while page <= total_pages:
        params["page"] = page
        resp = requests.get(BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()["response"]
        total_pages = data["pages"]

        for item in data["results"]:
            fields = item.get("fields", {})
            records.append({
                "id":         item["id"],
                "title":      item["webTitle"],
                "date":       item["webPublicationDate"][:10],
                "section":    item["sectionName"],
                "author":     fields.get("byline", ""),
                "word_count": fields.get("wordcount", None),
                "body":       fields.get("bodyText", ""),
                "publication":"The Guardian",
                "url":        item["webUrl"],
            })
        page += 1
        time.sleep(RATE_LIMIT)

    return records


def main():
    all_records = []
    combos = [(q, s) for q in GUARDIAN_QUERIES for s in GUARDIAN_SECTIONS]

    for query, section in tqdm(combos, desc="Fetching Guardian"):
        try:
            recs = fetch_section(query, section)
            all_records.extend(recs)
            print(f"  {query!r} / {section}: {len(recs)} articles")
        except requests.HTTPError as e:
            print(f"  ERROR {query!r} / {section}: {e}")

    df = pd.DataFrame(all_records)
    before = len(df)
    df = df.drop_duplicates(subset=["id"])
    print(f"\nFetched {before} total → {len(df)} after deduplication")
    df.to_csv(GUARDIAN_CSV, index=False)
    print(f"Saved → {GUARDIAN_CSV}")


if __name__ == "__main__":
    main()
