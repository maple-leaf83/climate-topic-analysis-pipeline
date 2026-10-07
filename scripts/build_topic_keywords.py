"""
build_topic_keywords.py
───────────────────────
Extracts the top-N c-TF-IDF keywords for every BERTopic topic in the
combined corpus and writes them to a formatted Excel workbook.

Sources:
  models/bertopic_model/ctfidf.safetensors
      Sparse CSR matrix (67 topics × 193,756 vocab terms).
      Row i in the matrix corresponds to topic_id = i - 1
        (row 0 = outlier topic -1 / noise; row 1 = topic 0, etc.)
  models/bertopic_model/ctfidf_config.json
      Contains the CountVectorizer vocabulary (word → column index).
  data/combined/topic_summary.csv
      topic_id, label, count, Group — used for display names and grouping.

Output:
  data/combined/topic_keywords.xlsx
      Sheet "All Topics"  — every topic, sorted by group then topic_id
      One sheet per thematic group (noise/off-topic excluded)
      Columns: Group | Topic ID | Description | Articles (n) |
               KW 1 | Score 1 | … | KW N | Score N
"""

import json
import struct
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from openpyxl import Workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, numbers
)
from openpyxl.utils import get_column_letter

# ── Paths ────────────────────────────────────────────────────────────────────
from config import DATA_DIR, MODELS_DIR

MODEL_DIR   = MODELS_DIR / "combined" / "bertopic_model"
CTFIDF_PATH = MODEL_DIR / "ctfidf.safetensors"
CFG_PATH    = MODEL_DIR / "ctfidf_config.json"
SUMMARY_CSV = DATA_DIR  / "combined" / "topic_summary.csv"
OUT_XLSX    = DATA_DIR  / "combined" / "topic_keywords.xlsx"

TOP_N       = 20          # keywords per topic
NOISE_GROUP = "Noise"     # excluded from per-group sheets (kept in All Topics)

# ── Load vocabulary (word → column index) ────────────────────────────────────
with open(CFG_PATH) as f:
    ctcfg = json.load(f)

vocab    = ctcfg["vectorizer_model"]["vocab"]          # word -> col_index
idx2word = {v: k for k, v in vocab.items()}            # col_index -> word
print(f"Vocabulary size: {len(vocab):,}")

# ── Parse c-TF-IDF safetensors (CSR sparse matrix) ───────────────────────────
# Binary layout: 8-byte header length, then JSON header, then raw tensor data.
# Tensors: shape (I64), data (F64, non-zero scores), indices (I32, col indices),
#          indptr (I32, row pointers), diag (F64, IDF diagonal — not used here).

def read_tensor(f, meta, data_start):
    """Read one tensor from an open safetensors file."""
    dtype_map = {"F64": np.float64, "I32": np.int32, "I64": np.int64}
    start, end = meta["data_offsets"]
    f.seek(data_start + start)
    raw = f.read(end - start)
    return np.frombuffer(raw, dtype=dtype_map[meta["dtype"]])

with open(CTFIDF_PATH, "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    hdr  = json.loads(f.read(hlen))
    ds   = 8 + hlen   # byte offset where tensor data begins

    data    = read_tensor(f, hdr["data"],    ds)   # non-zero c-TF-IDF scores
    indices = read_tensor(f, hdr["indices"], ds)   # column indices
    indptr  = read_tensor(f, hdr["indptr"],  ds)   # row pointers (len = n_rows + 1)

n_rows = len(indptr) - 1   # = 67 (rows 0–66, one row per topic_id 0–66)
print(f"c-TF-IDF matrix: {n_rows} rows × {len(idx2word):,} cols  "
      f"({len(data):,} non-zero values)")
# Note: row i = topic_id i directly (noise topic -1 is NOT in the matrix).
# Any topic_id ≥ n_rows (e.g. topic 67 if it was added after training) falls
# back to the pre-computed top-10 from topics.json.

with open(MODEL_DIR / "topics.json") as f:
    topics_json = json.load(f)
fallback_kws = topics_json.get("topic_representations", {})  # str(topic_id) -> [[word, score], …]

def top_keywords(topic_id, n=TOP_N):
    """Return [(word, score), …] for the top-n c-TF-IDF terms of topic_id.
    Falls back to topics.json pre-computed keywords if topic is out of matrix range."""
    if topic_id < n_rows:
        row      = topic_id                          # direct row mapping
        s, e     = int(indptr[row]), int(indptr[row + 1])
        if s < e:
            row_data = data[s:e]
            row_cols = indices[s:e]
            top_idx  = np.argsort(row_data)[::-1][:n]
            return [(idx2word.get(int(row_cols[i]), "?"), round(float(row_data[i]), 5))
                    for i in top_idx]
    # Fallback: use topics.json (top-10 only)
    fb = fallback_kws.get(str(topic_id), [])
    return [(w, round(s, 5)) for w, s in fb[:n]]

# ── Load topic summary ────────────────────────────────────────────────────────
summary = pd.read_csv(SUMMARY_CSV)
summary = summary.rename(columns={"Group": "group"})   # normalise column name
summary = summary.sort_values(["group", "topic_id"]).reset_index(drop=True)

groups      = [g for g in summary["group"].unique() if g != NOISE_GROUP]
all_topics  = summary[summary["topic_id"] >= 0].copy()  # exclude noise topic -1
print(f"Topics to export: {len(all_topics)} across {len(groups)} groups")

# ── Build rows ────────────────────────────────────────────────────────────────
def make_rows(topic_df):
    rows = []
    for _, row in topic_df.iterrows():
        tid    = int(row["topic_id"])
        kws    = top_keywords(tid, TOP_N)
        entry  = {
            "Group":        row["group"],
            "Topic ID":     f"T{tid:02d}",
            "Description":  row["label"].split("_", 1)[-1].replace("_", " ")
                            if "_" in str(row["label"]) else str(row["label"]),
            "Articles (n)": int(row["count"]),
        }
        for rank, (word, score) in enumerate(kws, 1):
            entry[f"KW {rank}"]    = word
            entry[f"Score {rank}"] = score
        rows.append(entry)
    return rows

# ── Excel styling helpers ─────────────────────────────────────────────────────
# Assign a distinct fill to each group
GROUP_COLORS = [
    "D6E4F0", "D5F5E3", "FDEBD0", "F9EBEA", "EBF5FB",
    "FDF2E9", "F4ECF7", "E8F8F5", "FEF9E7", "EAEDED",
]
group_color = {g: GROUP_COLORS[i % len(GROUP_COLORS)]
               for i, g in enumerate(groups)}

def header_fill(hex_color):
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    # Darken by ~30% for header
    dr = max(0, int(r * 0.7))
    dg = max(0, int(g * 0.7))
    db = max(0, int(b * 0.7))
    return f"{dr:02X}{dg:02X}{db:02X}"

THIN_BORDER = Border(
    left=Side(style="thin", color="CCCCCC"),
    right=Side(style="thin", color="CCCCCC"),
    top=Side(style="thin", color="CCCCCC"),
    bottom=Side(style="thin", color="CCCCCC"),
)

def style_sheet(ws, rows_data, fill_color):
    """Write rows_data to ws with header + alternating row colours."""
    if not rows_data:
        return
    cols = list(rows_data[0].keys())
    hdr_fill = header_fill(fill_color)

    # Header row
    for col_idx, col_name in enumerate(cols, 1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font      = Font(bold=True, color="FFFFFF", size=9, name="Arial")
        cell.fill      = PatternFill("solid", fgColor=hdr_fill)
        cell.alignment = Alignment(horizontal="center", vertical="center",
                                   wrap_text=True)
        cell.border    = THIN_BORDER

    # Data rows
    for row_idx, row_data in enumerate(rows_data, 2):
        alt = row_idx % 2 == 0
        row_fill = PatternFill("solid", fgColor=(fill_color if alt else "FFFFFF"))
        for col_idx, col_name in enumerate(cols, 1):
            val  = row_data.get(col_name, "")
            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            cell.fill      = row_fill
            cell.border    = THIN_BORDER
            cell.font      = Font(size=9, name="Arial")
            # Score columns: 4 decimal places
            if col_name.startswith("Score"):
                cell.number_format = "0.0000"
                cell.alignment = Alignment(horizontal="right")
            elif col_name in ("Topic ID", "Articles (n)"):
                cell.alignment = Alignment(horizontal="center")
            else:
                cell.alignment = Alignment(horizontal="left", wrap_text=False)

    # Column widths
    col_widths = {
        "Group": 28, "Topic ID": 8, "Description": 32, "Articles (n)": 10,
    }
    for col_idx, col_name in enumerate(cols, 1):
        ltr = get_column_letter(col_idx)
        if col_name in col_widths:
            ws.column_dimensions[ltr].width = col_widths[col_name]
        elif col_name.startswith("KW"):
            ws.column_dimensions[ltr].width = 18
        elif col_name.startswith("Score"):
            ws.column_dimensions[ltr].width = 8

    ws.freeze_panes = "A2"
    ws.row_dimensions[1].height = 30

# ── Build workbook ────────────────────────────────────────────────────────────
wb = Workbook()

# --- "All Topics" sheet (all non-noise topics) ---
ws_all = wb.active
ws_all.title = "All Topics"
all_rows = make_rows(all_topics)
style_sheet(ws_all, all_rows, "D0D0D0")

# --- One sheet per group ---
for group in groups:
    group_topics = summary[summary["group"] == group].copy()
    short_name   = group[:31]   # Excel sheet name limit = 31 chars
    ws = wb.create_sheet(title=short_name)
    rows = make_rows(group_topics)
    style_sheet(ws, rows, group_color.get(group, "E8E8E8"))
    print(f"  {short_name}: {len(rows)} topics")

wb.save(OUT_XLSX)
print(f"\nSaved → {OUT_XLSX}")
print(f"Sheets: {[ws.title for ws in wb.worksheets]}")
