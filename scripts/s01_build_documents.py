import pandas as pd
import json
import re
from collections import Counter


# Configuration
INPUT_CSV  = "data/equipment_database.csv"
OUTPUT_JSONL = "data/documents.jsonl"
SEP = " | "

KEEP_COLS = {
    "Equipment name":                        "name",
    "Aliases / synonyms":                    "aliases",
    "Short description (2-3 sentences)":     "description",
    "Typical applications (3+ bullets)":     "applications",
    "Tags (5-10 keywords)":                  "tags",
    "Confidence":                            "confidence",
}


# Helper functions for cleaning and building the document string
def clean(text: str) -> str:
    """Strip whitespace and collapse internal newlines/bullets to a single line."""
    if not text or pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s*[-•]\s*", ", ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r",\s*,", ",", text)
    return text.strip().strip(",").strip()


def build_document(row: pd.Series) -> str:
    """Concatenate selected fields into a single retrieval document string."""
    parts = []
    for csv_col, _ in KEEP_COLS.items():
        value = clean(row.get(csv_col, ""))
        if value:
            parts.append(value)
    return SEP.join(parts)


# Main
def main():
    confidence_counter = [] # check ratio of High and medium confidence
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    records = []
    for idx, row in df.iterrows():
        name = str(row.get("Equipment name", f"item_{idx}")).strip()
        confidence = str(row.get("Confidence"))
        confidence_counter.append(confidence)

        document = build_document(row)

        # Keep structured fields separately for transparency / ablation studies
        structured = {}
        for csv_col, short_name in KEEP_COLS.items():
            val = clean(row.get(csv_col, ""))
            if val:
                structured[short_name] = val

        record = {
            "id":       f"eq_{idx:04d}",
            "name":     name,
            "document": document,
            # "fields":   structured, # Optional: keep structured fields for transparency / ablation studies
            # "queries":  []
        }
        records.append(record)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Count ratio of High and Medium confidence
    # print(Counter(confidence_counter)) # Outcome: 'High': 8587, 'Medium': 14

    print(f"Written {len(records)} records to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()