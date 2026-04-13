"""
This file takes the raw csv data from `equipment_database.csv` and transforms 
them into jsonl file. 

It cleans the text using clean() function, and then it takes Name, Aliases, 
short description, typical applications and tags and puts them into the 
`documents.jsonl` file. 
"""

import pandas as pd
import json
import re


# Configuration
INPUT_CSV = "data/equipment_database.csv"
OUTPUT_JSONL = "data/documents.jsonl"


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


# Main
def main():
    df = pd.read_csv(INPUT_CSV)
    records = []
    
    for idx, row in df.iterrows():
        name = str(row.get("Equipment name", f"item_{idx}")).strip()

        aliases = clean(str(row.get("Aliases / synonyms")).strip())
        description = clean(
            str(row.get("Short description (2-3 sentences)")).strip()
        )
        applications = clean(
            str(row.get("Typical applications (3+ bullets)")).strip()
        )
        tags = clean(str(row.get("Tags (5-10 keywords)")).strip())

        confidence = str(row.get("Confidence")).strip()
        # Only take rows with high confidence into consideration (filter out 14 rows with medium)
        if confidence != "High":
            continue

        record = {
            "id": f"eq_{idx:04d}",
            "name": name,
            "aliases": aliases,
            "short description": description,
            "typical applications": applications,
            "tags": tags,
            "queries": [],
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
