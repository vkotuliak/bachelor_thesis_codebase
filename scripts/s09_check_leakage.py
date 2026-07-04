"""
Script to check lexical overlap between synthetic queries and equipment 
descriptions.
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import scientific_tokenizer


def calculate_leakage(desc_file, query_file):
    descriptions = {}
    with open(desc_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            eq_id = data["id"]

            desc_text = data.get("equipment_description", "")
            descriptions[eq_id] = set(scientific_tokenizer(desc_text))

    total_queries = 0
    running_leakage_sum = 0.0

    print(
        f"{'Equipment ID':<15} | {'Query Index':<12} | {'Overlap %':<10} | {'Leaked Stems Sample'}"
    )
    print("-" * 75)

    with open(query_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            eq_id = data["id"]
            queries = data.get("query", [])

            if eq_id not in descriptions:
                continue

            desc_tokens = descriptions[eq_id]

            for idx, query_text in enumerate(queries):
                query_tokens = set(scientific_tokenizer(query_text))

                if not query_tokens:
                    continue

                leaked_tokens = query_tokens.intersection(desc_tokens)

                overlap_ratio = len(leaked_tokens) / len(query_tokens)
                running_leakage_sum += overlap_ratio
                total_queries += 1

                leaked_sample = ", ".join(list(leaked_tokens)[:5])
                print(
                    f"{eq_id:<15} | Query #{idx:<6} | {overlap_ratio*100:>5.1f}%   | {leaked_sample}"
                )

    if total_queries > 0:
        average_leakage = (running_leakage_sum / total_queries) * 100
        print("-" * 75)
        print(f"Total Queries Analyzed: {total_queries}")
        print(f"Average Stem-Level Leakage: {average_leakage:.2f}%")
    else:
        print("No matching queries/descriptions found.")


if __name__ == "__main__":
    DESC_FILE = "data/full_data/rag_documents.jsonl"
    QUERY_FILE = "data/test_data/queries_from_personas.jsonl"

    calculate_leakage(DESC_FILE, QUERY_FILE)
