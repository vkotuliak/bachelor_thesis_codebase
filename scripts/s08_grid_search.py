import pandas as pd
import sys
from pathlib import Path

# Adds the parent directory of the current file to the search path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import evaluation
import utils

test_ks = [5, 10, 30, 60]
test_alphas = [0.2, 0.4, 0.5, 0.6, 0.8]
DEFAULT_QUERIES_PATH = "data/test_data/f100_queries.jsonl"

print("Loading corpus and queries...")
corpus = utils.load_corpus("data/full_data/rag_documents.jsonl")
queries = evaluation.load_queries(DEFAULT_QUERIES_PATH)

results_list = []
total_iterations = len(test_alphas) * len(test_ks)

print(f"Starting grid search over {total_iterations} configurations...")
for alpha in test_alphas:
    for k in test_ks:
        ndcg = evaluation.evaluate_hybrid(
            corpus[0],
            corpus[1],
            queries,
            [10],
            rrf_k=k,
            top_k=500,
            alpha=alpha,
        )

        results_list.append(
            {
                "alpha_e5": alpha,
                "alpha_bm25": round(1.0 - alpha, 1),
                "rrf_k": k,
                "ndcg": ndcg,
            }
        )

df_results = pd.DataFrame(results_list)
df_results = df_results.sort_values(by="ndcg", ascending=False).reset_index(
    drop=True
)

print("\n=== Grid Search Results (Sorted by Best nDCG) ===")
print(df_results.to_string(index=False))
