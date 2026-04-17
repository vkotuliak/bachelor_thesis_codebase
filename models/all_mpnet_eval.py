import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

QUERIES_PATH = "data/test_data/rag_ready_w_queries.jsonl"
CORPUS_PATH = "data/full_data/rag_documents.jsonl"
INDEX_PATH = "data/dense/faiss.index"
K = 10

model = SentenceTransformer("all-mpnet-base-v2")
index = faiss.read_index(INDEX_PATH)

def load_corpus(path):
    corpus = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            corpus.append(item["equipment_description"])
    return corpus

def extract_name(description: str) -> str:
    return description.split("[SEP]")[0].replace("name:", "").strip()

corpus = load_corpus(CORPUS_PATH)
corpus_names = [extract_name(d) for d in corpus]

# --- Load queries ---
def load_queries(path):
    pairs = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            gt_name = extract_name(item["equipment_description"])
            for query in item["queries"]:
                pairs.append((gt_name, query))
    return pairs

query_pairs = load_queries(QUERIES_PATH)
print(f"Loaded {len(query_pairs)} queries over {len(corpus)} equipment items.")

# --- Evaluate ---
hits = {1: 0, 3: 0, 5: 0, 10: 0}
cutoffs = sorted(hits.keys())
reciprocal_ranks = []  # collect 1/rank for each query (0 if not found in top-K)

for gt_name, query in query_pairs:
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    _, indices = index.search(query_embedding, K)

    retrieved_names = [corpus_names[i] for i in indices[0]]

    # Recall@k
    for k in cutoffs:
        if gt_name in retrieved_names[:k]:
            hits[k] += 1

    # MRR: find the rank of the correct result (1-indexed)
    if gt_name in retrieved_names:
        rank = retrieved_names.index(gt_name) + 1  # e.g. first place → rank 1
        reciprocal_ranks.append(1 / rank)
    else:
        reciprocal_ranks.append(0)  # not found in top-K, contributes 0

# --- Report ---
total = len(query_pairs)
mrr = sum(reciprocal_ranks) / total

print(f"\nResults on {total} queries:\n")
for k in cutoffs:
    recall = hits[k] / total
    print(f"  Recall@{k:<3} = {recall:.3f}  ({hits[k]}/{total})")
print(f"  MRR@{K:<4} = {mrr:.3f}")