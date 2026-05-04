import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

QUERIES_PATH = "data/test_data/rag_ready_w_queries.jsonl"
CORPUS_PATH = "data/full_data/rag_documents.jsonl"
INDEX_PATH = "data/dense/faiss.index"


def load_corpus(path):
    corpus = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            corpus.append(item["equipment_description"])
    return corpus


def load_queries(path: str) -> list[tuple]:
    pairs = []
    with open(path) as f:
        for doc_id, line in enumerate(f):
            item = json.loads(line)
            for query in item["queries"]:
                pairs.append((query, doc_id))
    return pairs


# Evaluation metrics
def recall_at_k(ranked_indices: list[int], relevant_idx: int, k: int) -> float:
    return 1.0 if relevant_idx in ranked_indices[:k] else 0.0


def reciprocal_rank(ranked_indices: list[int], relevant_idx: int) -> float:
    for rank, idx in enumerate(ranked_indices, start=1):
        if idx == relevant_idx:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_indices: list[int], relevant_idx: int, k: int) -> float:
    if relevant_idx in ranked_indices[:k]:
        return 1.0 / np.log2(ranked_indices[:k].index(relevant_idx) + 2)
    return 0.0


def evaluate(
    corpus: list[str], queries: list[tuple], ks: list[int] = [1, 5, 10]
):
    model = SentenceTransformer("all-mpnet-base-v2")
    index = faiss.read_index(INDEX_PATH)

    recall_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}
    rr_scores = []

    for query, relevant_idx in queries:
        query_embedding = model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        _, indices = index.search(query_embedding, max(ks))
        ranked_indices = indices[0].tolist()

        for k in ks:
            recall_scores[k].append(
                recall_at_k(ranked_indices, relevant_idx, k)
            )
            ndcg_scores[k].append(ndcg_at_k(ranked_indices, relevant_idx, k))
        rr_scores.append(reciprocal_rank(ranked_indices, relevant_idx))

    print(f"Evaluated {len(queries)} queries over {len(corpus)} documents\n")
    print(f"MRR: {sum(rr_scores) / len(rr_scores):.4f}\n")
    for k in ks:
        print(
            f"Recall@{k}: {sum(recall_scores[k]) / len(recall_scores[k]):.4f}"
        )
    print()
    for k in ks:
        print(f"nDCG@{k}: {sum(ndcg_scores[k]) / len(ndcg_scores[k]):.4f}")

    return recall_scores, ndcg_scores, rr_scores


def main():
    corpus = load_corpus(CORPUS_PATH)
    queries = load_queries(QUERIES_PATH)
    evaluate(corpus, queries)


if __name__ == "__main__":
    main()
