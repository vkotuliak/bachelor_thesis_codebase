import numpy as np
from rank_bm25 import BM25Okapi
import json

QUERIES_PATH = "data/test_data/rag_ready_w_queries.jsonl"
CORPUS_PATH = "data/full_data/rag_documents.jsonl"


# Original function, which evaluates using only the 50 equipments that have queries
def load_qrels(path):
    """Loads the evaluation data from a JSONL file, returning the corpus and query-relevance pairs."""
    corpus = []
    qrels = []  # query, relevant_doc_idx

    with open(path) as f:
        for line in f:
            item = json.loads(line)
            doc_id = item["id"]
            corpus.append(item["equipment_description"])
            for query in item["queries"]:
                qrels.append((query, doc_id))
    return corpus, qrels


# New functions, which evaluate using the whole corpus
def load_corpus(path):
    corpus = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            corpus.append(item["equipment_description"])
    return corpus


def load_queries(path):
    pairs = []
    with open(path) as f:
        for doc_id, line in enumerate(f):
            item = json.loads(line)
            for query in item["queries"]:
                pairs.append((query, doc_id))
    return pairs


def build_bm25(corpus: list[str]) -> BM25Okapi:
    """Builds a BM25 index from the given corpus."""
    tokenized = [doc.lower().split() for doc in corpus]
    return BM25Okapi(tokenized)


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
    corpus: list[str], qrels: list[tuple], ks: list[int] = [1, 5, 10]
):
    """Evaluates the BM25 model on the given corpus and query-relevance pairs, returning recall and MRR scores."""
    bm25 = build_bm25(corpus)

    recall_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}
    rr_scores = []

    for query, relevant_idx in qrels:
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        for k in ks:
            recall_scores[k].append(
                recall_at_k(ranked_indices, relevant_idx, k)
            )
            ndcg_scores[k].append(ndcg_at_k(ranked_indices, relevant_idx, k))
        rr_scores.append(reciprocal_rank(ranked_indices, relevant_idx))

    print(f"Evaluated {len(qrels)} queries over {len(corpus)} documents\n")
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
    # corpus, qrels = load_qrels(QUERIES_PATH)
    corpus = load_corpus(CORPUS_PATH)
    qrels = load_queries(QUERIES_PATH)
    evaluate(corpus, qrels)


if __name__ == "__main__":
    main()
