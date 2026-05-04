"""
evaluation.py  –  Unified retrieval evaluation for BM25, all-mpnet-base-v2, and E5-large-v2.

Usage examples
--------------
  # Run all three models
  python eval.py --models bm25 mpnet e5

  # Run only BM25 and E5
  python eval.py --models bm25 e5

  # Run a single model
  python eval.py --models mpnet

  # Override default paths
  python eval.py --models bm25 mpnet e5
"""

import argparse
import json
import time

import numpy as np

# Default paths
DEFAULT_QUERIES_PATH = "data/test_data/rag_ready_w_queries.jsonl"
DEFAULT_CORPUS_PATH = "data/full_data/rag_documents.jsonl"

DENSE_CONFIG = {
    "mpnet": {
        "model_name": "all-mpnet-base-v2",
        "index_path": "data/dense/faiss.index",
    },
    "e5": {
        "model_name": "intfloat/e5-large-v2",
        "index_path": "data/dense/e5_corpus.index",
        # E5 requires a task prefix on queries at inference time
        "query_prefix": "query: ",
    },
}


# Data loading
def load_corpus(path: str) -> list[str]:
    corpus = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            corpus.append(item["equipment_description"])
    return corpus


def load_queries(path: str) -> list[tuple[str, int]]:
    """Returns (query_text, relevant_doc_id) pairs.

    doc_id is the 0-based line index in the *queries* file, which must match
    the 0-based line index in the corpus file (i.e. same document order).
    """
    pairs = []
    with open(path) as f:
        for doc_id, line in enumerate(f):
            item = json.loads(line)
            queries = item.get("queries", [])
            if not queries:
                continue  # skip docs that have no synthetic queries
            for query in queries:
                pairs.append((query, doc_id))
    return pairs


# Metric helpers
def recall_at_k(ranked_indices: list[int], relevant_idx: int, k: int) -> float:
    return 1.0 if relevant_idx in ranked_indices[:k] else 0.0


def reciprocal_rank(ranked_indices: list[int], relevant_idx: int) -> float:
    for rank, idx in enumerate(ranked_indices, start=1):
        if idx == relevant_idx:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_indices: list[int], relevant_idx: int, k: int) -> float:
    """Binary-relevance nDCG (single relevant document)."""
    if relevant_idx in ranked_indices[:k]:
        position = ranked_indices[:k].index(relevant_idx)
        return 1.0 / np.log2(position + 2)  # +2 because log2(1) = 0
    return 0.0


def aggregate_and_print(
    label: str,
    queries: list[tuple],
    corpus_size: int,
    recall_scores: dict,
    ndcg_scores: dict,
    rr_scores: list,
    ks: list[int],
    elapsed: float,
) -> None:
    n = len(queries)
    print(f"\n{'=' * 55}")
    print(f"  Model : {label}")
    print(f"  Queries evaluated : {n}  |  Corpus size : {corpus_size}")
    print(f"  Wall time : {elapsed:.1f}s")
    print(f"{'=' * 55}")
    print(f"  MRR    : {sum(rr_scores) / n:.4f}")
    for k in ks:
        print(f"  Recall@{k:<2}: {sum(recall_scores[k]) / n:.4f}")
    for k in ks:
        print(f"  nDCG@{k:<2} : {sum(ndcg_scores[k]) / n:.4f}")
    print()


# BM25 evaluator
def evaluate_bm25(
    corpus: list[str],
    queries: list[tuple[str, int]],
    ks: list[int],
) -> None:
    from rank_bm25 import BM25Okapi

    t0 = time.time()
    tokenized = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    recall_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}
    rr_scores = []

    for query, relevant_idx in queries:
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

    aggregate_and_print(
        "BM25 (Okapi)",
        queries,
        len(corpus),
        recall_scores,
        ndcg_scores,
        rr_scores,
        ks,
        time.time() - t0,
    )


# Dense (FAISS) evaluator  –  shared by mpnet and e5
def evaluate_dense(
    corpus: list[str],
    queries: list[tuple[str, int]],
    ks: list[int],
    model_key: str,
) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer

    cfg = DENSE_CONFIG[model_key]
    model_name = cfg["model_name"]
    index_path = cfg["index_path"]
    query_prefix = cfg.get("query_prefix", "")

    t0 = time.time()
    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_path)

    # Sanity-check: index vectors should match corpus length
    if index.ntotal != len(corpus):
        print(
            f"  [WARNING] FAISS index has {index.ntotal} vectors but corpus "
            f"has {len(corpus)} documents. Corpus-size in the report will "
            f"reflect the index, not the loaded corpus."
        )
    corpus_size = index.ntotal

    recall_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}
    rr_scores = []

    top_k = max(ks)

    for query, relevant_idx in queries:
        prefixed = query_prefix + query
        embedding = model.encode([prefixed], normalize_embeddings=True)
        embedding = np.array(embedding, dtype=np.float32)
        _, indices = index.search(embedding, top_k)
        ranked_indices = indices[0].tolist()

        # FAISS may return -1 for padding when fewer than top_k docs exist
        ranked_indices = [i for i in ranked_indices if i >= 0]

        for k in ks:
            recall_scores[k].append(
                recall_at_k(ranked_indices, relevant_idx, k)
            )
            ndcg_scores[k].append(ndcg_at_k(ranked_indices, relevant_idx, k))
        rr_scores.append(reciprocal_rank(ranked_indices, relevant_idx))

    label = f"{model_name}" + (
        f" (prefix='{query_prefix}')" if query_prefix else ""
    )
    aggregate_and_print(
        label,
        queries,
        corpus_size,
        recall_scores,
        ndcg_scores,
        rr_scores,
        ks,
        time.time() - t0,
    )


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval models (BM25 / all-mpnet / E5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=["bm25", "mpnet", "e5"],
        metavar="MODEL",
        help="One or more of: bm25  mpnet  e5",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        metavar="K",
        help="Values of k for Recall@k and nDCG@k (default: 1 5 10)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading corpus  : {args.corpus}")
    corpus = load_corpus(args.corpus)
    print(f"Loading queries : {args.queries}")
    queries = load_queries(args.queries)
    print(f"Corpus size     : {len(corpus)}")
    print(f"Query pairs     : {len(queries)}")
    print(f"Models selected : {', '.join(args.models)}")
    print(f"k values        : {args.ks}")

    for model_key in args.models:
        if model_key == "bm25":
            evaluate_bm25(corpus, queries, args.ks)
        else:
            evaluate_dense(corpus, queries, args.ks, model_key)


if __name__ == "__main__":
    main()
