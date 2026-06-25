"""
evaluation.py  –  Unified retrieval evaluation for TF-IDF BM25,
    all-mpnet-base-v2, E5-large-v2, and Hybrid retrieval (E5 + BM25).

Usage examples
--------------
  # Run all four models
  python evaluation.py --models tfidf bm25 mpnet e5 hybrid

  # Run only BM25 and E5
  python evaluation.py --models bm25 e5

  # Run a single model
  python evaluation.py --models mpnet
"""

import argparse
import json
import pickle
import time
import numpy as np

import utils

# Default paths
DEFAULT_QUERIES_PATH = "data/test_data/queries_from_personas.jsonl"
DEFAULT_CORPUS_PATH = "data/full_data/rag_documents.jsonl"


def load_queries(path: str) -> list[tuple[str, str]]:
    """Returns (query_text, relevant_doc_id) pairs.

    doc_id is the 0-based line index in the *queries* file, which must match
    the 0-based line index in the corpus file (i.e. same document order).
    """
    pairs = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            queries = item.get("query", [])
            if not queries:
                continue  # skip docs that have no synthetic queries
            eq_id = item["id"]
            for query in queries:
                pairs.append((query, eq_id))
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
        return 1.0 / np.log2(position + 2)
    return 0.0


def get_bm25_ranking(query_tokens: list[str], bm25_model) -> list[int]:
    scores = bm25_model.get_scores(query_tokens)
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def evaluate_tfidf(
    corpus_texts: list[str],
    corpus_ids: list[str],
    queries: list[tuple[str, str]],
    ks: list[int],
) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    id_to_pos = {eq_id: pos for pos, eq_id in enumerate(corpus_ids)}

    t0 = time.time()

    vectorizer = TfidfVectorizer(
        tokenizer=utils.scientific_tokenizer,
        lowercase=False,
        token_pattern=None,
    )
    corpus_matrix = normalize(
        vectorizer.fit_transform(corpus_texts)
    )  # (N, V), L2-normalised

    recall_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}
    rr_scores = []

    for query, relevant_eq_id in queries:
        relevant_pos = id_to_pos[relevant_eq_id]

        query_vec = normalize(vectorizer.transform([query]))  # (1, V)
        scores = (
            (corpus_matrix @ query_vec.T).toarray().ravel()
        )  # cosine similarity
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        for k in ks:
            recall_scores[k].append(
                recall_at_k(ranked_indices, relevant_pos, k)
            )
            ndcg_scores[k].append(ndcg_at_k(ranked_indices, relevant_pos, k))
        rr_scores.append(reciprocal_rank(ranked_indices, relevant_pos))

    utils.aggregate_and_print(
        "TF-IDF",
        queries,
        len(corpus_texts),
        recall_scores,
        ndcg_scores,
        rr_scores,
        ks,
        time.time() - t0,
    )


# BM25 evaluator
def evaluate_bm25(
    corpus_texts: list[str],
    corpus_ids: list[str],
    queries: list[tuple[str, str]],
    ks: list[int],
) -> None:

    id_to_pos = {eq_id: pos for pos, eq_id in enumerate(corpus_ids)}

    t0 = time.time()
    with open("data/dense/bm25_corpus.pkl", "rb") as f:
        bm25 = pickle.load(f)

    recall_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}
    rr_scores = []

    for query, relevant_eq_id in queries:
        relevant_pos = id_to_pos[relevant_eq_id]

        ranked_indices = get_bm25_ranking(
            utils.scientific_tokenizer(query), bm25
        )

        for k in ks:
            recall_scores[k].append(
                recall_at_k(ranked_indices, relevant_pos, k)
            )
            ndcg_scores[k].append(ndcg_at_k(ranked_indices, relevant_pos, k))
        rr_scores.append(reciprocal_rank(ranked_indices, relevant_pos))

    utils.aggregate_and_print(
        "BM25 (Okapi)",
        queries,
        len(corpus_texts),
        recall_scores,
        ndcg_scores,
        rr_scores,
        ks,
        time.time() - t0,
    )


# Dense (FAISS) evaluator  –  shared by mpnet and e5
def evaluate_dense(
    corpus_texts: list[str],
    corpus_ids: list[str],
    queries: list[tuple[str, str]],
    ks: list[int],
    model_key: str,
) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer

    id_to_pos = {eq_id: pos for pos, eq_id in enumerate(corpus_ids)}

    cfg = utils.DENSE_CONFIG[model_key]
    model_name = cfg["model_name"]
    index_path = cfg["index_path"]
    query_prefix = cfg.get("query_prefix", "")

    t0 = time.time()
    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_path)

    # Sanity-check: index vectors should match corpus length
    if index.ntotal != len(corpus_texts):
        print(
            f"  [WARNING] FAISS index has {index.ntotal} vectors but corpus "
            f"has {len(corpus_texts)} documents. Corpus-size in the report will "
            f"reflect the index, not the loaded corpus."
        )
    corpus_size = index.ntotal

    recall_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}
    rr_scores = []

    query_texts = [query_prefix + q[0] for q in queries]
    all_embeddings = model.encode(
        query_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    for i, (_, relevant_eq_id) in enumerate(queries):
        relevant_pos = id_to_pos[relevant_eq_id]

        embedding = all_embeddings[i : i + 1]
        _, indices = index.search(embedding, max(ks))
        ranked_indices = indices[0].tolist()

        # FAISS may return -1 for padding when fewer than top_k docs exist
        ranked_indices = [i for i in ranked_indices if i >= 0]

        for k in ks:
            recall_scores[k].append(
                recall_at_k(ranked_indices, relevant_pos, k)
            )
            ndcg_scores[k].append(ndcg_at_k(ranked_indices, relevant_pos, k))
        rr_scores.append(reciprocal_rank(ranked_indices, relevant_pos))

    label = f"{model_name}" + (
        f" (prefix='{query_prefix}')" if query_prefix else ""
    )
    utils.aggregate_and_print(
        label,
        queries,
        corpus_size,
        recall_scores,
        ndcg_scores,
        rr_scores,
        ks,
        time.time() - t0,
    )


# Hybrid Evaluator - combines E5 and BM25 using RRF
def evaluate_hybrid(
    corpus_texts: list[str],
    corpus_ids: list[str],
    queries: list[tuple[str, str]],
    ks: list[int],
    rrf_k: int = 10,
    top_k: int = 200,
    alpha: float = 0.4,  # weight for e5 (bm25 = 1-alpha)
) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer

    id_to_pos = {eq_id: pos for pos, eq_id in enumerate(corpus_ids)}

    t0 = time.time()

    # --- Load BM25 index ---
    with open("data/dense/bm25_corpus.pkl", "rb") as f:
        bm25 = pickle.load(f)

    # --- Load E5 model and FAISS index ---
    cfg = utils.DENSE_CONFIG["e5"]
    model = SentenceTransformer(cfg["model_name"])
    index = faiss.read_index(cfg["index_path"])
    query_prefix = cfg.get("query_prefix", "")

    recall_scores = {k: [] for k in ks}
    ndcg_scores = {k: [] for k in ks}
    rr_scores = []

    corpus_size = len(corpus_texts)

    query_texts = [query_prefix + q[0] for q in queries]
    all_embeddings = model.encode(
        query_texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    for i, (query, relevant_eq_id) in enumerate(queries):
        relevant_pos = id_to_pos[relevant_eq_id]  # resolve ID → position

        bm25_ranking = get_bm25_ranking(
            utils.scientific_tokenizer(query), bm25
        )[:top_k]

        _, indices = index.search(all_embeddings[i : i + 1], top_k)
        e5_ranking = indices[0].tolist()

        weight_bm25 = 1 - alpha
        weight_e5 = alpha

        rrf_scores: dict[int, float] = {}

        for rank, doc_idx in enumerate(bm25_ranking, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (
                weight_bm25 / (rrf_k + rank)
            )

        for rank, doc_idx in enumerate(e5_ranking, start=1):
            rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (
                weight_e5 / (rrf_k + rank)
            )

        # 3. Sort by the new weighted scores
        ranked_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        for k in ks:
            recall_scores[k].append(
                recall_at_k(ranked_indices, relevant_pos, k)
            )
            ndcg_scores[k].append(ndcg_at_k(ranked_indices, relevant_pos, k))
        rr_scores.append(reciprocal_rank(ranked_indices, relevant_pos))

    utils.aggregate_and_print(
        "Hybrid BM25 + E5 (RRF)",
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
        choices=["tfidf", "bm25", "mpnet", "e5", "hybrid"],
        metavar="MODEL",
        help="One or more of: tfidf bm25  mpnet  e5 hybrid",
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

    print(f"Loading corpus  : {DEFAULT_CORPUS_PATH}")
    corpus = utils.load_corpus(DEFAULT_CORPUS_PATH)
    print(f"Loading queries : {DEFAULT_QUERIES_PATH}")
    queries = load_queries(DEFAULT_QUERIES_PATH)
    print(f"Corpus size     : {len(corpus[0])}")
    print(f"Query pairs     : {len(queries)}")
    print(f"Models selected : {', '.join(args.models)}")
    print(f"k values        : {args.ks}")

    for model_key in args.models:
        if model_key == "tfidf":
            evaluate_tfidf(corpus[0], corpus[1], queries, args.ks)
        elif model_key == "bm25":
            evaluate_bm25(corpus[0], corpus[1], queries, args.ks)
        elif model_key == "hybrid":
            evaluate_hybrid(corpus[0], corpus[1], queries, args.ks)
        else:
            evaluate_dense(corpus[0], corpus[1], queries, args.ks, model_key)


if __name__ == "__main__":
    main()
