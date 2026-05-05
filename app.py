"""
app.py  –  Interactive equipment search using BM25, all-mpnet-base-v2, or E5-large-v2.

Usage examples
--------------
  python app.py --model bm25
  python app.py --model mpnet
  python app.py --model e5
  python app.py --model mpnet --k 10
  python app.py --model bm25 --corpus data/full_data/rag_documents.jsonl
"""

import argparse
from collections import defaultdict
import json

import numpy as np

# Defaults
DEFAULT_CORPUS_PATH = "data/full_data/rag_documents.jsonl"
DEFAULT_K = 5

DENSE_CONFIG = {
    "mpnet": {
        "model_name": "all-mpnet-base-v2",
        "index_path": "data/dense/faiss.index",
        "query_prefix": "",
    },
    "e5": {
        "model_name": "intfloat/e5-large-v2",
        "index_path": "data/dense/e5_corpus.index",
        "query_prefix": "query: ",
    },
}


# Shared helpers
def load_corpus(path: str) -> list[str]:
    corpus = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            corpus.append(item["equipment_description"])
    return corpus


def extract_name(description: str) -> str:
    """Extracts the equipment name from the structured description string.

    Expected format: "name: <NAME> [SEP] ..."
    Falls back to the full description if the format is unexpected.
    """
    parts = description.split("[SEP]")
    name_part = parts[0].strip()
    if name_part.lower().startswith("name:"):
        return name_part[len("name:") :].strip()
    return name_part


def print_results(results: list[str]) -> None:
    print("\nTop matches:")
    for rank, name in enumerate(results, start=1):
        print(f"  {rank}. {name}")
    print()


# Model runners
def run_bm25(corpus: list[str], query: str, k: int) -> list[str]:
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [doc.lower().split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    top_docs = bm25.get_top_n(tokenized_query, corpus, n=k)

    return top_docs


def run_dense(
    corpus: list[str], query: str, k: int, model_key: str
) -> list[str]:
    import faiss
    from sentence_transformers import SentenceTransformer

    cfg = DENSE_CONFIG[model_key]
    model = SentenceTransformer(cfg["model_name"])
    index = faiss.read_index(cfg["index_path"])

    if index.ntotal != len(corpus):
        print(
            f"[WARNING] FAISS index has {index.ntotal} vectors but corpus has "
            f"{len(corpus)} documents. Results may be misaligned."
        )

    prefixed_query = cfg["query_prefix"] + query
    embedding = model.encode([prefixed_query], normalize_embeddings=True)
    embedding = np.array(embedding, dtype=np.float32)

    _, indices = index.search(embedding, k)
    ranked_indices = [
        i for i in indices[0] if i >= 0
    ]  # filter FAISS -1 padding

    return ranked_indices


def reciprocal_rank_fusion(results_list: list[list[str]], k: int = 60):
    scores = defaultdict(float)
    for results in results_list:
        for rank, doc_id in enumerate(results):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)


def run_hybrid(
    corpus: list[str], query: str, k: int = 100, n: int = 5
) -> list[str]:
    dense_indices = run_dense(
        corpus,
        query,
        k,
        "e5",
    )
    dense_indices = [corpus[int(i)] for i in dense_indices]
    sparse_indices = run_bm25(corpus, query, k)

    fused = reciprocal_rank_fusion([dense_indices, sparse_indices])

    return fused[:n]


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive equipment search (BM25 / all-mpnet / E5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["bm25", "mpnet", "e5", "hybrid"],
        help="Retrieval model to use.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of results to return (default: {DEFAULT_K}).",
    )
    parser.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS_PATH,
        help=f"Path to corpus JSONL (default: {DEFAULT_CORPUS_PATH}).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading corpus from: {args.corpus}")
    corpus = load_corpus(args.corpus)
    print(f"Corpus size: {len(corpus)} documents")
    print(f"Model: {args.model}  |  k: {args.k}\n")

    # Load heavy dependencies once, then loop
    while True:
        try:
            query = input(
                "Enter search query (or Ctrl+C to quit):\n> "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            print("Empty query, please try again.\n")
            continue

        if args.model == "bm25":
            results = [
                extract_name(doc) for doc in run_bm25(corpus, query, args.k)
            ]
        elif args.model == "hybrid":
            results = [extract_name(doc) for doc in run_hybrid(corpus, query)]
        else:
            results = [
                extract_name(corpus[int(i)])
                for i in run_dense(corpus, query, args.k, args.model)
            ]

        print_results(results)


if __name__ == "__main__":
    main()
