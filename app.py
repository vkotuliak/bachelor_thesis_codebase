"""
app.py  –  Interactive equipment search using BM25, all-mpnet-base-v2,
    E5-large-v2 or hybrid (BM25+E5)
    Use --generate flag to employ qwen2.5:7b as a generator.

Usage examples
--------------
  python app.py --model bm25
  python app.py --model mpnet --k 10
  python app.py --model bm25 --corpus data/full_data/rag_documents.jsonl
  python app.py --model e5 --generate
"""

import argparse
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

import utils

# Defaults
DEFAULT_CORPUS_PATH = "data/full_data/rag_documents.jsonl"
DEFAULT_K = 10


def print_results(results: list[str]) -> None:
    print("\nTop matches:")
    names = [utils.extract_name(result) for result in results]
    for rank, name in enumerate(names, start=1):
        print(f"  {rank}. {name}")
    print()


# Model runners
def run_bm25(bm25, corpus: list[str], query: str, k: int) -> list[str]:

    tokenized_query = utils.scientific_tokenizer(query)
    top_docs = bm25.get_top_n(tokenized_query, corpus, n=k)

    return top_docs


def run_dense(
    corpus: list[str], query: str, k: int, model_key: str
) -> list[str]:
    import faiss
    from sentence_transformers import SentenceTransformer

    cfg = utils.DENSE_CONFIG[model_key]
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
    bm25, corpus: list[str], query: str, k: int = 100, n: int = 5
) -> list[str]:
    dense_indices = run_dense(
        corpus,
        query,
        k,
        "e5",
    )
    dense_indices = [corpus[int(i)] for i in dense_indices]
    sparse_indices = run_bm25(bm25, corpus, query, k)

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
    parser.add_argument("--generate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading corpus from: {args.corpus}")
    corpus, _ = utils.load_corpus(args.corpus)
    print(f"Corpus size: {len(corpus)} documents")
    print(f"Model: {args.model}  |  k: {args.k}")

    if args.model == "bm25" or args.model == "hybrid":
        print("Tokenising...")
        tokenized_corpus = [utils.scientific_tokenizer(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        print("Tokenisation finished. \n")

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
            results = [doc for doc in run_bm25(bm25, corpus, query, args.k)]
        elif args.model == "hybrid":
            results = [doc for doc in run_hybrid(bm25, corpus, query)]
        else:
            results = []
            for i in run_dense(corpus, query, args.k, args.model):
                results.append(corpus[int(i)])

        if args.generate:
            from rag_generator import generate
            print(generate(query, results), "\n") 
        else:
            print_results(results)


if __name__ == "__main__":
    main()
