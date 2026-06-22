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
import pickle

import numpy as np

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


def load_dense_model(model_key: str):
    import faiss
    from sentence_transformers import SentenceTransformer

    cfg = utils.DENSE_CONFIG[model_key]
    model = SentenceTransformer(cfg["model_name"])
    index = faiss.read_index(cfg["index_path"])
    return model, index, cfg


def run_dense(query: str, k: int, model, index, cfg) -> list[str]:

    prefixed_query = cfg.get("query_prefix", "") + query
    embedding = model.encode([prefixed_query], normalize_embeddings=True)
    embedding = np.array(embedding, dtype=np.float32)

    _, indices = index.search(embedding, k)
    ranked_indices = [
        i for i in indices[0] if i >= 0
    ]  # filter FAISS -1 padding

    return ranked_indices


def reciprocal_rank_fusion(
    results_list: list[list[str]], k: int = 10, alpha: float = 0.4
):
    scores = defaultdict(float)
    weights = [alpha, 1 - alpha]

    for i, results in enumerate(results_list):
        weight = weights[i]
        for rank, doc_id in enumerate(results, start=1):
            scores[doc_id] += weight / (k + rank)

    return sorted(scores, key=scores.__getitem__, reverse=True)


def run_hybrid(
    bm25,
    corpus: list[str],
    query: str,
    model,
    index,
    cfg,
    k: int = 200,
    num_of_results: int = DEFAULT_K,
) -> list[str]:
    dense_indices = run_dense(query, k, model, index, cfg)
    dense_indices = [corpus[int(i)] for i in dense_indices]
    sparse_indices = run_bm25(bm25, corpus, query, k)

    fused = reciprocal_rank_fusion([dense_indices, sparse_indices])

    return fused[:num_of_results]


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


def print_details(result: str) -> None:
    print("\n--- Equipment Details ---")
    fields = result.split("[SEP]")
    for field in fields:
        field = field.strip()
        if not field:
            continue
        if ": " in field:
            label, value = field.split(": ", 1)
            print(f"{label.strip().upper()}: {value.strip()}")
    print("-------------------------\n")


def main() -> None:
    args = parse_args()

    print(f"Loading corpus from: {args.corpus}")
    corpus, _ = utils.load_corpus(args.corpus)
    print(f"Corpus size: {len(corpus)} documents")
    print(f"Model: {args.model}  |  k: {args.k}\n")

    if args.model == "bm25" or args.model == "hybrid":
        print("Loading corpus...")
        with open("data/dense/bm25_corpus.pkl", "rb") as f:
            bm25 = pickle.load(f)
        print("Corpus loaded.\n")

    if args.model in ("hybrid", "e5", "mpnet"):
        print("Loading dense model...")
        model_name = "e5" if args.model == "hybrid" else args.model
        dense_model, dense_index, dense_cfg = load_dense_model(model_name)
        print("Model loaded\n")

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
            results = [
                doc
                for doc in run_hybrid(
                    bm25,
                    corpus,
                    query,
                    dense_model,
                    dense_index,
                    dense_cfg,
                    num_of_results=args.k,
                )
            ]
        else:
            results = []
            for i in run_dense(
                query,
                args.k,
                dense_model,
                dense_index,
                dense_cfg,
            ):
                results.append(corpus[int(i)])

        if args.generate:
            from rag_generator import generate
            print("Generating answer...\n")
            print(generate(query, results), "\n")
        else:
            print_results(results)

            while True:
                try:
                    choice = input(
                        f"Enter number (1-{len(results)}) for details, or press Enter for new search:\n> "
                    ).strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nExiting.")
                    return

                if not choice:
                    break

                if choice.isdigit() and 1 <= int(choice) <= len(results):
                    print_details(results[int(choice) - 1])
                else:
                    print(f"Bad number. Pick 1-{len(results)}.\n")


if __name__ == "__main__":
    main()
