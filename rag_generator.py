import argparse
import json
import os

import urllib.request
import app
import utils

# global variables
CORPUS_PATH = "data/full_data/rag_documents.jsonl"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
DEFAULT_K = 5


def build_prompt(query: str, text: str) -> str:
    return f"""You are a laboratory equipment assistant helping researchers find suitable equipment for their tasks.

You will be given a researcher's question and exactly 5 equipment items retrieved from a database.
Using ONLY the provided context, write a structured response with three parts:

1. EQUIPMENT OVERVIEW — For each of the 5 items, write 1-2 sentences describing what it does and how it relates to the researcher's task.
2. COMPARISON — In 3-5 sentences, compare the options across relevant dimensions (e.g. throughput, precision, cost, ease of use). Focus on differences that matter for the task.
3. CONCLUSION — In 1-3 sentences, recommend the most suitable option(s) and briefly justify why.

Stick strictly to the context. If an item seems unrelated to the task, note that briefly instead of skipping it.

Context:
{text}

Question: {query}

Answer:"""


def ollama_generate(prompt: str) -> str:
    payload = json.dumps(
        {"model": MODEL, "prompt": prompt, "stream": False}
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"]


def get_answer(query, text):
    prompt = build_prompt(query, text)
    answer = ollama_generate(prompt)
    return answer


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading corpus from: {CORPUS_PATH}")
    corpus, _ = utils.load_corpus(CORPUS_PATH)
    print(f"Corpus size: {len(corpus)} documents")
    print(f"Model: {args.model}  |  k: {args.k}\n")

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
            results = "\n".join(
                [doc for doc in app.run_bm25(corpus, query, args.k)]
            )
        elif args.model == "hybrid":
            results = "\n".join([doc for doc in app.run_hybrid(corpus, query)])
        else:
            results = []
            for i in app.run_dense(corpus, query, args.k, args.model):
                results.append(corpus[int(i)])
            results = "\n".join(results)

        answer = get_answer(query, results)
        print(answer)
        print()


if __name__ == "__main__":
    main()
