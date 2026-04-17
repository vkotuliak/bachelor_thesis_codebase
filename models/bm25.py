from rank_bm25 import BM25Okapi
import json

CORPUS_PATH = "data/full_data/rag_documents.jsonl"


def load_data(file_path):
    """Loads the corpus from a JSONL file, extracting the "equipment_description" field from each line."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            dict_line = json.loads(line)
            data.append(dict_line["equipment_description"])
    return data


def run_bm25():
    """Loads the corpus, builds the BM25 index, and allows the user to input a search query."""
    corpus = load_data(CORPUS_PATH)
    tokenised_corpus = [doc.split() for doc in corpus]

    bm25 = BM25Okapi(tokenised_corpus)

    query = str(input("Please input your search query: "))
    tokenised_query = query.split()

    scores = bm25.get_scores(tokenised_query)
    top_docs = bm25.get_top_n(tokenised_query, corpus)

    return scores, top_docs


def main():
    scores, top_docs = run_bm25()
    # print(f"array of scores per doc {scores}")
    print("Top matches:")
    for i, doc in enumerate(top_docs):
        first_sep = doc.find("[SEP]")
        print(f"{i}. {doc[5:first_sep]}")


if __name__ == "__main__":
    main()
