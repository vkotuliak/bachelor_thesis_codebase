from rank_bm25 import BM25Okapi
import json

# Load the data from `rag_documents.jsonl`
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            dict_line = json.loads(line)
            data.append(dict_line["equipment_description"])
    return data

def run_bm25():
    file_path = "data/rag_documents.jsonl"
    corpus = load_data(file_path)
    tokenised_corpus = [doc.split() for doc in corpus]

    bm25 = BM25Okapi(tokenised_corpus)

    query = str(input("Please input your search query: "))
    tokenised_query = query.split()

    scores = bm25.get_scores(tokenised_query)
    top_docs = bm25.get_top_n(tokenised_query, corpus)

    return scores, top_docs

def main():
    scores, top_docs = run_bm25()
    print(f"array of scores per doc {scores}")
    for i, doc in enumerate(top_docs):
        first_sep = doc.find("[SEP]")
        print(f"Equipment number {i+1}: {doc[5:first_sep]}")
    # print(f"top docs: {top_docs}")

if __name__ == "__main__":
    main()
