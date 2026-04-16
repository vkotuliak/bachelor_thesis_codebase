import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")
index = faiss.read_index("data/dense/faiss.index")
corpus = json.loads("data/full_data/rag_documents.jsonl")


def load_data(path):
    corpus = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            corpus.append(item["equipment_description"])


def search(query: str, corpus: list) -> str:
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    _, indices = index.search(1, query_embedding, 1)
    # Do I actually want to return corpus or what?
    return corpus[indices[0][0]]


def main():
    query = input("\nType your question below:".strip())
    print(f"\nBest match:\n {search(query, corpus)}")


if __name__ == "__main__":
    main()
