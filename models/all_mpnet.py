import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")
index = faiss.read_index("data/dense/faiss.index")
# corpus = json.loads("data/full_data/rag_documents.jsonl")

CORPUS_PATH = "data/full_data/rag_documents.jsonl"


def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            data.append(item["equipment_description"])
    return data


def extract_name(description: str) -> str:
    # Description always starts with "name: <NAME> [SEP] ..."
    name_part = description.split("[SEP]")[0]
    return name_part.replace("name:", "").strip()


def search(query: str, k: int = 5) -> list:
    corpus = load_data(CORPUS_PATH)

    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    _, indices = index.search(query_embedding, k)
    return [extract_name(corpus[i]) for i in indices[0]]


def main():
    query = input("\nType your question below:\n").strip()
    results = search(query)
    print("\nTop matches:")
    for rank, name in enumerate(results, start=1):
        print(f"{rank}. {name}")


if __name__ == "__main__":
    main()
