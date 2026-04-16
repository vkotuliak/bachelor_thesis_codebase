import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

DATA_PATH = "data/full_data/rag_documents.jsonl"


def load_data(path):
    corpus = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            corpus.append(item["equipment_description"])
    return corpus


def main():
    corpus = load_data(DATA_PATH)
    model = SentenceTransformer("all-mpnet-base-v2", device="cpu")

    embeddings = model.encode(
        corpus,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(len(embeddings), embeddings)

    Path("data/dense").mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, "data/dense/faiss.index")
    print("Index saved.")


if __name__ == "__main__":
    main()
