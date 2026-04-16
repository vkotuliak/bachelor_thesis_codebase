import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


INPUT_DATA = "data/full_data/rag_documents.jsonl"


def load_qrels(path):
    """Loads the evaluation data from a JSONL file, returning the corpus and query-relevance pairs."""
    corpus = []
    qrels = []  # query, relevant_doc_idx

    with open(path) as f:
        for doc_idx, line in enumerate(f):
            item = json.loads(line)
            corpus.append(item["equipment_description"])
            for query in item["queries"]:
                qrels.append((query, doc_idx))

    return corpus, qrels


corpus, doc_ids = load_qrels(INPUT_DATA)
model = SentenceTransformer("all-mpnet-base-v2")

embeddings = model.encode(
    corpus, batch_size=32, show_progress_bar=True, normalize_embeddings=True
)
embeddings = np.array(embeddings, dtype=np.float32)  # shape: (n_docs, 768)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(len(embeddings), embeddings)

Path("data/dense").mkdir(parents=True, exist_ok=True)
faiss.write_index(index, "data/dense/faiss.index")
np.save("data/dense/embeddings.npy", embeddings)

index = faiss.read_index("data/dense/faiss.index")
embeddings = np.load("data/dense/embeddings.npy")


def search(query: str, k: int = 10):
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    scores, indices = index.search(1, query_embedding, k)
    return [
        (doc_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0])
    ]
