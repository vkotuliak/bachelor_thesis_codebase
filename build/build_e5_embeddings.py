import json
import faiss
from sentence_transformers import SentenceTransformer

# DATA_PATH = "data/full_data/rag_documents.jsonl"
DATA_PATH = "data/test_data/rag_ready_w_queries.jsonl"


def load_data(path):
    corpus = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            corpus.append(item["equipment_description"])
    return corpus


def main():
    docs = load_data(DATA_PATH)
    model = SentenceTransformer("intfloat/e5-large-v2")

    # E5 requires the "passage: " prefix for documents
    texts = ["passage: " + doc for doc in docs]

    embeddings = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, "data/dense/e5_corpus.index")

    print(embeddings.shape)
    print("index saved.")


if __name__ == "__main__":
    main()
