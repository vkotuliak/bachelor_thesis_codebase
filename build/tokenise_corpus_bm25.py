import pickle
from rank_bm25 import BM25Okapi
import sys
from pathlib import Path

# Adds the parent directory of the current file to the search path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import utils

corpus, _ = utils.load_corpus("data/full_data/rag_documents.jsonl")
tokenized = [utils.scientific_tokenizer(doc) for doc in corpus]
bm25 = BM25Okapi(tokenized)

with open("data/dense/bm25_corpus.pkl", "wb") as f:
    pickle.dump(bm25, f)

print("BM25 index successfully saved!")
