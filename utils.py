import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# def initial_tokenization(text: str) -> list[str]:
#     return text.lower().split()

# def scientific_tokenizer(text: str) -> list[str]:
#  tokens = re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", text.lower())
#     return tokens

nltk.download("stopwords")
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def scientific_tokenizer(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)  # remove punctuation
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens


def load_corpus(path: str) -> tuple[list[str], list[str]]:
    """
    Function used in app.py and evaluation.py to load equipment description
    from corpus in jsonl format.
    """
    corpus = []
    ids = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            corpus.append(item["equipment_description"])
            ids.append(item["id"])
    return corpus, ids


def extract_name(description: str) -> str:
    """Extracts the equipment name from the structured description string.

    Expected format: "name: <NAME> [SEP] ..."
    Falls back to the full description if the format is unexpected.
    """
    parts = description.split("[SEP]")
    name_part = parts[0].strip()
    if name_part.lower().startswith("name:"):
        return name_part[len("name:") :].strip()
    return name_part


def aggregate_and_print(
    label: str,
    queries: list[tuple],
    corpus_size: int,
    recall_scores: dict,
    ndcg_scores: dict,
    rr_scores: list,
    ks: list[int],
    elapsed: float,
) -> None:
    """Function to aggregate and print results from evaluation nicely."""
    n = len(queries)
    print(f"\n{'=' * 55}")
    print(f"  Model : {label}")
    print(f"  Queries evaluated : {n}  |  Corpus size : {corpus_size}")
    print(f"  Wall time : {elapsed:.1f}s")
    print(f"{'=' * 55}")
    print(f"  MRR    : {sum(rr_scores) / n:.4f}")
    for k in ks:
        print(f"  Recall@{k:<2}: {sum(recall_scores[k]) / n:.4f}")
    for k in ks:
        print(f"  nDCG@{k:<2} : {sum(ndcg_scores[k]) / n:.4f}")
    print()
