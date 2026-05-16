import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  # or SnowballStemmer("english")

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


def load_corpus(path: str) -> list[str]:
    """
    Function used in app.py and evaluation.py to load equipment description 
    from corpus in jsonl format.
    """
    corpus = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            corpus.append(item["equipment_description"])
    return corpus


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
