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
