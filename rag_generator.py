import json
import os

import urllib.request

# global variables
CORPUS_PATH = "data/full_data/rag_documents.jsonl"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
DEFAULT_K = 5


def build_prompt(query: str, text: str) -> str:
    return f"""You are a laboratory equipment assistant helping researchers find suitable equipment for their tasks.

You will be given a researcher's question and exactly 5 equipment items retrieved from a database.
Using ONLY the provided context, write a structured response with three parts:

1. EQUIPMENT OVERVIEW — For each of the 5 items, write 1-2 sentences describing what it does and how it relates to the researcher's task.
2. COMPARISON — In 3-5 sentences, compare the options across relevant dimensions (e.g. throughput, precision, cost, ease of use). Focus on differences that matter for the task.
3. CONCLUSION — In 1-3 sentences, recommend the most suitable option(s) and briefly justify why.

Stick strictly to the context. If an item seems unrelated to the task, note that briefly instead of skipping it.

Context:
{text}

Question: {query}

Answer:"""


def ollama_generate(prompt: str) -> str:
    payload = json.dumps(
        {"model": MODEL, "prompt": prompt, "stream": False}
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"]


def generate(query: str, retrieved_docs: list[str]) -> str:
    text = "\n".join(retrieved_docs)
    prompt = build_prompt(query, text)
    return ollama_generate(prompt)
