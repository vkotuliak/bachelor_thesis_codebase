import json
import re
import sys
import os
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL      = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

ONE_SHOT_QUERY = (
    "I work at Polyvation. We are developing a new polymer, which is a variation on PEEK "
    "for medical grade purposes. The material turns out to be way softer than anticipated. "
    "Which equipment could be used to analyse why our experimental material turns out so soft?"
)

INPUT_FILE  = os.environ.get("INPUT_FILE",  "data/rag_documents.jsonl")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "data/documents_with_queries.jsonl")


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in: {text!r}")
    return json.loads(match.group())


def ollama_generate(prompt: str) -> str:
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"]


def build_prompt(item: dict) -> str:
    return f"""You generate search queries for the University of Groningen equipment portal.
Portal users are researchers or industry partners who have a concrete scientific problem and are looking for the right instrument at RUG — they do NOT yet know the equipment name.

Here is an example of a real query submitted to the portal:

<example_query>
{ONE_SHOT_QUERY}
</example_query>

Now generate 1 query for the equipment description below.
Match the style of the example — first-person, problem-driven, plain language — but use different wording and vocabulary. Do not reuse phrases like "soft polymer", "variation on PEEK", or any other phrasing from the example.

Rules:
- Do NOT include the equipment name or any of its aliases in the query.
- The query should sound like it was typed into a search box, not written as a sentence in a paper.

<document>
Equipment Name: {item.get("name", "")}
Aliases: {item.get("aliases", "")}
Short Description: {item.get("short description", "")}
Typical Applications: {item.get("typical applications", "")}
Tags: {item.get("tags", "")}
</document>

Return ONLY a JSON object: {{"query": "..."}}"""


os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout:
    for i, line in enumerate(fin):
        item = json.loads(line)
        try:
            print(build_prompt(item))
            raw = ollama_generate(build_prompt(item))
            print(f"\n[{i}] raw:\n{raw}\n", flush=True)
            parsed = extract_json(raw)
            item["query"] = parsed.get("query", "")
        except Exception as e:
            print(f"[{i}] ERROR: {e}", file=sys.stderr, flush=True)
            item["query"] = {"error": str(e)}

        fout.write(json.dumps(item) + "\n")
        fout.flush()
