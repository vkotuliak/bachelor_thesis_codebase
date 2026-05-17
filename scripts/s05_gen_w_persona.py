import json
import re
import sys
import os
import urllib.request
from difflib import SequenceMatcher

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
# MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")

INPUT_FILE = os.environ.get("INPUT_FILE", "data/test_data/w_queries_500.jsonl")
OUTPUT_FILE = os.environ.get(
    "OUTPUT_FILE", "data/test_data/personas_testing.jsonl"
)

# Fallback personas used if grounded persona generation fails for a document
FALLBACK_PERSONAS = [
    {
        "name": "Industry Partner",
        "description": "A quality control manager at a manufacturing company dealing with a product failure or material inconsistency.",
    },
    {
        "name": "Academic Researcher",
        "description": "A PhD student needing to verify a theoretical property or get high-resolution data for a paper.",
    },
    {
        "name": "Plain Language User",
        "description": "Someone who knows their problem (e.g., 'the glue won't stick') but knows zero technical terminology.",
    },
]

DEDUP_THRESHOLD = (
    0.8  # similarity ratio above which a query is considered a duplicate
)


# utils
def extract_json_object(text: str) -> dict:
    """Extract a single JSON object from a raw LLM response."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in: {text!r}")
    return json.loads(match.group())


def extract_json_array(text: str) -> list:
    """Extract a JSON array from a raw LLM response."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in: {text!r}")
    return json.loads(match.group())


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


def is_duplicate(
    new_query: str, seen: list[str], threshold: float = DEDUP_THRESHOLD
) -> bool:
    """Return True if new_query is too similar to any query already in seen."""
    new_lower = new_query.lower()
    for q in seen:
        ratio = SequenceMatcher(None, new_lower, q.lower()).ratio()
        if ratio > threshold:
            return True
    return False


# Text-to-Persona: derive grounded personas from the document itself
def build_persona_gen_prompt(item: dict) -> str:
    return f"""You are helping build a search query test set for a university scientific equipment portal.

Given the equipment below, list exactly 5 distinct types of people who might search for it.
Describe each person by their *specific problem or goal*, not their job title.
The descriptions should be concrete and varied - different industries, career stages, and levels of technical knowledge.

<document>
Name: {item.get("name", "")}
Short Description: {item.get("short description", "")}
Typical Applications: {item.get("typical applications", "")}
Research-Oriented Use: {item.get("research-oriented use", "")}
Tags: {item.get("tags", "")}
</document>

Return ONLY a JSON array with exactly 5 objects:
[
  {{"name": "short label for this persona type", "description": "their specific problem or goal in 1-2 sentences"}},
  ...
]"""


def generate_personas(item: dict, doc_index: int) -> list[dict]:
    """
    Run Text-to-Persona for one document.
    Falls back to FALLBACK_PERSONAS if generation fails.
    """
    try:
        prompt = build_persona_gen_prompt(item)
        raw = ollama_generate(prompt)
        personas = extract_json_array(raw)

        # Validate structure — keep only well-formed entries
        valid = [
            p
            for p in personas
            if isinstance(p, dict) and "name" in p and "description" in p
        ]
        if len(valid) < 2:
            raise ValueError(f"Too few valid personas returned ({len(valid)})")

        print(f"  [{doc_index}] Generated {len(valid)} grounded personas.")
        return valid

    except Exception as e:
        print(
            f"  [{doc_index}] Persona generation failed, using fallback: {e}",
            file=sys.stderr,
        )
        return FALLBACK_PERSONAS


# query generation (PersonaHub-style immersive prompt)
def build_query_prompt(item: dict, persona: dict) -> str:
    return f"""You are {persona['name']} with following problem/goal:

{persona['description']}

You have never heard of the equipment described below, but you have the problem above.
Write the search query you would actually type into a university equipment portal search box.

Rules:
1. Do NOT include the equipment name or any technical instrument names.
2. Use first-person, problem-driven language (e.g., "I need to...", "We are trying to...").
3. Write at the vocabulary level of the persona — avoid jargon unless it fits their background.
4. Focus on the scientific or practical problem, not on the solution.

<document>
Name: {item.get("name", "")}
Short Description: {item.get("short description", "")}
Typical Applications: {item.get("typical applications", "")}
Research-Oriented Use: {item.get("research-oriented use", "")}
Tags: {item.get("tags", "")}
</document>

Return ONLY a JSON object:
{{"query": "The actual search query string"}}
"""


# main pipeline
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

total_written = 0
total_skipped_duplicate = 0
total_errors = 0

with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout:
    for i, line in enumerate(fin):
        item = json.loads(line)
        print(f"\n[{i}] Processing: {item.get('name', '(unnamed)')}")

        # get the personas
        personas = generate_personas(item, i)

        seen_queries: list[str] = []
        queries_list = []

        # generate one query per persona
        for persona in personas:
            try:
                prompt = build_query_prompt(item, persona)
                raw = ollama_generate(prompt)
                parsed = extract_json_object(raw)
                query = parsed.get("query", "").strip()

                if not query:
                    raise ValueError("Empty query returned.")

                # deduplicate
                if is_duplicate(query, seen_queries):
                    print(
                        f"  [{i}] SKIPPED duplicate query for persona '{persona['name']}'."
                    )
                    total_skipped_duplicate += 1
                    continue

                seen_queries.append(query)

                queries_list.append(query)
                print(
                    f"  [{i}] ✓ Query written for persona '{persona['name']}'."
                )

            except Exception as e:
                total_errors += 1
                print(
                    f"  [{i}] ERROR for persona '{persona['name']}': {e}",
                    file=sys.stderr,
                )

        if queries_list:
            new_entry = {
                "equipment_id": item.get("id", i),
                "name": item.get("name"),
                "query": queries_list,
            }
            fout.write(json.dumps(new_entry) + "\n")
            fout.flush()
            total_written += len(queries_list)
        else:
            print(f"  [{i}] WARNING: No queries generated, skipping entry.")

print(
    f"\nDone. Written: {total_written} | "
    f"Duplicates skipped: {total_skipped_duplicate} | "
    f"Errors: {total_errors}"
)
