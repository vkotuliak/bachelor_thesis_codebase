import json
import re
import sys
import os
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

INPUT_FILE = os.environ.get("INPUT_FILE", "data/rag_documents.jsonl")
OUTPUT_FILE = os.environ.get(
    "OUTPUT_FILE", "data/documents_with_queries.jsonl"
)

# New: Define distinct personas to ensure data diversity
PERSONAS = [
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


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in: {text!r}")
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


def build_prompt(item: dict, persona: dict) -> str:
    return f"""You generate search queries for the University of Groningen equipment portal.
    
The user persona is: **{persona['name']}** ({persona['description']})

Rules:
1. Do NOT include the equipment name or its aliases.
2. Use first-person, problem-driven language (e.g., "I am trying to...", "Our project is stuck because...").
3. Avoid "academic-speak" unless you are the Researcher persona.
4. Focus on the *scientific problem* the equipment solves, not the machine itself.

<document>
Name: {item.get("name", "")}
Short Description: {item.get("short description", "")}
Typical Applications: {item.get("typical applications", "")}
</document>

Return ONLY a JSON object: 
{{
  "thought": "Briefly explain the specific scientific problem this persona would have that this equipment solves.",
  "query": "The actual search query"
}}"""


os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(INPUT_FILE, "r") as fin, open(OUTPUT_FILE, "w") as fout:
    for i, line in enumerate(fin):
        item = json.loads(line)

        for persona in PERSONAS:
            try:
                prompt = build_prompt(item, persona)
                raw = ollama_generate(prompt)
                parsed = extract_json(raw)

                # Create a new entry for every query-doc pair
                new_entry = {
                    "equipment_id": item.get("id", i),  # Assume you have an ID
                    "document": f"Name: {item.get('name')}. {item.get('short description')}",
                    "query": parsed.get("query", ""),
                    "persona": persona["name"],
                }
                fout.write(json.dumps(new_entry) + "\n")
                fout.flush()
                print(f"[{i}] Generated {persona['name']} query.")
            except Exception as e:
                print(
                    f"[{i}] ERROR with {persona['name']}: {e}", file=sys.stderr
                )

print("Done! Check your output file for the expanded dataset.")
