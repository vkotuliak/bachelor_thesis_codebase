#!/usr/bin/env python3
"""
Generate synthetic search queries for each equipment entry in a JSONL file.
Calls the Anthropic API to produce 3 researcher-style queries per item,
then appends them to each line as a `queries` field.
"""

import json
import time
import argparse
from pathlib import Path
import anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-opus-4-5"
MAX_TOKENS = 512
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds between retries on rate-limit / server errors

PROMPT_TEMPLATE = """You are a scientist describing an experimental need, not yet knowing which device to use.
Given the equipment description below, generate 3 realistic search queries that a researcher might write when looking for a tool to fulfill this experimental need.
- Query 1: use precise technical terminology
- Query 2: describe the procedure or experimental goal in plain language
- Query 3: frame it as a problem to solve (e.g. "how do I remove solvent from a sample")
Do not include the equipment name or its aliases in any of the queries.
Equipment Name: {name}
Aliases: {aliases}
Description: {short_description}
Typical Applications: {typical_applications}
Tags: {tags}
Return ONLY a JSON object: {{"queries": ["...", "...", "..."]}}"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_prompt(item: dict) -> str:
    return PROMPT_TEMPLATE.format(
        name=item.get("Equipment Name", item.get("name", "")),
        aliases=item.get("Aliases", item.get("aliases", "")),
        short_description=item.get("Description", item.get("short_description", "")),
        typical_applications=item.get("Typical Applications", item.get("typical_applications", "")),
        tags=item.get("Tags", item.get("tags", "")),
    )


def call_claude(client: anthropic.Anthropic, prompt: str) -> list[str]:
    """Call the Anthropic API and return the list of generated queries."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            message = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()

            # Strip accidental markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)
            queries = parsed.get("queries", [])
            if not isinstance(queries, list) or len(queries) == 0:
                raise ValueError(f"Unexpected response structure: {parsed}")
            return queries

        except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
            if attempt < RETRY_ATTEMPTS:
                print(f"  [warn] API error ({e}), retrying in {RETRY_DELAY}s …")
                time.sleep(RETRY_DELAY)
            else:
                raise
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse JSON from model response: {raw!r}") from e


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_file(input_path: Path, output_path: Path) -> None:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    lines = input_path.read_text(encoding="utf-8").splitlines()
    total = len(lines)
    print(f"Processing {total} equipment entries from '{input_path}' …\n")

    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[{idx}/{total}] Skipping malformed JSON line: {e}")
                out_f.write(line + "\n")
                continue

            name = item.get("Equipment Name", item.get("name", f"item #{idx}"))
            print(f"[{idx}/{total}] Generating queries for: {name}")

            try:
                prompt = build_prompt(item)
                queries = call_claude(client, prompt)
                item["queries"] = queries
                print(f"         ✓ {queries[0][:60]}…")
            except Exception as e:
                print(f"         ✗ Failed: {e}")
                item["queries"] = []

            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDone. Output written to '{output_path}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic researcher queries for equipment JSONL entries."
    )
    parser.add_argument("input", type=Path, help="Path to the input .jsonl file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Path to the output .jsonl file (default: <input>_with_queries.jsonl)"
    )
    args = parser.parse_args()

    input_path: Path = Path("documents.jsonl")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path: Path = args.output or input_path.with_stem(input_path.stem + "_with_queries")
    process_file(input_path, output_path)


if __name__ == "__main__":
    main()