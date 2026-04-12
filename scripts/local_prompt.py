import json
from transformers import pipeline
import torch
import re

def extract_json(text: str) -> dict:
    """
    Extract the first JSON object found in the model output.
    Gemma 4 sometimes wraps output in <think>...</think> blocks or markdown fences.
    """
    # Strip <think>...</think> blocks (Gemma 4 reasoning traces)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
 
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
 
    # Find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in output: {text!r}")
    return json.loads(match.group())

access_token = "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"

# model_it = "google/gemma-4-E4B-it"
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
model_id = "Qwen/Qwen3-0.6B"

pipe = pipeline(
    "text-generation",
    model=model_id,
    token=access_token,
    model_kwargs={
                "torch_dtype": torch.bfloat16,
            },
    device_map="auto",
)

with open("data/f100lines_documents.jsonl", "r") as fin, open("data/f100lines_documents_with_queries.jsonl", "w") as fout:
    for line in fin:
        item = json.loads(line)

        prompt = f"""You are a scientist describing an experimental need, not yet knowing which device to use.
        Given the equipment description below, generate 3 realistic search queries that a researcher might write when looking for a tool to fulfill this experimental need.
        - Query 1: use precise technical terminology
        - Query 2: describe the procedure or experimental goal in plain language
        - Query 3: frame it as a problem to solve (e.g. "how do I remove solvent from a sample")
        Do not include the equipment name or its aliases in any of the queries.

        Equipment Description:
        Equipment Name: {item.get("Equipment Name", "")}
        Aliases: {item.get("Aliases", "")}
        Description: {item.get("Description", "")}
        Typical Applications: {item.get("Typical Applications", "")}
        Tags: {item.get("Tags", "")}

        Return ONLY a JSON object: {{"queries": ["...", "...", "..."]}}
        """

        messages = [{"role": "user", "content": prompt}]
        result = pipe(messages, max_new_tokens=512)
        print()
        print(f"cleaned LLM output: \n {result[0]["generated_text"][-1]["content"]}")
        print()
        fin_json = extract_json(result[0]["generated_text"][-1]["content"])
        print(fin_json)
        item["queries"] = fin_json
        print(json.dumps(item))
        fout.write(json.dumps(item) + '\n')
