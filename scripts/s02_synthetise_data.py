import pyterrier as pt
pt.init()
from pyterrier_doc2query import Doc2Query
# import pyterrier_doc2query
import json

# doc2query = pyterrier_doc2query.Doc2Query()
doc2query = Doc2Query()
file_path = "data/documents.jsonl"

documents = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data = json.loads(line)
            documents.append({"docno": data["id"], "text": data["document"]})
        except json.JSONDecodeError as e:
            print(f"Error parsing line: {line.strip()}")
            print(e)

if documents:
    results = list(doc2query(documents))
    # Save the results to a new file
    with open("data/synthesized_documents.jsonl", 'w', encoding='utf-8') as outfile:
        for result in results:
            json.dump(result, outfile)
            outfile.write('\n')
    print("Synthesized data saved to data/synthesized_documents.jsonl")
else:
    print("No documents to process")