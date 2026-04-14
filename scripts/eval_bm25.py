import json
import numpy as np
from rank_bm25 import BM25Okapi

def evaluate_bm25_on_dataset(file_path):
    # 1. Load the data
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines if there are any
            if line.strip(): 
                data.append(json.loads(line))

    # 2. Prepare the Corpus and Ground Truth mappings
    documents = []
    queries_with_ground_truth = []

    for doc_id, entry in enumerate(data):
        # Extract the document text
        documents.append(entry['equipment_details'])
        
        # Map each query to its correct document ID
        for query in entry['queries']:
            # Strip out the tags from the queries if you want to evaluate pure text
            # Otherwise, BM25 might artificially match the literal word "cite"
            clean_query = query.split(".strip()")
            
            queries_with_ground_truth.append({
                'query': clean_query,
                'true_doc_id': doc_id
            })

    # 3. Tokenize and Build the BM25 Index
    print(f"Tokenizing {len(documents)} documents for the BM25 corpus...")
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    # 4. Run the Evaluation
    total_queries = len(queries_with_ground_truth)
    print(f"Evaluating {total_queries} queries...")

    reciprocal_ranks = []
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0

    for item in queries_with_ground_truth:
        tokenized_query = item['query'][0].lower().split()
        
        # Get scores for all documents in the corpus for this query
        scores = bm25.get_scores(tokenized_query)
        
        # Sort document indices by score in descending order
        ranked_doc_ids = np.argsort(scores)[::-1]
        
        # Find the rank position of the correct document (0-indexed, so we add 1)
        rank_position = np.where(ranked_doc_ids == item['true_doc_id'])[0][0]
        rank = rank_position + 1 
        
        # Calculate Metrics
        reciprocal_ranks.append(1.0 / rank)
        
        if rank == 1:
            hits_at_1 += 1
        if rank <= 3:
            hits_at_3 += 1
        if rank <= 5:
            hits_at_5 += 1

    # 5. Calculate and Print Final Metrics
    mrr = np.mean(reciprocal_ranks)
    recall_at_1 = hits_at_1 / total_queries
    recall_at_3 = hits_at_3 / total_queries
    recall_at_5 = hits_at_5 / total_queries

    print("\n--- BM25 Evaluation Results ---")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Recall@1: {recall_at_1:.4f} ({hits_at_1}/{total_queries} queries)")
    print(f"Recall@3: {recall_at_3:.4f} ({hits_at_3}/{total_queries} queries)")
    print(f"Recall@5: {recall_at_5:.4f} ({hits_at_5}/{total_queries} queries)")

# Run the evaluation
evaluate_bm25_on_dataset('data/rag_ready_data.jsonl')