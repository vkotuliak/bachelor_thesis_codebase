## Bachelor thesis for implementation of RAG system on university equipment

Bachelor thesis (BSc Artificial Intelligence, University of Groningen, 2026). This project compares sparse (BM25), dense (all-mpnet, E5) and hybrid retrieval for a RAG system that helps researchers find equipment in the university's 8,587-item database. Systems are evaluated on 2,500 synthetic queries (500 items × 5 persona-based queries) with Recall@k and nDCG@k.

**Key result:** a BM25 + E5 hybrid with weighted reciprocal rank fusion performed best.

| System        | nDCG@10 | Recall@10 |
|---------------|---------|-----------|
| BM25 + E5 hybrid (weighted RRF) | **0.68** | **0.83** |
| BM25          | 0.66    | 0.78      |
| E5            | 0.57    | 0.71      |
| all-mpnet     | 0.47    | 0.63      |

Full thesis: [thesis.pdf](thesis.pdf)

### Running the application:

The whole retrieval pipeline is contained in file `app.py`. To start it, run the following command:
```
python app.py --model <model_name>
```
model_name can have 4 values: bm25, mpnet, e5 and hybrid (combination of bm25 and e5).
These values cannot be combined.

To utilize generator add a flag `--generate`:
```
python app.py --model <model_name> --generate
```

It is possible to change number of results given with flag `--k`:
```
python app.py --model <model_name> --k 10
```

### Evaluation:

To evaluate these models, run the file `evaluation.py` with following command:
```
python evaluation.py --models <model_name>
```
model_name can have 5 values: bm25, mpnet, e5 and hybrid (combination of bm25 and e5). To run multiple evaluations, it is possible to combine the parameters.

It is possible to change k in recall and nDCG evaluation with flag `--ks`
```
python evaluation.py --models <model_name> --ks 5 10
```

### data/

#### data/full_data
Files in folder `data/full_data` contain the full database, mainly used for retrieval.

- `equipment_database.csv`: contains original uncleaned database in csv format
- `documents.jsonl`: contains cleaned database with only specific fields in jsonl format
- `rag_documents.jsonl`: contains everything from `documents.jsonl` but in format ideal for RAG retrieval

#### data/test_data
Files in folder `data/test_data` contain only snippets of the full database, and should be used for testing purposes.

- `queries_from_personas`: contains 500 pieces of equipment with 5 queries each. These can be used for extensive testing
- `f100_queries`: smaller part of `queries_from_personas`. Used for light-weight and faster testing

#### data/dense
Files in folder `data/dense` contain dense embeddings for *mpnet* and *E5*.

### build/

Directory for building embeddings for *E5* and *all-mpnet*

### scripts/

1. `s01_build_documents.py`: Cleans the data from csv and creates two jsonl files
2. `s02_visualise_data.py`: Visualise percentage of completeness for columns in the csv dataset.
3. `s04_gen_w_example.py`: Original generation file, used to generate queries using example from instruction sheet. Deprecated now.
4. `s05_gen_w_persona.py`: New generation file, taken from Ge, Tao, et al. (2024).
