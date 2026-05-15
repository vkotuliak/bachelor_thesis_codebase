## Bachelor thesis for implementation of RAG system on university equipment

This project aims to develop a Retrieval Augmented Generation (RAG) system, to make search in university database easier.

### Running the application:

The whole retrieval pipeline is contained in file `app.py`. To start it, run the following command:
```python
python app.py --model <model_name>
```
model_name can have 4 values: bm25, mpnet, e5 and hybrid (combination of bm25 and e5).
These values cannot be combined.

### Evaluation:

To evaluate these models, run the file `evaluation.py` with fillowing command:
```python
python evaluation.py --model <model_name>
```
model_name can have 4 values: bm25, mpnet, e5 and hybrid (combination of bm25 and e5). To run multiple evaluations, it is possible to combine the parameters.

### Other parts of the project:

#### data/

Files in folder `data/full_data` contain the full database, mainly used for retrieval.

Files in folder `data/test_data` contain only snippets of the full database, and should be used for testing purposes.

Files in folder `data/dense` contain dense embeddings for *mpnet* and *E5*.

#### models/

Here is a folder for different models I will gradually implement for RAG system.

#### scripts/

Complementary scripts to work with the data