## Task 1 — Dense Retriever

### Objective

The goal of this task is to implement a dense retriever using vector search with:

* Tunable `k`
* Tunable `score_threshold`
* Input validation
* Automated success and failure tests

The dense retriever searches documents based on semantic similarity rather than exact keyword matching.

### Implementation

The implementation uses:

* `Document` from LangChain for storing text and metadata
* `OpenAIEmbeddings` for converting documents and queries into vector embeddings
* `FAISS` as the vector store
* LangChain's retriever interface for performing the search

A common document corpus is created so that the same documents can later be reused for sparse, hybrid, metadata filter, and benchmark tasks.

The retriever is created using:

```python
vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": k,
        "score_threshold": score_threshold,
    },
)
```

Here:

* `k` controls the maximum number of documents returned.
* `score_threshold` controls the minimum relevance required for a document to be included.

The values are passed as function arguments, which makes both parameters tunable.

### Validation

Basic input validation is added before creating the retriever.

```python
if k <= 0:
    raise ValueError("k must be greater than 0")

if not 0 <= score_threshold <= 1:
    raise ValueError("score_threshold must be between 0 and 1")
```

This prevents invalid retriever configurations from being used.

### Run the Task

From the project root:

```bash
uv run python -m dense_retriever.dense_retriever
```

### Save the Output

```bash
uv run python -m dense_retriever.dense_retriever > outputs/dense_retriever.txt 2>&1
```

### Tests

The automated tests cover:

* **Success case** — verifies that the dense retriever returns documents and respects the configured `k`.
* **Failure case** — verifies that an invalid value such as `k=0` raises a `ValueError`.

Run the tests using:

```bash
uv run pytest tests/test_dense_retriever.py -v
```

Save the pytest output using:

```bash
uv run pytest tests/test_dense_retriever.py -v > outputs/test_dense_retriever.txt 2>&1
```

### Task 1 Result

The dense retriever successfully:

* Converts documents into embeddings.
* Stores document vectors using FAISS.
* Retrieves documents using semantic similarity.
* Supports configurable `k`.
* Supports configurable similarity score threshold.
* Validates invalid configuration values.
* Includes automated success and failure tests.

This completes the Dense Retriever requirement for Task 1.
