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
uv run python -m dense_retriever.dense_retriever > outputs/dense_retriever_output.txt 2>&1
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

## Task 2 — Sparse Retriever

### Objective

The goal of this task is to implement a sparse retriever using BM25 over the same document corpus used by the dense retriever.

Unlike dense retrieval, which searches using semantic similarity, BM25 mainly ranks documents based on matching terms between the query and the document.

### Implementation

The implementation uses LangChain's `BM25Retriever`.

```python
retriever = BM25Retriever.from_documents(DOCUMENTS)
retriever.k = k
```

The same `DOCUMENTS` corpus from Task 1 is reused so that dense and sparse retrieval can later be compared fairly.

The `k` value is configurable and controls the maximum number of documents returned by the retriever.

### Validation

Basic validation is added to prevent invalid `k` values:

```python
if k <= 0:
    raise ValueError("k must be greater than 0")
```

This provides a clear failure path instead of allowing an invalid retriever configuration.

### Run the Task

```bash
uv run python -m sparse_retriever.sparse_retriever
```

### Save the Output

```bash
uv run python -m sparse_retriever.sparse_retriever > outputs/sparse_retriever_output.txt 
```

### Tests

The automated tests cover:

* **Success case** — verifies that results are returned, `k` is respected, and the expected BM25 document is ranked first.
* **Failure case** — verifies that `k=0` raises a `ValueError`.

Run the tests:

```bash
uv run pytest tests/test_sparse_retriever.py -v
```

Save the test output:

```bash
uv run pytest tests/test_sparse_retriever.py -v > outputs/test_sparse_retriever.txt 2>&1
```

### Task 2 Result

The sparse retriever successfully:

* Uses BM25 for lexical retrieval.
* Reuses the same corpus as the dense retriever.
* Ranks documents based on matching terms.
* Supports configurable `k`.
* Validates invalid `k` values.
* Includes automated success and failure tests.

This completes the Sparse Retriever requirement for Task 2.

