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

---

## Task 3 — Hybrid Fusion

### Objective

The goal of this task is to combine the dense and sparse retrievers created in the previous tasks into one hybrid retrieval strategy.

The hybrid retriever uses both:

* Dense retrieval for semantic similarity.
* Sparse BM25 retrieval for keyword-based matching.

### Implementation

The implementation uses LangChain's `EnsembleRetriever`.

The dense retriever from Task 1 and sparse retriever from Task 2 are reused instead of creating new retrieval logic.

Both retrievers are then passed to `EnsembleRetriever`:

```python
hybrid = EnsembleRetriever(
    retrievers=[dense, sparse],
    weights=[0.6, 0.4],
)
```
### Validation

The function validates the value of `k` before creating the retriever.

```python
if k <= 0:
    raise ValueError("k must be greater than 0")
```

This prevents invalid retriever configurations.

### Run the Task

```bash
uv run python -m hybrid_retriever.hybrid_retriever
```

### Save the Output

```bash
uv run python -m hybrid_retriever.hybrid_retriever > outputs/hybrid_retriever_output.txt 2>&1
```

### Tests

The automated tests cover:

* **Success case** — verifies that the hybrid retriever returns documents and includes the expected relevant document.
* **Failure case** — verifies that an invalid `k` value raises a `ValueError`.

Run the tests:

```bash
uv run pytest tests/test_hybrid_retriever.py -v
```

Save the pytest output:

```bash
uv run pytest tests/test_hybrid_retriever.py -v > outputs/test_hybrid_retriever.txt 2>&1
```

### Task 3 Result

The hybrid retriever successfully:

* Reuses the dense retriever from Task 1.
* Reuses the BM25 retriever from Task 2.
* Combines both using `EnsembleRetriever`.
* Uses configurable weights for the two retrieval strategies.
* Validates invalid `k` values.
* Includes automated success and failure tests.

---

## Task 4 — Metadata Filters

### Objective

The goal of this task is to restrict retrieval results using document metadata and prove that the applied filter is respected.

For this task, the `source` metadata field is used as the filter.

### Implementation

The same document corpus from the previous tasks is reused. Each document already contains metadata such as:

```python
metadata={
    "id": "doc3",
    "source": "blog",
    "tag": "hybrid"
}
```

A FAISS vector store is created using the documents and embeddings.

The metadata filter is then passed through `search_kwargs`:

```python
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": k,
        "filter": {"source": source},
    }
)
```

For example, when:

```python
source = "blog"
```

the retriever only returns documents whose metadata contains:

```python
{"source": "blog"}
```

This allows vector retrieval to be restricted to a specific group of documents.


### Run the Task

```bash
uv run python -m metadata_filters.metadata_filters
```

### Save the Output

```bash
uv run python -m metadata_filters.metadata_filters > outputs/metadata_filters_output.txt 2>&1
```

### Tests

The automated tests cover:

* **Success case** — verifies that results are returned and every returned document has `source="blog"`.
* **Failure case** — verifies that an empty source raises a `ValueError`.

Run the tests:

```bash
uv run pytest tests/test_metadata_filters.py -v
```

Save the pytest output:

```bash
uv run pytest tests/test_metadata_filters.py -v > outputs/test_metadata_filters.txt 2>&1
```

### Task 4 Result

The metadata filter implementation successfully:

* Reuses the existing document corpus.
* Performs vector-based retrieval.
* Filters documents using the `source` metadata field.
* Supports configurable `source` and `k`.
* Prevents empty filter values.
* Proves that every returned document matches the requested metadata.
* Includes automated success and failure tests.



