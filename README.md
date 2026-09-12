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

---

## Task 5 — Benchmark

### Objective

The goal of this task is to compare the dense, sparse, and hybrid retrieval strategies using measurable results rather than opinions.

The two metrics used are:

* `recall@5` — measures whether the relevant document appears within the top 5 retrieved results.
* `p95 latency` — measures retrieval performance at the 95th percentile.

All three retrieval strategies are evaluated using the same labelled query set to keep the comparison consistent.

### Labelled Query Set

A small labelled query set is created where every query has a known relevant document.

For example:

```python id="29yvmj"
{
    "query": "BM25 term frequency",
    "relevant_id": "doc2",
}
```

This means `doc2` is considered the expected relevant document for this query.

### Recall@5

For each query, the IDs of the top five retrieved documents are checked.

```python id="lpp7ci"
retrieved_ids = [
    document.metadata["id"]
    for document in results[:5]
]
```

If the expected relevant document is present, the query receives a recall value of `1`. Otherwise, it receives `0`.

### Latency Measurement

Retrieval latency is measured using `time.perf_counter()`.

```python id="m9cm64"
start = time.perf_counter()

results = retriever.invoke(query)

end = time.perf_counter()

latency_ms = (end - start) * 1000
```

The recorded latencies are then used to calculate p95 latency.


### Benchmark Limitation

The corpus used in this assessment contains only five documents, while the metric being measured is recall@5.

Therefore, this is a small and relatively easy retrieval benchmark, which helps explain why all three strategies achieved a recall@5 of `1.0`.

A larger corpus would provide a more challenging benchmark and could show clearer differences in retrieval quality between dense, sparse, and hybrid strategies.

### Validation

The benchmark validates that latency measurements are available before calculating p95.

```python id="ebcxdf"
if not latencies:
    raise ValueError("latencies cannot be empty")
```

This prevents a percentile calculation from being performed without any measurements.

### Tests

The automated tests cover:

* **Success case** — runs the benchmark and verifies that recall@5 is between `0` and `1` and that latency is non-negative.
* **Failure case** — verifies that attempting to calculate p95 with an empty latency list raises a `ValueError`.

Run the tests:

```bash id="rg1o8u"
uv run pytest tests/test_benchmark.py -v
```

Save the test output:

```bash id="dh3rj1"
uv run pytest tests/test_benchmark.py -v > outputs/test_benchmark_output.txt 2>&1
```

### Run the Benchmark

```bash id="ymn9du"
uv run python -m benchmark.benchmark
```

### Save the Benchmark Output

```bash id="izsjxu"
uv run python -m benchmark.benchmark > outputs/benchmark.txt 2>&1
```

### Task 5 Result

The benchmark successfully:

* Uses the same labelled query set for all retrieval strategies.
* Measures recall@5.
* Measures p95 retrieval latency.
* Compares dense, sparse, and hybrid retrieval using actual numbers.
* Includes automated success and failure tests.
* Identifies the limitations of the small benchmark corpus.



