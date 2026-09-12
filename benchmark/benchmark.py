import time
import statistics

from dense_retriever.dense_retriever import create_dense_retriever
from sparse_retriever.sparse_retriever import create_sparse_retriever
from hybrid_retriever.hybrid_retriever import create_hybrid_retriever


LABELLED_QUERIES = [
    {
        "query": "semantic vector search",
        "relevant_id": "doc1",
    },
    {
        "query": "BM25 term frequency",
        "relevant_id": "doc2",
    },
    {
        "query": "combine dense and sparse retrieval",
        "relevant_id": "doc3",
    },
    {
        "query": "filter documents using metadata",
        "relevant_id": "doc4",
    },
    {
        "query": "measure retrieval recall",
        "relevant_id": "doc5",
    },
]


def calculate_recall(results, relevant_id: str) -> int:
    retrieved_ids = [
        document.metadata["id"]
        for document in results[:5]
    ]

    if relevant_id in retrieved_ids:
        return 1

    return 0


def calculate_p95(latencies: list[float]) -> float:
    if not latencies:
        raise ValueError("latencies cannot be empty")

    sorted_latencies = sorted(latencies)

    index = int(0.95 * len(sorted_latencies)) - 1
    index = max(index, 0)

    return sorted_latencies[index]


def benchmark_retriever(name, retriever):
    recalls = []
    latencies = []

    for item in LABELLED_QUERIES:
        query = item["query"]
        relevant_id = item["relevant_id"]

        start = time.perf_counter()

        results = retriever.invoke(query)

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        recall = calculate_recall(
            results,
            relevant_id,
        )

        recalls.append(recall)
        latencies.append(latency_ms)

    recall_at_5 = sum(recalls) / len(recalls)

    p95_latency = calculate_p95(latencies)

    return {
        "retriever": name,
        "recall@5": round(recall_at_5, 2),
        "p95_latency_ms": round(p95_latency, 2),
    }


if __name__ == "__main__":

    dense = create_dense_retriever(
        k=5,
        score_threshold=0.0,
    )

    sparse = create_sparse_retriever(k=5)

    hybrid = create_hybrid_retriever(k=5)

    retrievers = [
        ("dense", dense),
        ("sparse", sparse),
        ("hybrid", hybrid),
    ]

    print("BENCHMARK RESULTS\n")

    for name, retriever in retrievers:
        result = benchmark_retriever(name,retriever)
        print(result)