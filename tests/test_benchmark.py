import pytest

from benchmark.benchmark import benchmark_retriever,calculate_p95

from sparse_retriever.sparse_retriever import create_sparse_retriever


def test_benchmark_success():
    retriever = create_sparse_retriever(k=5)

    result = benchmark_retriever("sparse",retriever)

    assert result["retriever"] == "sparse"

    assert 0 <= result["recall@5"] <= 1

    assert result["p95_latency_ms"] >= 0


def test_p95_empty_latencies():
    with pytest.raises(ValueError):
        calculate_p95([])