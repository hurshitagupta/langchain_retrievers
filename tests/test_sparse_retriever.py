import pytest

from sparse_retriever.sparse_retriever import create_sparse_retriever


def test_sparse_retriever_success():
    retriever = create_sparse_retriever(k=2)

    results = retriever.invoke("BM25 term frequency")

    assert len(results) > 0
    assert len(results) <= 2
    assert results[0].metadata["id"] == "doc2"


def test_sparse_retriever_invalid_k():
    with pytest.raises(ValueError):
        create_sparse_retriever(k=0)