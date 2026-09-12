import pytest

from hybrid_retriever.hybrid_retriever import create_hybrid_retriever


def test_hybrid_retriever_success():
    retriever = create_hybrid_retriever(k=3)

    results = retriever.invoke("BM25 ranking")

    assert len(results) > 0

    ids = [document.metadata["id"] for document in results]

    assert "doc2" in ids


def test_hybrid_retriever_invalid_k():
    with pytest.raises(ValueError):
        create_hybrid_retriever(k=0)