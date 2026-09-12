import pytest

from dense_retriever.dense_retriever import create_dense_retriever


def test_dense_retriever_success():
    retriever = create_dense_retriever(
        k=3,
        score_threshold=0.0,
    )

    results = retriever.invoke(
        "semantic search using meaning"
    )

    assert len(results) > 0
    assert len(results) <= 3


def test_dense_retriever_invalid_k():
    with pytest.raises(ValueError):
        create_dense_retriever(
            k=0,
            score_threshold=0.3,
        )