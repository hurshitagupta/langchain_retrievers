import pytest

from metadata_filters.metadata_filters import create_filtered_retriever


def test_metadata_filter_success():
    retriever = create_filtered_retriever(source="blog",k=3)

    results = retriever.invoke( "retrieval methods and filtering")

    assert len(results) > 0

    for document in results:
        assert document.metadata["source"] == "blog"


def test_metadata_filter_empty_source():
    with pytest.raises(ValueError):
        create_filtered_retriever(source="",k=3)