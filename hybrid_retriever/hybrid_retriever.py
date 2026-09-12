from langchain_classic.retrievers import EnsembleRetriever

from dense_retriever.dense_retriever import create_dense_retriever
from sparse_retriever.sparse_retriever import create_sparse_retriever


def create_hybrid_retriever(k: int = 3):
    if k <= 0:
        raise ValueError("k must be greater than 0")

    dense = create_dense_retriever(
        k=k,
        score_threshold=0.0,
    )

    sparse = create_sparse_retriever(k=k)

    hybrid = EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[0.5, 0.5],
    )

    return hybrid


if __name__ == "__main__":
    retriever = create_hybrid_retriever(k=3)

    query = "How does BM25 ranking work?"

    results = retriever.invoke(query)

    print("QUERY:")
    print(query)

    print("\nHYBRID RESULTS:")

    for document in results[:3]:
        print(
            f"{document.metadata['id']} | "
            f"{document.page_content}"
        )

    print(f"\nTOTAL RESULTS: {len(results[:3])}")