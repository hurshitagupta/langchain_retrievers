from langchain_community.retrievers import BM25Retriever

from dense_retriever.dense_retriever import DOCUMENTS


def create_sparse_retriever(k: int = 3):
    if k <= 0:
        raise ValueError("k must be greater than 0")

    retriever = BM25Retriever.from_documents(DOCUMENTS)
    retriever.k = k

    return retriever


if __name__ == "__main__":
    retriever = create_sparse_retriever(k=3)

    query = "BM25 term frequency"

    results = retriever.invoke(query)

    print("QUERY:")
    print(query)

    print("\nRESULTS:")

    for document in results:
        print(
            f"{document.metadata['id']} | "
            f"{document.page_content}"
        )

    print(f"\nTOTAL RESULTS: {len(results)}")