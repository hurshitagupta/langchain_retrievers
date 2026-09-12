import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from dense_retriever.dense_retriever import DOCUMENTS


load_dotenv()


def create_filtered_retriever(source: str, k: int = 3):
    if not source.strip():
        raise ValueError("source cannot be empty")

    if k <= 0:
        raise ValueError("k must be greater than 0")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )

    vector_store = FAISS.from_documents(
        DOCUMENTS,
        embeddings,
    )

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": k,
            "filter": {"source": source},
        }
    )

    return retriever


if __name__ == "__main__":
    source = "blog"
    query = "retrieval methods and document filtering"

    retriever = create_filtered_retriever(
        source=source,
        k=3,
    )

    results = retriever.invoke(query)

    print("QUERY:")
    print(query)

    print("\nFILTER:")
    print(f"source = {source}")

    print("\nRESULTS:")

    for document in results:
        print(
            f"{document.metadata['id']} | "
            f"source={document.metadata['source']} | "
            f"{document.page_content}"
        )

    print(f"\nTOTAL RESULTS: {len(results)}")