import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


load_dotenv()

DOCUMENTS = [
    Document(
        page_content="Vector search finds documents using semantic similarity.",
        metadata={"id": "doc1", "source": "guide", "tag": "dense"},
    ),
    Document(
        page_content="BM25 ranks documents using matching words and term frequency.",
        metadata={"id": "doc2", "source": "guide", "tag": "sparse"},
    ),
    Document(
        page_content="Hybrid retrieval combines dense and sparse search results.",
        metadata={"id": "doc3", "source": "blog", "tag": "hybrid"},
    ),
    Document(
        page_content="Metadata filters restrict retrieval using document properties.",
        metadata={"id": "doc4", "source": "blog", "tag": "filter"},
    ),
    Document(
        page_content="Recall measures how many relevant documents were successfully retrieved.",
        metadata={"id": "doc5", "source": "notes", "tag": "benchmark"},
    ),
]


def create_dense_retriever(k: int = 3, score_threshold: float = 0.3):
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be between 0 and 1")

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
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": k,
            "score_threshold": score_threshold,
        },
    )

    return retriever


if __name__ == "__main__":
    retriever = create_dense_retriever(
        k=3,
        score_threshold=0.3,
    )

    query = "How can I search documents by meaning?"

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