from langchain_qdrant import QdrantVectorStore

from app.rag.embeddings import get_embeddings


COLLECTION_NAME = "research_papers"
QDRANT_PATH = "./qdrant_data"


def create_vector_store(documents):

    embeddings = get_embeddings()

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
    )

    print("Documents successfully stored in Qdrant!")

    return vector_store