from langchain_qdrant import QdrantVectorStore

from app.rag.embeddings import get_embeddings


COLLECTION_NAME = "research_papers_local"
QDRANT_PATH = "./qdrant_local_data"


def create_vector_store(documents):

    embeddings = get_embeddings()

    vector_store = QdrantVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
    )

    print("Documents successfully stored in local Qdrant!")

    return vector_store


def add_documents_to_vector_store(documents):

    embeddings = get_embeddings()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
    )

    vector_store.add_documents(documents)

    print(f"Added {len(documents)} chunks to local Qdrant.")

    return vector_store