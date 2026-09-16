from langchain_qdrant import QdrantVectorStore
from app.rag.embeddings import get_embeddings


COLLECTION_NAME = "research_papers"
QDRANT_PATH = "./qdrant_data"


def get_vector_store():

    embeddings = get_embeddings()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
    )

    return vector_store


def retrieve_documents(query: str, k: int = 4):

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        query,
        k=k
    )

    return documents