from app.rag.loader import load_pdf
from app.rag.chunker import split_documents
from app.rag.vectorstore import create_vector_store


PDF_PATH = "../data/papers/A_deep_learning_approach_to_optimize_remaining_use (1).pdf"


# Load PDF
documents = load_pdf(PDF_PATH)

# Split PDF
chunks = split_documents(documents)

# Create vector database
vector_store = create_vector_store(chunks)

print("\n================================")
print("VECTOR DATABASE READY!")
print("================================")