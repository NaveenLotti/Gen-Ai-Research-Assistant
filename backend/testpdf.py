from app.rag.loader import load_pdf
from app.rag.chunker import split_documents


PDF_PATH = "../data/papers/A_deep_learning_approach_to_optimize_remaining_use (1).pdf"


# Step 1: Load PDF
documents = load_pdf(PDF_PATH)

print("\n========== FIRST PAGE ==========")
print(documents[0].page_content[:1000])


# Step 2: Split into chunks
chunks = split_documents(documents)

print("\n========== FIRST CHUNK ==========")
print(chunks[0].page_content)

print("\n========== CHUNK METADATA ==========")
print(chunks[0].metadata)