from app.rag.embeddings import get_embeddings


print("Loading local embedding model...")

embeddings = get_embeddings()

print("Embedding model loaded!")

text = "This is a research paper about lithium-ion battery remaining useful life."

vector = embeddings.embed_query(text)

print("Embedding generated successfully!")

print("Vector dimensions:", len(vector))

print("First 10 values:")
print(vector[:10])