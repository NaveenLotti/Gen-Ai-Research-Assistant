from app.rag.rag_chain import generate_answer


query = "What deep learning architecture is proposed for predicting the remaining useful life of Li-ion batteries?"


result = generate_answer(query)


print("\n========================================")
print("RAG ANSWER")
print("========================================")

print(result["answer"])


print("\n========================================")
print("SOURCES")
print("========================================")

for i, source in enumerate(result["sources"], start=1):

    print(f"\nSource {i}")
    print("Page:", source["page"])
    print("Content:", source["content"][:500])