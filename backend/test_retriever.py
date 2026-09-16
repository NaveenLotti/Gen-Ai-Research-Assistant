from app.rag.retriever import retrieve_documents


query = "What deep learning architecture is proposed for predicting the remaining useful life of Li-ion batteries?"


documents = retrieve_documents(query, k=4)


print("\n========================================")
print("RETRIEVED RESEARCH PAPER CHUNKS")
print("========================================")


for i, doc in enumerate(documents, start=1):

    print(f"\n---------- RESULT {i} ----------")

    print("\nPage:", doc.metadata.get("page"))

    print("\nContent:")
    print(doc.page_content[:1500])
    