from app.rag.retriever import retrieve_documents


queries = [
    "What is the objective of the research?",
    "What dataset was used?",
    "What machine learning or deep learning model was used?",
]


for query in queries:

    print("\n" + "=" * 70)
    print("QUERY:", query)
    print("=" * 70)

    documents = retrieve_documents(query, k=5)

    print("Retrieved documents:", len(documents))

    for i, doc in enumerate(documents, start=1):

        paper_name = doc.metadata.get(
            "paper_name",
            "Unknown Paper"
        )

        page = doc.metadata.get(
            "page",
            0
        ) + 1

        print(f"\n--- Result {i} ---")
        print("Paper:", paper_name)
        print("Page:", page)
        print("Content:")
        print(doc.page_content[:500])