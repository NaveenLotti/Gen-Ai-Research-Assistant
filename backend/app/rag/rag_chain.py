from app.rag.retriever import retrieve_documents
from app.rag.llm import get_llm


def generate_answer(query: str):

    # Retrieve relevant chunks
    documents = retrieve_documents(query, k=4)

    context_parts = []

    for i, doc in enumerate(documents, start=1):

        # PyPDFLoader uses zero-based page numbers
        pdf_page = doc.metadata.get("page", 0)

        # Convert to human-readable page number
        page_number = pdf_page + 1

        context_parts.append(
            f"""
SOURCE {i}
PAGE: {page_number}

{doc.page_content}
"""
        )

    context = "\n".join(context_parts)

    # Prompt Gemini
    prompt = f"""
You are a research paper assistant.

Answer the user's question using ONLY the provided
research-paper context.

IMPORTANT RULES:

1. Do not invent information.
2. If the answer cannot be found in the context, say:
   "I could not find this information in the provided research paper."
3. Always cite the human-readable page number.
4. Use page numbers exactly as provided in the context.
5. Do NOT use zero-based page numbers.

Research Paper Context:
-----------------------
{context}
-----------------------

User Question:
{query}

Provide a clear and concise answer.
"""

    llm = get_llm()

    response = llm.invoke(prompt)

    # Gemini may return structured content
    answer = response.content

    if isinstance(answer, list):

        text_parts = []

        for item in answer:

            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))

            elif isinstance(item, str):
                text_parts.append(item)

        answer = "\n".join(text_parts)

    # Return answer + sources
    return {
        "answer": answer,

        "sources": [
            {
                "page": doc.metadata.get("page", 0) + 1,
                "content": doc.page_content
            }
            for doc in documents
        ]
    }