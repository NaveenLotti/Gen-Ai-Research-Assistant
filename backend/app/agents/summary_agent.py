from app.rag.retriever import retrieve_documents
from app.rag.llm import get_llm


def summary_agent(query: str):

    documents = retrieve_documents(
        "research paper objective methodology model results conclusion limitations",
        k=8
    )

    context_parts = []

    for doc in documents:

        page = doc.metadata.get("page", 0) + 1

        context_parts.append(
            f"""
PAGE {page}

{doc.page_content}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are a research paper summarization agent.

Summarize the research paper using ONLY the provided context.

Include the following sections:

1. Research Objective
2. Problem Statement
3. Methodology
4. Model Architecture
5. Dataset
6. Results
7. Limitations
8. Conclusion

Rules:

- Do not invent information.
- If information is unavailable, say so.
- Use human-readable page numbers.
- Keep the summary technically accurate.
- Make the explanation easy to understand.

Research Paper Context:
-----------------------
{context}
-----------------------

User Request:
{query}

Provide a structured summary.
"""

    llm = get_llm()

    response = llm.invoke(prompt)

    answer = response.content

    if isinstance(answer, list):

        text_parts = []

        for item in answer:

            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))

            elif isinstance(item, str):
                text_parts.append(item)

        answer = "\n".join(text_parts)

    return {
        "answer": answer,
        "agent": "SUMMARY",
        "sources": [
            {
                "page": doc.metadata.get("page", 0) + 1,
                "content": doc.page_content
            }
            for doc in documents
        ]
    }