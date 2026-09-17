from app.rag.retriever import get_vector_store
from app.rag.llm import get_llm


def comparison_agent(query: str):

    vector_store = get_vector_store()

    # Retrieve a larger number of chunks because
    # multiple papers need to be represented.
    documents = vector_store.similarity_search(
        query,
        k=20
    )

    if not documents:
        return {
            "answer": "No relevant information was found in the uploaded papers.",
            "agent": "COMPARISON",
            "sources": []
        }

    # Group retrieved chunks by paper
    papers = {}

    for doc in documents:

        paper_id = doc.metadata.get(
            "paper_id",
            "unknown"
        )

        paper_name = doc.metadata.get(
            "paper_name",
            "Unknown Paper"
        )

        page = doc.metadata.get(
            "page",
            0
        ) + 1

        if paper_id not in papers:

            papers[paper_id] = {
                "paper_name": paper_name,
                "chunks": []
            }

        papers[paper_id]["chunks"].append({
            "page": page,
            "content": doc.page_content
        })

    # Build context
    context_parts = []

    for paper_id, paper in papers.items():

        context_parts.append(
            f"\n===== PAPER: {paper['paper_name']} =====\n"
        )

        for chunk in paper["chunks"]:

            context_parts.append(
                f"""
PAGE: {chunk['page']}

{chunk['content']}
"""
            )

    context = "\n".join(context_parts)

    # LLM prompt
    prompt = f"""
You are an expert research-paper comparison assistant.

The user wants to compare research papers.

USER REQUEST:
{query}

Use ONLY the information contained in the
provided research-paper context.

RESEARCH PAPER CONTEXT:
================================

{context}

================================

IMPORTANT RULES:

1. Do not invent information.

2. Do not assume information that is not
   explicitly present in the context.

3. Identify papers using their filenames.

4. Cite the page number when presenting
   important factual information.

5. Page numbers are human-readable.
   Page 1 means the first page of the PDF.

6. If only one paper is represented in the
   retrieved context, clearly state that a
   complete paper-to-paper comparison cannot
   be performed.

7. Distinguish between information that is
   explicitly available and information that
   is unavailable.

8. Do not create results, datasets, models,
   metrics, or conclusions that are not present.

Structure the response as follows:

## Comparison

### Paper 1

- Objective
- Methodology
- Model / Architecture
- Dataset
- Results
- Limitations

### Paper 2

- Objective
- Methodology
- Model / Architecture
- Dataset
- Results
- Limitations

### Key Differences

Explain the factual differences supported
by the retrieved evidence.

### Similarities

Explain the factual similarities supported
by the retrieved evidence.

### Overall Comparison

Provide a concise factual synthesis.

Include page citations such as:

(Page 5)

or

(Page 5, Page 8)
"""

    llm = get_llm()

    response = llm.invoke(prompt)

    answer = response.content

    # Gemini may return structured content
    if isinstance(answer, list):

        text_parts = []

        for item in answer:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text_parts.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):

                text_parts.append(item)

        answer = "\n".join(text_parts)

    # Build source list
    sources = []

    for paper_id, paper in papers.items():

        for chunk in paper["chunks"]:

            sources.append({
                "paper_id": paper_id,
                "paper_name": paper["paper_name"],
                "page": chunk["page"],
                "content": chunk["content"]
            })

    return {
        "answer": answer,
        "agent": "COMPARISON",
        "sources": sources
    }